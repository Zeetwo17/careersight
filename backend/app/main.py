"""FastAPI server for CareerSight.

Endpoints:
  GET  /api/health
  GET  /api/demo_profile/{key}     -- score a pre-built persona (priya/arjun/meera/rahul)
  POST /api/predict                -- score an arbitrary StudentProfile JSON body
  POST /api/score_from_resume      -- upload a PDF, get the full risk surface (one-shot JSON)
  POST /api/score_from_resume_stream -- SSE forensic flow (validity -> parse -> IPR -> score)
  GET  /api/portfolio              -- lender portfolio view
  GET  /api/pri                    -- PlacementRisk Index history
  GET  /                           -- serve the dashboard SPA
  GET  /static/*                   -- frontend assets
"""
from __future__ import annotations

import os
import sys
import time
import types
import warnings
from pathlib import Path
from typing import Any

import warnings
warnings.filterwarnings("ignore")

# OpenMP duplicate-lib workaround — Anaconda + LightGBM both ship libomp.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# torch is required by netcal at IMPORT time (its top-level Beta calibration
# class definition imports torch). On some Windows environments torch fails
# at DLL load. We only need torch for *training*, never for inference on
# already-fitted Beta calibrators. If torch can't be imported, install a stub
# so netcal — and therefore the bundle.joblib — can unpickle.
try:
    import torch  # noqa: F401
except (ImportError, OSError):
    _stub = types.ModuleType("torch")
    _stub.Tensor = type("Tensor", (), {})
    _stub.tensor = lambda *a, **k: None
    _stub.from_numpy = lambda *a, **k: None
    _stub.no_grad = lambda *a, **k: (lambda f: f)
    _nn = types.ModuleType("torch.nn")
    _nn.Module = type("Module", (), {})
    _stub.nn = _nn
    sys.modules["torch"] = _stub
    sys.modules["torch.nn"] = _nn
    # pycox / torchtuples follow the same fate but they're only needed for
    # the DeepHit survival path — predict.py degrades to interpolated curves.
    for _m in ("torchtuples", "pycox", "pycox.models"):
        sys.modules.setdefault(_m, types.ModuleType(_m))

# Load .env from the project root so GEMINI_API_KEY (and any other secrets)
# are available in os.environ before any module imports them.
try:
    from dotenv import load_dotenv
    _ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
    if _ENV_PATH.exists():
        load_dotenv(_ENV_PATH)
except ImportError:
    pass

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .demo_profiles import DEMO_PROFILES
from .portfolio import get_portfolio, get_pri_history
from .predict import predict_profile
from .resume_parser import parse_resume_pdf, parse_resume_text
from .schema import COURSE_TYPES, StudentProfile

app = FastAPI(
    title="CareerSight API",
    description="AI placement-risk intelligence for education lenders",
    version="1.0.0",
)


def _warm_bundle():
    """Load bundle + run one dummy SHAP call so LightGBM's native SHAP
    C++ code is JIT-warmed before the first real user request."""
    import numpy as _np
    from .predict import _bundle
    try:
        b = _bundle()
        n_feats = len(b["feature_names"])
        dummy = _np.zeros((1, n_feats), dtype=_np.float64)
        b["classifiers"]["placed_6m"].booster_.predict(dummy, pred_contrib=True)
    except Exception:
        pass


@app.on_event("startup")
async def _preload_bundle():
    """Eagerly load the model bundle and pre-warm SHAP at server startup
    so the first user request pays no cold-start cost."""
    import asyncio as _asyncio
    loop = _asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, _warm_bundle)
    except Exception:
        pass  # FileNotFoundError surfaced properly on first /api/health call


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProfileBody(BaseModel):
    name: str = "Anonymous"
    course_type: str = Field(default="BTech-CS")
    institute_name: str = "Tier-2 Institute"
    institute_tier: int = Field(default=2, ge=1, le=3)
    region: str = "Tier2"
    graduation_year: int = 2024
    cgpa: float = Field(default=7.0, ge=0, le=10)
    backlogs_count: int = 0
    internships: list[dict[str, Any]] = []
    certifications: list[str] = []
    skills: list[str] = []
    projects_count: int = 0
    github_projects: int = 0
    coding_problem_count: int = 0
    hackathon_wins: int = 0
    leadership_roles_count: int = 0
    paper_publications: int = 0
    extracurriculars_count: int = 0
    languages_known: int = 1
    portal_activity_30d: int = 5
    interview_invites_count: int = 0
    salary_expectation_lpa: float = 6.0
    # Preserve elite-outlier flag so the salary anchor isn't reset on round-trip
    is_elite_outlier: bool = False
    elite_outlier_reasons: list[str] = []


def _to_profile(body: ProfileBody) -> StudentProfile:
    return StudentProfile(
        name=body.name,
        course_type=body.course_type,
        institute_name=body.institute_name,
        institute_tier=body.institute_tier,
        region=body.region,
        graduation_year=body.graduation_year,
        cgpa=body.cgpa,
        backlogs_count=body.backlogs_count,
        internships=body.internships,
        certifications=body.certifications,
        skills=body.skills,
        projects_count=body.projects_count,
        github_projects=body.github_projects,
        coding_problem_count=body.coding_problem_count,
        hackathon_wins=body.hackathon_wins,
        leadership_roles_count=body.leadership_roles_count,
        paper_publications=body.paper_publications,
        extracurriculars_count=body.extracurriculars_count,
        languages_known=body.languages_known,
        portal_activity_30d=body.portal_activity_30d,
        interview_invites_count=body.interview_invites_count,
        salary_expectation_lpa=body.salary_expectation_lpa,
        is_elite_outlier=body.is_elite_outlier,
        elite_outlier_reasons=body.elite_outlier_reasons,
    )


def _probe_openrouter() -> dict:
    """Tiny 4-token probe to verify OpenRouter quota. Updates LLM_STATUS."""
    import os, time as _time
    from openai import OpenAI
    from .resume_parser import LLM_STATUS, DEFAULT_OPENROUTER_MODEL
    api_key = os.environ["OPENROUTER_API_KEY"]
    model_name = os.environ.get("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)
    LLM_STATUS["provider"] = "openrouter"
    LLM_STATUS["model"] = model_name
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "https://github.com/team-aurdinary/careersight",
            "X-Title": "CareerSight",
        },
    )
    t0 = _time.perf_counter()
    completion = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": "Reply with exactly: ok"}],
        max_tokens=4,
        temperature=0.0,
    )
    txt = (completion.choices[0].message.content or "").strip()
    LLM_STATUS["calls_total"] += 1
    LLM_STATUS["calls_ok"] += 1
    LLM_STATUS["last_call_status"] = "ok"
    LLM_STATUS["last_call_at"] = _time.time()
    LLM_STATUS["last_call_latency_ms"] = round((_time.perf_counter() - t0) * 1000, 1)
    LLM_STATUS["last_error"] = None
    return {"reply": txt}


def _probe_gemini() -> dict:
    """Tiny probe for Gemini. Updates LLM_STATUS."""
    import os, time as _time
    import google.generativeai as genai
    from .resume_parser import LLM_STATUS, DEFAULT_GEMINI_MODEL
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"]
    model_name = os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    LLM_STATUS["provider"] = "gemini"
    LLM_STATUS["model"] = model_name
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    t0 = _time.perf_counter()
    r = model.generate_content("Reply with exactly: ok",
                               generation_config={"max_output_tokens": 4})
    txt = (r.text or "").strip()
    LLM_STATUS["calls_total"] += 1
    LLM_STATUS["calls_ok"] += 1
    LLM_STATUS["last_call_status"] = "ok"
    LLM_STATUS["last_call_at"] = _time.time()
    LLM_STATUS["last_call_latency_ms"] = round((_time.perf_counter() - t0) * 1000, 1)
    LLM_STATUS["last_error"] = None
    return {"reply": txt}


@app.get("/api/llm_health")
def llm_health(probe: bool = False) -> dict:
    """Live indicator of the resume-parser LLM cascade.

    Active provider order: OpenRouter -> Gemini -> heuristic.
    Returns the first available provider's status. With `probe=true`,
    fires a tiny 4-token call to verify quota.
    """
    import os
    from .resume_parser import LLM_STATUS, _classify_error

    has_openrouter = bool(os.environ.get("OPENROUTER_API_KEY"))
    has_gemini = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))

    if not (has_openrouter or has_gemini):
        return {
            "mode": "heuristic_only",
            "ok": False,
            "label": "Heuristic only · no API key",
            "provider": None,
            "model": None,
            "key_set": False,
            "calls_total": LLM_STATUS["calls_total"],
            "calls_ok": LLM_STATUS["calls_ok"],
        }

    LLM_STATUS["key_set"] = True

    if probe:
        try:
            if has_openrouter:
                _probe_openrouter()
            else:
                _probe_gemini()
        except Exception as exc:
            LLM_STATUS["last_call_status"] = _classify_error(exc)
            LLM_STATUS["last_error"] = str(exc)[:240]

    # LLM_STATUS["provider"] is "none" until the first real call — fall back to
    # whichever key is configured so the UI shows the right provider label.
    _raw_prov = LLM_STATUS.get("provider", "none")
    provider = _raw_prov if _raw_prov not in ("none", None, "") \
               else ("openrouter" if has_openrouter else "gemini")
    model = LLM_STATUS.get("model") or \
            ("openai/gpt-oss-120b:free" if provider == "openrouter" else "gemini-2.0-flash")
    status = LLM_STATUS["last_call_status"]
    ok = status == "ok"
    pretty_provider = "OpenRouter" if provider == "openrouter" else "Gemini"
    label_map = {
        "ok":           f"{pretty_provider} · live · {model}",
        "rate_limited": f"{pretty_provider} · rate-limited (free-tier quota)",
        "auth_failed":  f"{pretty_provider} · auth failed",
        "parse_failed": f"{pretty_provider} · parse failed (heuristic fallback)",
        "error":        f"{pretty_provider} · error (heuristic fallback)",
        "no_calls_yet": f"{pretty_provider} · ready · {model}",
        "sdk_missing":  f"{pretty_provider} · SDK missing (heuristic only)",
    }
    return {
        "mode": provider,
        "provider": provider,
        "ok": ok,
        "status": status,
        "label": label_map.get(status, f"{pretty_provider} · {status}"),
        "model": model,
        "key_set": True,
        "openrouter_available": has_openrouter,
        "gemini_available": has_gemini,
        "last_call_at": LLM_STATUS["last_call_at"],
        "last_call_latency_ms": LLM_STATUS["last_call_latency_ms"],
        "last_error": LLM_STATUS["last_error"],
        "calls_total": LLM_STATUS["calls_total"],
        "calls_ok": LLM_STATUS["calls_ok"],
    }


@app.get("/api/health")
def health() -> dict:
    from .ipr import stats as ipr_stats
    from .predict import _bundle
    b = _bundle()
    return {
        "status": "ok",
        "model_aucs": b["aucs"],
        "salary_mae_lpa": b["salary_mae"],
        "n_train": b["n_train"],
        "course_types": COURSE_TYPES,
        "calibration": b.get("calibration_reports"),
        "survival": {
            "method": "deephit" if b.get("deephit") else "interpolated",
            "c_index": b.get("deephit", {}).get("c_index"),
            "cuts": b.get("deephit", {}).get("cuts"),
        },
        "ipr": ipr_stats(),
    }


@app.get("/api/ipr_stats")
def ipr_stats_endpoint() -> dict:
    """Coverage of the Institute Placement Registry — sample for the
    Architecture tab."""
    from .ipr import INSTITUTE_PRIORS, stats as ipr_stats
    sample = []
    for slug, inst in list(INSTITUTE_PRIORS.items())[:6]:
        for (degree, branch), d in list(inst["degree_branches"].items())[:2]:
            sample.append({
                "institute": inst["canonical_name"], "tier": inst["tier"],
                "degree": degree, "branch": branch,
                "p50_lpa": d["p50"], "p90_lpa": d["p90"],
                "placement_rate_6m": d["placement_rate_6m"],
                "n": d["sample_size"], "year_bin": d["year_bin"],
            })
    return {"stats": ipr_stats(), "sample": sample}


@app.get("/api/demo_profile/{key}")
def demo_profile(key: str) -> dict:
    if key not in DEMO_PROFILES:
        raise HTTPException(404, f"Unknown demo profile '{key}'. Try one of: {list(DEMO_PROFILES)}")
    started = time.perf_counter()
    profile = DEMO_PROFILES[key]
    result = predict_profile(profile)
    result["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
    result["parser_used"] = "demo_profile"
    return result


@app.post("/api/predict")
def predict(body: ProfileBody) -> dict:
    started = time.perf_counter()
    result = predict_profile(_to_profile(body))
    result["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
    return result


@app.post("/api/score_from_resume")
async def score_from_resume(file: UploadFile = File(...)) -> dict:
    """One-shot scoring: validity gate -> parse -> IPR -> score, returns full
    payload as a single JSON. Use /api/score_from_resume_stream for the SSE
    forensic-discovery flow."""
    from .validity_gate import assess_pdf, rejection_message

    if file.filename and not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Please upload a PDF resume.")

    started = time.perf_counter()
    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(400, "Empty file.")
    if len(pdf_bytes) > 10 * 1024 * 1024:
        raise HTTPException(413, "File too large (>10MB).")

    # Validity gate first — reject non-resumes before any parsing/scoring
    try:
        validity = assess_pdf(pdf_bytes)
    except Exception as exc:
        raise HTTPException(400, f"Could not read PDF: {type(exc).__name__}")
    if not validity.proceed:
        return {
            "rejected": True,
            "reason": rejection_message(validity),
            "validity": validity.to_dict(),
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "filename": file.filename,
        }

    profile, parser_used = parse_resume_pdf(pdf_bytes)
    result = predict_profile(profile)
    result["validity"] = validity.to_dict()
    result["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
    result["parser_used"] = parser_used
    result["filename"] = file.filename
    return result


@app.post("/api/score_from_resume_stream")
async def score_from_resume_stream(file: UploadFile = File(...)) -> StreamingResponse:
    """SSE streaming endpoint — the judge-facing forensic flow.

    Emits stage-by-stage events as the PDF moves through validity / parse /
    IPR / scoring. Frontend consumes via fetch + ReadableStream (since
    EventSource cannot POST).
    """
    if file.filename and not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Please upload a PDF resume.")
    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(400, "Empty file.")
    if len(pdf_bytes) > 10 * 1024 * 1024:
        raise HTTPException(413, "File too large (>10MB).")

    from .stream import stream_pdf
    return StreamingResponse(
        stream_pdf(pdf_bytes, filename=file.filename),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


class TextBody(BaseModel):
    text: str


@app.post("/api/score_from_text")
def score_from_text(body: TextBody) -> dict:
    """Useful for testing without a PDF — paste resume text directly.
    Tries the same cascade as /api/score_from_resume, including the
    validity gate."""
    from .validity_gate import assess_text, rejection_message

    started = time.perf_counter()
    validity = assess_text(body.text)
    if not validity.proceed:
        return {
            "rejected": True,
            "reason": rejection_message(validity),
            "validity": validity.to_dict(),
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        }
    profile, parser_used = parse_resume_text(body.text)
    result = predict_profile(profile)
    result["validity"] = validity.to_dict()
    result["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
    result["parser_used"] = parser_used
    return result


@app.post("/api/score_from_text_stream")
def score_from_text_stream(body: TextBody) -> StreamingResponse:
    """SSE streaming over pasted text — same forensic flow without PDF parsing."""
    from .stream import stream_text
    return StreamingResponse(
        stream_text(body.text),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/api/portfolio")
def portfolio(limit: int = 50, risk_tier: str | None = None) -> dict:
    pf = get_portfolio()
    students = pf["students"]
    if risk_tier:
        students = [s for s in students if s["risk_tier"].lower() == risk_tier.lower()]
    return {
        "summary": pf["summary"],
        "pri_by_sector": pf["pri_by_sector"],
        "students": students[:limit],
        "total": len(students),
    }


@app.get("/api/pri")
def pri() -> dict:
    pf = get_portfolio()
    return {
        "overall": pf["summary"]["pri_overall"],
        "by_sector": pf["pri_by_sector"],
        "history": get_pri_history(),
    }


_DRIFT_CACHE: dict | None = None


@app.get("/api/drift")
def drift() -> dict:
    """Per-feature PSI + KS drift between training distribution and the
    current production cohort. Cached at first call to keep latency low."""
    global _DRIFT_CACHE
    if _DRIFT_CACHE is None:
        from pathlib import Path
        import pandas as pd
        from .drift import compute_drift, synthetic_current_cohort

        ref_path = Path(__file__).resolve().parents[2] / "data" / "synthetic_students.csv"
        ref_df = pd.read_csv(ref_path)
        cur_df = synthetic_current_cohort(ref_df)
        results = compute_drift(ref_df, cur_df)
        _DRIFT_CACHE = {
            "n_reference": int(len(ref_df)),
            "n_current": int(len(cur_df)),
            "thresholds": {"stable": 0.10, "moderate": 0.25},
            "summary": {
                "stable":   sum(1 for r in results if r.band == "stable"),
                "moderate": sum(1 for r in results if r.band == "moderate"),
                "drift":    sum(1 for r in results if r.band == "drift"),
            },
            "features": [r.to_dict() for r in results[:25]],
        }
    return _DRIFT_CACHE


class OutcomeIn(BaseModel):
    """Outcome payload for /api/record_outcome.

    The Co-Pilot recommends an action; later, after the student has acted,
    the lender posts back whether the recommended action led to a placement
    in the next 90 days. The bandit posterior updates immediately."""
    segment_id: str
    action_id: str
    success: int  # 0 or 1


@app.post("/api/record_outcome")
def record_outcome(payload: OutcomeIn) -> dict:
    from .bandit import record
    return record(payload.segment_id, payload.action_id, payload.success)


@app.get("/api/bandit_state")
def bandit_state() -> dict:
    """Beta posteriors per (segment × action) — used by the Architecture tab."""
    from .bandit import state_summary
    return state_summary()


@app.get("/api/federated_shap")
def federated_shap_endpoint() -> dict:
    """Federated SHAP across 2 lender shards with calibrated Gaussian DP."""
    from .predict import _bundle
    b = _bundle()
    return b.get("federated_shap", {"note": "no federated artefact"})


@app.get("/api/counterfactual")
def counterfactual() -> dict:
    """Doubly-robust ATE per Co-Pilot action (EconML LinearDRLearner) +
    placebo refuter. Replaces the hand-picked uplift numbers in the bandit."""
    from .predict import _bundle
    b = _bundle()
    return {"actions": b.get("action_ates", []),
            "note": "DR estimator with placebo refuter (Coston et al., FAccT 2020)."}


@app.get("/api/edge_model")
def edge_model() -> dict:
    """Distilled Born-Again single tree, suitable for offline/edge deployment.
    JSON shape is consumed by the 30-line evaluator in `edge_model.EVALUATOR_PY`."""
    from .predict import _bundle
    b = _bundle()
    return b.get("edge_model", {"note": "no edge artefact"})


@app.get("/api/causal")
def causal() -> dict:
    """Discovered CPDAG over a feature subset + the Markov blanket of the
    placement-6m target. Powers the force-directed graph in Architecture."""
    from .predict import _bundle
    b = _bundle()
    return b.get("causal", {"note": "no causal artefact"})


@app.get("/api/fairness")
def fairness() -> dict:
    """Fairlearn audit grounded in DPDP Act 2023 §10 + EEOC four-fifths rule."""
    from .predict import _bundle
    from .fairness import THRESHOLDS
    b = _bundle()
    reports = b.get("fairness_reports", [])
    pass_count = sum(1 for r in reports if not r["breaches"])
    return {
        "n_audits": len(reports),
        "pass_count": pass_count,
        "thresholds": THRESHOLDS,
        "audits": reports,
    }


class WhatIfBody(BaseModel):
    """Body for /api/whatif. Either supply a `profile` (full StudentProfile
    fields) OR a `demo_key` (one of priya/arjun/meera/rahul) to use a
    pre-built persona as the baseline."""
    interventions: list[str] = Field(default_factory=list)
    profile: ProfileBody | None = None
    demo_key: str | None = None


@app.post("/api/whatif")
def whatif(body: WhatIfBody) -> dict:
    """Counterfactual exploration: apply UI-friendly interventions to a
    student profile, re-run the full prediction, and return before/after
    deltas + a one-line narrative.

    Powers the What-If Simulator card on the Live Demo tab.
    """
    from .whatif import simulate, catalog
    if body.profile is not None:
        profile = _to_profile(body.profile)
    elif body.demo_key:
        if body.demo_key not in DEMO_PROFILES:
            raise HTTPException(
                404,
                f"Unknown demo profile '{body.demo_key}'. "
                f"Try one of: {list(DEMO_PROFILES)}",
            )
        profile = DEMO_PROFILES[body.demo_key]
    else:
        raise HTTPException(400, "Provide either `profile` or `demo_key`.")

    return {
        "result": simulate(profile, body.interventions or []),
        "catalog": catalog(),
    }


@app.get("/api/whatif_catalog")
def whatif_catalog() -> dict:
    """List of UI-exposable interventions for the What-If Simulator card."""
    from .whatif import catalog
    return {"interventions": catalog()}


@app.get("/api/ood")
def ood() -> dict:
    """Out-of-distribution validation against Kaggle Indian-student data
    (or a covariate-shifted internal fold as fallback)."""
    from .predict import _bundle
    b = _bundle()
    reports = b.get("ood_reports", [])
    return {
        "n_folds": len(reports),
        "reports": reports,
        "kaggle_present": any(r["source"].startswith("kaggle_") for r in reports),
    }


# -- Static frontend ---------------------------------------------------------

_FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
if _FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=_FRONTEND_DIR), name="static")

    @app.get("/")
    def root() -> FileResponse:
        return FileResponse(_FRONTEND_DIR / "index.html")
