"""SSE streaming pipeline for the live demo.

Generates server-sent-events as the resume moves through the validity gate,
parsing, IPR lookup, and scoring. The judge sees one finding at a time —
forensic discovery rather than a single JSON dump.

Each event is shaped as:
    event: <type>
    data: {...json...}

The frontend's EventSource splits on these and renders a bullet/card per
event.

Latency: each stage emits as soon as it has data; we add small visible
sleeps between sub-bullets to make the reveal humanly readable. Total
wall time ~3.5–5s for a typical resume (generous for the demo).
"""
from __future__ import annotations

import asyncio
import json
import time
from functools import partial
from typing import AsyncIterator

from .ipr import lookup as ipr_lookup
from .predict import predict_profile
from .resume_parser import parse_resume_pdf, parse_resume_text
from .schema import StudentProfile
from .validity_gate import assess_pdf, assess_text, rejection_message


# Pacing — each value is the post-event sleep to give the UI breathing room.
PACE_FAST = 0.10        # between fast sub-bullets
PACE_SLOW = 0.35        # between major stage cards


def _sse(event: str, data: dict | str) -> str:
    """Format one SSE message."""
    payload = data if isinstance(data, str) else json.dumps(data, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


async def stream_pdf(pdf_bytes: bytes, *, filename: str | None = None) -> AsyncIterator[str]:
    """Streams events for a PDF upload."""
    started = time.perf_counter()

    # ---- Stage 0: file received ----
    yield _sse("upload", {
        "filename": filename or "resume.pdf",
        "size_bytes": len(pdf_bytes),
        "size_kb": round(len(pdf_bytes) / 1024, 1),
    })
    await asyncio.sleep(PACE_SLOW)

    # ---- Stage 1: validity gate ----
    yield _sse("stage", {"name": "validity", "label": "Checking if this is a resume..."})
    await asyncio.sleep(PACE_FAST)
    try:
        report = assess_pdf(pdf_bytes)
    except Exception as exc:
        yield _sse("error", {"stage": "validity",
                             "message": f"Could not read this PDF ({type(exc).__name__}). Try a different file."})
        return

    # Emit each FOUND signal as a bullet, paced
    for sig in report.signals:
        if sig.found:
            yield _sse("validity_signal", sig.to_dict())
            await asyncio.sleep(PACE_FAST)

    yield _sse("validity_result", {
        **report.to_dict(),
        "text": None,                 # don't ship the full text; UI doesn't need it
    })
    await asyncio.sleep(PACE_SLOW)

    if not report.proceed:
        yield _sse("rejected", {
            "reason": rejection_message(report),
            "p_resume": round(report.p_resume, 3),
        })
        yield _sse("done", {"latency_ms": round((time.perf_counter() - started) * 1000, 1)})
        return

    # ---- Stage 2: parse the resume ----
    yield _sse("stage", {"name": "parse", "label": "Reading resume content..."})
    await asyncio.sleep(PACE_FAST)
    try:
        loop = asyncio.get_event_loop()
        profile, parser_used = await loop.run_in_executor(
            None, parse_resume_pdf, pdf_bytes
        )
    except Exception as exc:
        yield _sse("error", {"stage": "parse",
                             "message": f"Parser failed ({type(exc).__name__}); falling back to defaults."})
        profile, parser_used = StudentProfile(), "fallback"

    # Stream the parsed fields one by one with their confidence
    fc = profile.field_confidence or {}
    for field, label in [
        ("name", "Name"),
        ("course_type", "Course"),
        ("institute_name", "Institute"),
        ("graduation_year", "Graduation"),
        ("cgpa", "CGPA"),
        ("backlogs_count", "Backlogs"),
    ]:
        value = getattr(profile, field, None)
        conf = float(fc.get(field, 0.0)) if not isinstance(fc.get(field), str) else 0.0
        yield _sse("parsed_field", {
            "field": field, "label": label,
            "value": value, "confidence": round(conf, 2),
            "imputed": conf < 0.4,
        })
        await asyncio.sleep(PACE_FAST)

    if profile.internships:
        yield _sse("parsed_field", {
            "field": "internships", "label": "Internships",
            "value": [i.get("company") for i in profile.internships if i.get("company")],
            "count": len(profile.internships),
            "confidence": float(fc.get("internships", 0.0)) if not isinstance(fc.get("internships"), str) else 0.0,
            "imputed": False,
        })
        await asyncio.sleep(PACE_FAST)
    if profile.skills:
        yield _sse("parsed_field", {
            "field": "skills", "label": "Skills",
            "value": profile.skills[:10],
            "count": len(profile.skills),
            "confidence": float(fc.get("skills", 0.0)) if not isinstance(fc.get("skills"), str) else 0.0,
            "imputed": False,
        })
        await asyncio.sleep(PACE_FAST)
    if profile.certifications:
        yield _sse("parsed_field", {
            "field": "certifications", "label": "Certifications",
            "value": profile.certifications[:6],
            "count": len(profile.certifications),
            "confidence": float(fc.get("certifications", 0.0)) if not isinstance(fc.get("certifications"), str) else 0.0,
            "imputed": False,
        })
        await asyncio.sleep(PACE_FAST)

    yield _sse("parse_result", {
        "parser_used": parser_used,
        "layout": fc.get("_layout", "single"),
        "page_count": fc.get("_page_count", 1),
        "duplicate_pages": fc.get("_duplicate_pages", 0),
        "is_elite_outlier": profile.is_elite_outlier,
        "elite_outlier_reasons": profile.elite_outlier_reasons or [],
    })
    await asyncio.sleep(PACE_SLOW)

    # ---- Stage 3: IPR lookup (real institute data) ----
    yield _sse("stage", {"name": "ipr", "label": "Looking up institute placement data..."})
    await asyncio.sleep(PACE_FAST)
    ipr = ipr_lookup(
        institute_name=profile.institute_name,
        course_type=profile.course_type,
        tier_hint=profile.institute_tier,
    )
    yield _sse("ipr_card", ipr.to_dict())
    await asyncio.sleep(PACE_SLOW)

    # ---- Stage 4: full scoring (CPU-bound — run in thread to keep event loop free) ----
    yield _sse("stage", {"name": "score", "label": "Running placement model + salary anchor..."})
    await asyncio.sleep(PACE_FAST)
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, predict_profile, profile)
    except Exception as exc:
        yield _sse("error", {"stage": "score",
                             "message": f"Scoring failed ({type(exc).__name__})"})
        yield _sse("done", {"latency_ms": round((time.perf_counter() - started) * 1000, 1)})
        return

    # Emit the salient cards in sequence so the UI animates them in
    yield _sse("salary_card", {
        "salary_band_lpa": result["salary_band_lpa"],
        "ipr_p50": result["ipr"]["salary_percentiles_lpa"]["p50"],
        "anchor_weight": result["ipr"]["anchor_weight"],
        "is_elite_outlier": result["elite_outlier"]["is_elite_outlier"],
        "elite_reasons": result["elite_outlier"]["reasons"],
    })
    await asyncio.sleep(PACE_FAST)
    yield _sse("placement_card", {
        "placement_probabilities": result["placement_probabilities"],
        "ipr_placement_rate": result["ipr"]["placement_rate"],
        "risk": result["risk"],
        "early_warning_days": result["early_warning_days"],
        "survival_curve": result["survival_curve"],
        "survival_method": result["survival_method"],
    })
    await asyncio.sleep(PACE_FAST)
    yield _sse("drivers_card", {
        "top_drivers": result["top_drivers"],
        "explanation": result["explanation"],
    })
    await asyncio.sleep(PACE_FAST)
    yield _sse("copilot_card", {
        "copilot_actions": result["copilot_actions"],
    })

    # ---- Done ----
    result["parser_used"] = parser_used   # surface which parser was used in the result
    yield _sse("result", result)   # full payload, in case the UI wants to re-render
    yield _sse("done", {
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        "parser_used": parser_used,
    })


async def stream_text(text: str) -> AsyncIterator[str]:
    """SSE for the /api/score_from_text path. Skips PDF-specific bits."""
    started = time.perf_counter()

    yield _sse("upload", {"filename": "(pasted text)", "size_bytes": len(text)})
    await asyncio.sleep(PACE_SLOW)

    yield _sse("stage", {"name": "validity", "label": "Checking if this is a resume..."})
    await asyncio.sleep(PACE_FAST)
    report = assess_text(text)
    for sig in report.signals:
        if sig.found:
            yield _sse("validity_signal", sig.to_dict())
            await asyncio.sleep(PACE_FAST)
    yield _sse("validity_result", {**report.to_dict(), "text": None})
    await asyncio.sleep(PACE_SLOW)

    if not report.proceed:
        yield _sse("rejected", {"reason": rejection_message(report),
                                "p_resume": round(report.p_resume, 3)})
        yield _sse("done", {"latency_ms": round((time.perf_counter() - started) * 1000, 1)})
        return

    yield _sse("stage", {"name": "parse", "label": "Reading resume content..."})
    await asyncio.sleep(PACE_FAST)
    loop = asyncio.get_event_loop()
    profile, parser_used = await loop.run_in_executor(
        None, parse_resume_text, text
    )
    fc = profile.field_confidence or {}
    for field, label in [("name", "Name"), ("course_type", "Course"),
                         ("institute_name", "Institute"), ("cgpa", "CGPA")]:
        value = getattr(profile, field, None)
        conf = float(fc.get(field, 0.0)) if not isinstance(fc.get(field), str) else 0.0
        yield _sse("parsed_field", {"field": field, "label": label,
                                    "value": value, "confidence": round(conf, 2),
                                    "imputed": conf < 0.4})
        await asyncio.sleep(PACE_FAST)
    yield _sse("parse_result", {"parser_used": parser_used,
                                "is_elite_outlier": profile.is_elite_outlier,
                                "elite_outlier_reasons": profile.elite_outlier_reasons or []})
    await asyncio.sleep(PACE_SLOW)

    yield _sse("stage", {"name": "ipr", "label": "Looking up institute placement data..."})
    await asyncio.sleep(PACE_FAST)
    ipr = ipr_lookup(institute_name=profile.institute_name,
                     course_type=profile.course_type,
                     tier_hint=profile.institute_tier)
    yield _sse("ipr_card", ipr.to_dict())
    await asyncio.sleep(PACE_SLOW)

    yield _sse("stage", {"name": "score", "label": "Running placement model + salary anchor..."})
    await asyncio.sleep(PACE_FAST)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, predict_profile, profile)
    yield _sse("salary_card", {
        "salary_band_lpa": result["salary_band_lpa"],
        "ipr_p50": result["ipr"]["salary_percentiles_lpa"]["p50"],
        "anchor_weight": result["ipr"]["anchor_weight"],
        "is_elite_outlier": result["elite_outlier"]["is_elite_outlier"],
        "elite_reasons": result["elite_outlier"]["reasons"],
    })
    await asyncio.sleep(PACE_FAST)
    yield _sse("placement_card", {
        "placement_probabilities": result["placement_probabilities"],
        "ipr_placement_rate": result["ipr"]["placement_rate"],
        "risk": result["risk"],
        "survival_curve": result["survival_curve"],
        "survival_method": result["survival_method"],
    })
    await asyncio.sleep(PACE_FAST)
    yield _sse("drivers_card", {"top_drivers": result["top_drivers"],
                                 "explanation": result["explanation"]})
    await asyncio.sleep(PACE_FAST)
    yield _sse("copilot_card", {"copilot_actions": result["copilot_actions"]})
    result["parser_used"] = parser_used
    yield _sse("result", result)
    yield _sse("done", {"latency_ms": round((time.perf_counter() - started) * 1000, 1),
                        "parser_used": parser_used})
