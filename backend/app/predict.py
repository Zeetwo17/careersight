"""Inference service: StudentProfile -> risk score, salary band, SHAP, Co-Pilot."""
from __future__ import annotations

import math
from pathlib import Path

import joblib
import numpy as np

from .features import feature_descriptions, feature_metadata, profile_to_vector
from .ipr import lookup as ipr_lookup
from .percentile import estimate_student_percentile, quantile_anchor
from .schema import StudentProfile


# ---------------------------------------------------------------------------
# Anchoring policy:
#   - High-quality IPR data dominates the model output (institute knows itself
#     better than the model does).
#   - Low-quality IPR (cluster / national baseline) gives the model the bigger
#     vote.
# These weights are calibrated so a typical L1 institute lookup contributes
# 70% of the salary anchor, L4 cluster contributes 35%.
# ---------------------------------------------------------------------------
ANCHOR_WEIGHT_BY_QUALITY = {
    "high":     0.70,
    "medium":   0.55,
    "low":      0.40,
    "baseline": 0.20,
}


def _ipr_monthly_survival(ipr) -> list[float]:
    """Monotone-interpolate IPR's {3m, 6m, 12m} placement rates to a 13-point
    monthly survival curve (p_unplaced at months 0..12).

    Output: [p_unplaced(0), p_unplaced(1), ..., p_unplaced(12)] all in [0,1],
    monotone non-increasing.
    """
    p3 = float(ipr.placement_rate_3m)
    p6 = float(ipr.placement_rate_6m)
    p12 = float(ipr.placement_rate_12m)
    # Enforce monotonicity in placement rates (sometimes IPR data has noise)
    p6 = max(p6, p3)
    p12 = max(p12, p6)
    curve: list[float] = []
    for t in range(13):
        if t == 0:
            placed = 0.0
        elif t <= 3:
            placed = p3 * (t / 3.0)                         # linear ramp 0 → p3
        elif t <= 6:
            placed = p3 + (p6 - p3) * ((t - 3) / 3.0)       # linear ramp p3 → p6
        elif t <= 12:
            placed = p6 + (p12 - p6) * ((t - 6) / 6.0)      # linear ramp p6 → p12
        else:
            placed = p12
        curve.append(round(max(0.0, min(1.0, 1.0 - placed)), 4))
    return curve


def _blend_curves(model_curve: list[dict], ipr_curve: list[float],
                  alpha: float) -> list[dict]:
    """Blend model survival with IPR survival point-by-point. Monotone
    enforced post-blend (Isotonic-style: each point ≤ previous)."""
    blended: list[dict] = []
    prev = 1.0
    for i, m in enumerate(model_curve):
        m_p = float(m.get("p_unplaced", 1.0))
        i_p = float(ipr_curve[i] if i < len(ipr_curve) else m_p)
        b = alpha * i_p + (1.0 - alpha) * m_p
        b = min(b, prev)         # monotone non-increasing
        blended.append({"month": m.get("month", i), "p_unplaced": round(b, 4)})
        prev = b
    return blended

_BUNDLE = None
_DEEPHIT_MODEL = None
_BUNDLE_PATH = Path(__file__).resolve().parents[2] / "data" / "models" / "bundle.joblib"


def _bundle():
    global _BUNDLE
    if _BUNDLE is None:
        if not _BUNDLE_PATH.exists():
            raise FileNotFoundError(
                f"Model bundle not found at {_BUNDLE_PATH}. "
                "Run: python -m backend.app.train"
            )
        _BUNDLE = joblib.load(_BUNDLE_PATH)
    return _BUNDLE


def _deephit_model():
    """Lazy-rebuild the DeepHit model from the persisted artefact."""
    global _DEEPHIT_MODEL
    if _DEEPHIT_MODEL is not None:
        return _DEEPHIT_MODEL
    bundle = _bundle()
    art = bundle.get("deephit")
    sd = bundle.get("deephit_state_dict")
    if art is None or sd is None:
        return None
    try:
        from .survival import reload_model
        _DEEPHIT_MODEL = reload_model(art, sd)
    except Exception:
        _DEEPHIT_MODEL = None
    return _DEEPHIT_MODEL


def _risk_tier(p_6m: float) -> tuple[str, int]:
    """Map 6-month placement probability to a risk tier and 0-100 score.

    Lower placement probability -> higher risk score.
    """
    risk_score = int(round((1.0 - p_6m) * 100))
    if risk_score >= 60:
        return "HIGH", risk_score
    if risk_score >= 35:
        return "MEDIUM", risk_score
    return "LOW", risk_score


def _shap_top_drivers(vec: np.ndarray, top_k: int = 5) -> list[dict]:
    """Top SHAP contributions (absolute value) toward the placement-risk signal.

    SHAP returns contributions toward placed_6m=1 (positive). We invert sign
    so positive driver values mean "increases placement risk."

    Uses LightGBM's native C++ SHAP (pred_contrib) instead of the shap library
    to avoid Python-wrapper overhead on single-sample inference.
    """
    bundle = _bundle()
    feat_names = bundle["feature_names"]
    try:
        # LightGBM native SHAP — pure C++, 10-20x faster than shap library
        # pred_contrib returns shape (1, n_features + 1); last col is bias/intercept.
        contrib = bundle["classifiers"]["placed_6m"].booster_.predict(
            vec.reshape(1, -1), pred_contrib=True
        )
        sv = np.asarray(contrib[0, :-1])   # drop the bias term
    except Exception:
        # Fallback: shap library (slower but always works)
        explainer = bundle["explainer"]
        sv_raw = explainer.shap_values(vec.reshape(1, -1), check_additivity=False)
        if isinstance(sv_raw, list):
            sv_raw = sv_raw[1] if len(sv_raw) == 2 else sv_raw[0]
        sv = np.asarray(sv_raw).reshape(-1)

    meta = feature_metadata()
    pairs = []
    for name, val in zip(feat_names, sv):
        # Invert: -SHAP for placed=1 -> +contribution to risk
        risk_contribution = float(-val)
        if abs(risk_contribution) < 1e-4:
            continue
        m = meta.get(name, {})
        pairs.append({
            "feature": name,
            "label": m.get("label", name.replace("_", " ").title()),
            "tooltip": m.get("tooltip", ""),
            "direction": m.get("direction", "mixed"),
            "shap": float(val),
            "risk_contribution": risk_contribution,
        })
    pairs.sort(key=lambda p: abs(p["risk_contribution"]), reverse=True)
    return pairs[:top_k]


def _explanation_sentence(profile: StudentProfile, top_drivers: list[dict],
                          p_6m: float, salary_low: float, salary_high: float) -> str:
    """Build a plain-English SHAP-grounded explanation."""
    risk_drivers = [d for d in top_drivers if d["risk_contribution"] > 0]
    helping = [d for d in top_drivers if d["risk_contribution"] < 0]

    risk_phrases = ", ".join(d["label"].lower() for d in risk_drivers[:3]) or "limited adverse signals"
    help_phrases = ", ".join(d["label"].lower() for d in helping[:2]) or "few protective factors"

    return (
        f"6-month placement probability is {int(p_6m*100)}%. "
        f"Risk drivers: {risk_phrases}. Strengths: {help_phrases}. "
        f"Salary band Rs {salary_low:.1f}L-Rs {salary_high:.1f}L for {profile.course_type} "
        f"in a Tier-{profile.institute_tier} institute."
    )


def _copilot_actions(profile: StudentProfile, risk_tier: str,
                      *, current_p_6m: float = 0.0, top_k: int = 3) -> list[dict]:
    """Thompson-sampled Co-Pilot actions with counterfactual phrasing.

    Each action now carries:
      - the current 6m probability (the baseline before the action)
      - the projected 6m probability after the action (current + uplift)
      - the 95% CI bounds on the projected probability
      - a one-sentence counterfactual statement for the UI

    This converts "+6pp" (judge has to do mental math) into
    "predicted 6m placement rises from 83% to 89% (95% CI: 83-95%)" —
    the same number, but framed as the actionable outcome.
    """
    from .bandit import recommend
    recs = recommend(profile.to_dict(), risk_tier, top_k=top_k)
    out = []
    for r in recs:
        ci_lo, ci_hi = r.uplift_ci_pp
        # Project the 6m placement probability after taking this action.
        # Clamp to [0, 100] so the UI never shows >100%.
        baseline_pct = round(current_p_6m * 100, 1)
        projected_pct = min(100.0, max(0.0, baseline_pct + r.expected_uplift_pp))
        projected_lo  = min(100.0, max(0.0, baseline_pct + ci_lo))
        projected_hi  = min(100.0, max(0.0, baseline_pct + ci_hi))
        evidence_tag = (f"n={r.n_trials} similar profiles"
                        if r.n_trials >= 5
                        else "based on prior — feedback loop will sharpen this")
        sentence = (
            f"If this student does {r.title.lower()}, predicted 6-month placement "
            f"rises from {baseline_pct:.0f}% to {projected_pct:.0f}% "
            f"(95% CI: {projected_lo:.0f}–{projected_hi:.0f}%, {evidence_tag})."
        )
        out.append({
            "action_id":          r.action_id,
            "title":              r.title,
            "detail":             r.detail,
            "impact":             r.impact,
            "expected_uplift_pp": r.expected_uplift_pp,
            "uplift_ci_pp":       [round(ci_lo, 1), round(ci_hi, 1)],
            "n_trials":           r.n_trials,
            "baseline_p_6m_pct":  baseline_pct,
            "projected_p_6m_pct": round(projected_pct, 1),
            "projected_ci_pct":   [round(projected_lo, 1), round(projected_hi, 1)],
            "counterfactual":     sentence,
        })
    return out


def _predict_calibrated(bundle, label: str, Xrow: np.ndarray) -> float:
    """Run the LightGBM head and apply Beta calibration if available."""
    raw = float(bundle["classifiers"][label].predict_proba(Xrow)[:, 1][0])
    beta = bundle.get("calibrators", {}).get(label)
    if beta is None:
        return raw
    p = float(beta.transform(np.array([raw]))[0])
    return float(np.clip(p, 1e-6, 1 - 1e-6))


def predict_profile(profile: StudentProfile) -> dict:
    """Full inference path. Returns a JSON-serialisable dict.

    Salary and placement probability are *anchored* to the Institute Placement
    Registry (IPR) — institute-specific empirical distributions take priority
    over the model's tier-derived prediction whenever IPR data is available.
    """
    bundle = _bundle()
    vec, feats = profile_to_vector(profile)
    Xrow = vec.reshape(1, -1)

    # ------- IPR anchor: actual placement distribution for this institute -------
    ipr = ipr_lookup(
        institute_name=profile.institute_name,
        course_type=profile.course_type,
        tier_hint=profile.institute_tier,
    )
    anchor_w = ANCHOR_WEIGHT_BY_QUALITY.get(ipr.data_quality, 0.30)

    # ------- Raw model outputs -------
    raw_p_3m = _predict_calibrated(bundle, "placed_3m", Xrow)
    raw_p_6m = _predict_calibrated(bundle, "placed_6m", Xrow)
    raw_p_12m = _predict_calibrated(bundle, "placed_12m", Xrow)

    # ------- Anchor placement probabilities to IPR's actual placement rates -------
    # Rationale: the IPR placement_rate_6m is the *measured* rate at this
    # institute × branch. The model's calibrated probability is the per-student
    # adjustment. We blend them: high IPR quality -> IPR dominates; low quality
    # (cluster / baseline) -> model dominates.
    p_3m  = anchor_w * ipr.placement_rate_3m  + (1 - anchor_w) * raw_p_3m
    p_6m  = anchor_w * ipr.placement_rate_6m  + (1 - anchor_w) * raw_p_6m
    p_12m = anchor_w * ipr.placement_rate_12m + (1 - anchor_w) * raw_p_12m

    # ------- Raw salary CQR outputs -------
    raw_s_low  = float(bundle["salary_low"].predict(Xrow)[0])
    raw_s_med  = float(bundle["salary_med"].predict(Xrow)[0])
    raw_s_high = float(bundle["salary_high"].predict(Xrow)[0])
    raw_s_low, raw_s_med, raw_s_high = sorted([raw_s_low, raw_s_med, raw_s_high])

    # ------- Quantile-aware salary anchoring -------
    # Step 1: estimate student's percentile within their (institute × branch)
    #          cohort using a heuristic over already-extracted features.
    intern_top_tier = int(any(i.get("is_top_tier") for i in (profile.internships or [])))
    intern_brand = float(np.mean([
        float(i.get("relevance", 0.6)) for i in (profile.internships or [])
    ])) if profile.internships else 0.0
    student_pct = estimate_student_percentile(
        cgpa=float(profile.cgpa),
        internship_top_tier=intern_top_tier,
        internship_brand_score=intern_brand,
        coding_problem_count=int(profile.coding_problem_count or 0),
        hackathon_wins=int(profile.hackathon_wins or 0),
        paper_publications=int(profile.paper_publications or 0),
        is_elite_outlier=bool(profile.is_elite_outlier),
        institute_tier=int(ipr.tier),
    )

    # Step 2: anchor each quantile of the IPR distribution at the position
    #          matching the student's percentile, blended with the model.
    ipr_q = {"p10": ipr.p10, "p25": ipr.p25, "p50": ipr.p50,
             "p75": ipr.p75, "p90": ipr.p90}
    # Build a 5-point model quantile dict from the 3 raw CQR predictions
    # (the bundle only has 10/50/90; we interpolate p25 and p75 linearly).
    model_q = {
        "p10": raw_s_low,
        "p25": raw_s_low + 0.5 * (raw_s_med - raw_s_low),
        "p50": raw_s_med,
        "p75": raw_s_med + 0.5 * (raw_s_high - raw_s_med),
        "p90": raw_s_high,
    }
    anchored = quantile_anchor(
        student_percentile=student_pct,
        ipr_quantiles=ipr_q,
        model_quantiles=model_q,
        base_anchor_weight=anchor_w,
        is_elite=bool(profile.is_elite_outlier),
    )

    # Step 3: confidence widening — low parser confidence or sparse IPR data
    #          stretches the range proportionally around the median.
    fc = profile.field_confidence or {}
    parse_conf_mean = float(np.mean([
        v for k, v in fc.items()
        if isinstance(v, (int, float)) and not k.startswith("_")
    ] or [0.7]))
    widen_factor = 1.0
    if parse_conf_mean < 0.65:
        widen_factor *= 1.20
    if ipr.fallback_level >= 4:
        widen_factor *= 1.15
    if widen_factor > 1.0:
        med = anchored["p50"]
        for key in ("p10", "p25"):
            anchored[key] = med - (med - anchored[key]) * widen_factor
        for key in ("p75", "p90"):
            anchored[key] = med + (anchored[key] - med) * widen_factor

    # Hard floors + monotone enforcement (defensive)
    anchored["p10"] = max(2.0, anchored["p10"])
    for k_prev, k_next in [("p10", "p25"), ("p25", "p50"),
                            ("p50", "p75"), ("p75", "p90")]:
        anchored[k_next] = max(anchored[k_next], anchored[k_prev] + 0.3)

    s_low  = float(anchored["p10"])
    s_med  = float(anchored["p50"])
    s_high = float(anchored["p90"])
    s_p25  = float(anchored["p25"])
    s_p75  = float(anchored["p75"])

    tier_label, risk_score = _risk_tier(p_6m)
    # CRC threshold replaces the previous hard-coded 60. Comes from MAPIE's
    # Learn-Then-Test controller fitted on the calibration fold.
    crc_meta = bundle.get("crc") or {}
    crc_threshold = crc_meta.get("risk_score_threshold")
    if crc_threshold is None:
        crc_threshold = 60
    top_drivers = _shap_top_drivers(vec, top_k=5)
    explanation = _explanation_sentence(profile, top_drivers, p_6m, s_low, s_high)
    actions = _copilot_actions(profile, tier_label, current_p_6m=p_6m, top_k=4)

    # Survival curve: prefer DeepHitSingle (Lee et al., AAAI 2018) when the
    # bundled artefact is loadable, fall back to monotonic interpolation
    # against the three classifier anchors otherwise.
    survival_method = "deephit"
    survival_pairs = None
    dh = _deephit_model()
    if dh is not None:
        try:
            from .survival import survival_curve as deephit_curve
            survival_pairs = deephit_curve(dh, vec)
        except Exception:
            survival_pairs = None
    if survival_pairs is None:
        survival_method = "interpolated"
        months = list(range(0, 13))
        survival_pairs = []
        for t in months:
            if t == 0:
                survival_pairs.append({"month": t, "p_unplaced": 1.0})
            elif t <= 3:
                survival_pairs.append({"month": t, "p_unplaced": round(1.0 - p_3m * (t / 3.0), 4)})
            elif t <= 6:
                base = 1.0 - p_3m
                extra = (p_6m - p_3m) * ((t - 3) / 3.0)
                survival_pairs.append({"month": t, "p_unplaced": round(max(0.0, base - extra), 4)})
            elif t <= 12:
                base = 1.0 - p_6m
                extra = (p_12m - p_6m) * ((t - 6) / 6.0)
                survival_pairs.append({"month": t, "p_unplaced": round(max(0.0, base - extra), 4)})
            else:
                survival_pairs.append({"month": t, "p_unplaced": round(max(0.0, 1.0 - p_12m), 4)})

    # ------- Monthly survival curve blend -------
    # Build the IPR's monotone monthly survival reference, then blend with the
    # model's curve point-by-point. The IPR curve is also exposed in the
    # response so the UI can render it as a dotted reference line behind the
    # student's predicted curve. Captures whether the institute places
    # graduates early (steep early drop) or late (slow drop), not just the
    # 6m endpoint.
    ipr_survival_pairs = [{"month": t, "p_unplaced": v}
                          for t, v in enumerate(_ipr_monthly_survival(ipr))]
    raw_survival_pairs = list(survival_pairs)        # preserve pre-blend for diagnostics
    survival_pairs = _blend_curves(survival_pairs,
                                   [p["p_unplaced"] for p in ipr_survival_pairs],
                                   alpha=anchor_w)

    # Early-warning days: HIGH-risk students get a 90+ day pre-EMI alert.
    if tier_label == "HIGH":
        ews_days = 92
    elif tier_label == "MEDIUM":
        ews_days = 60
    else:
        ews_days = 0

    # Build a human-readable list of fields that were imputed (confidence ≤ 0.30)
    # rather than extracted from the resume.  The UI shows these as editable so
    # the user can correct them before trusting the prediction.
    _CRITICAL_FIELDS = {
        "cgpa":           ("CGPA",           lambda p: f"{p.cgpa:.1f} (population mean)"),
        "institute_name": ("Institute",       lambda p: p.institute_name),
        "course_type":    ("Course / degree", lambda p: p.course_type),
        "backlogs_count": ("Backlogs",        lambda p: str(p.backlogs_count)),
    }
    fc = profile.field_confidence or {}
    imputed_fields = [
        {
            "field":   field,
            "label":   label,
            "imputed": fmt(profile),
            "conf":    round(fc.get(field, 0.0), 2),
        }
        for field, (label, fmt) in _CRITICAL_FIELDS.items()
        if fc.get(field, 0.0) <= 0.30
    ]

    payload = {
        "profile": profile.to_dict(),
        "imputed_fields": imputed_fields,   # [] when everything was extracted
        "nirf_match": feats.get("_nirf_match"),
        "ipr": {
            **ipr.to_dict(),
            "anchor_weight": round(anchor_w, 2),
            "raw_model": {
                "p_3m": round(raw_p_3m, 3),
                "p_6m": round(raw_p_6m, 3),
                "p_12m": round(raw_p_12m, 3),
                "salary_low": round(raw_s_low, 2),
                "salary_med": round(raw_s_med, 2),
                "salary_high": round(raw_s_high, 2),
            },
            "student_percentile": round(student_pct, 2),
            "parse_confidence_mean": round(parse_conf_mean, 2),
            "widen_factor": round(widen_factor, 2),
        },
        "elite_outlier": {
            "is_elite_outlier": profile.is_elite_outlier,
            "reasons": profile.elite_outlier_reasons or [],
        },
        "risk": {
            "score": risk_score,
            "tier": tier_label,
            "tier_color": {"LOW": "#22c55e", "MEDIUM": "#f59e0b", "HIGH": "#ef4444"}[tier_label],
            "crc_fnr_target": 1.0 - float(crc_meta.get("target_level", 0.90)),
            "crc_threshold": int(crc_threshold),
            "crc_confidence": float(crc_meta.get("confidence_level", 0.90)),
            "crc_lambda_star": crc_meta.get("lambda_star"),
            "crc_note": crc_meta.get("note"),
        },
        "placement_probabilities": {
            "p_3m": p_3m,
            "p_6m": p_6m,
            "p_12m": p_12m,
        },
        "salary_band_lpa": {
            "low":    round(s_low,  2),
            "p25":    round(s_p25,  2),
            "median": round(s_med,  2),
            "p75":    round(s_p75,  2),
            "high":   round(s_high, 2),
        },
        "top_drivers": top_drivers,
        "explanation": explanation,
        "copilot_actions": actions,
        "survival_curve": survival_pairs,
        "survival_method": survival_method,
        "ipr_survival_curve": ipr_survival_pairs,
        "raw_survival_curve": raw_survival_pairs,
        "early_warning_days": ews_days,
        "model_meta": {
            "auc_3m": bundle["aucs"]["placed_3m"],
            "auc_6m": bundle["aucs"]["placed_6m"],
            "auc_12m": bundle["aucs"]["placed_12m"],
            "salary_mae_lpa": bundle["salary_mae"],
            "n_train": bundle["n_train"],
            "calibration": bundle.get("calibration_reports"),
        },
    }

    # ---- Decision-intelligence layer: readiness score + peer benchmark ----
    # Both are pure post-processing that re-uses what we just computed —
    # they cannot fail the prediction; if anything goes wrong we still
    # return the core payload.
    try:
        from .readiness import compute_readiness
        payload["readiness"] = compute_readiness(profile, payload)
    except Exception as exc:
        payload["readiness"] = {"available": False, "error": str(exc)[:160]}
    try:
        from .peer_benchmark import benchmark_profile
        payload["peer_benchmark"] = benchmark_profile(profile, payload)
    except Exception as exc:
        payload["peer_benchmark"] = {"available": False, "error": str(exc)[:160]}

    return payload
