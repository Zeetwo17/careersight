"""Placement Readiness Score (0-100) — single composite metric.

Rolls the model's already-rich output (placement probability, IPR anchor,
salary band, SHAP drivers, profile signals) into one headline number with
five interpretable sub-scores.

The score reuses inputs the prediction pipeline has already computed —
this module never re-runs the LightGBM heads. It is a thin presentation
layer designed for the hackathon "summary" view.

Sub-scores (each in [0, 1]):
  - academic   : CGPA percentile vs cohort + backlog penalty
  - exposure   : internship count + quality + project depth
  - skill      : breadth of skills, certs, coding output
  - activity   : portal activity, interviews, mock interviews
  - institute  : IPR placement_rate_6m + tier prior

Weighted aggregation (sums to 1.0):
    readiness = 100 * (
        0.25 * academic
      + 0.30 * exposure
      + 0.25 * skill
      + 0.10 * activity
      + 0.10 * institute
    )

Risk mapping mirrors the existing predict.py tier bands so the headline
score and the model risk tier never disagree:
    [70, 100] -> STRONG  (LOW risk)
    [40,  70) -> MEDIUM
    [ 0,  40) -> WEAK    (HIGH risk)
"""
from __future__ import annotations

from typing import Any

from .schema import StudentProfile


_WEIGHTS = {
    "academic":  0.25,
    "exposure":  0.30,
    "skill":     0.25,
    "activity":  0.10,
    "institute": 0.10,
}


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _academic_score(profile: StudentProfile) -> dict[str, Any]:
    """CGPA-driven academic strength, penalised for backlogs.

    The 7.2 anchor is the synthetic-cohort tier-2 mean (matches percentile.py).
    """
    cgpa = float(profile.cgpa or 0)
    raw = (cgpa - 5.0) / 4.0          # CGPA 5 -> 0, CGPA 9 -> 1
    backlog_penalty = 0.10 * float(profile.backlogs_count or 0)
    score = _clip01(raw - backlog_penalty)
    return {
        "score": round(score, 3),
        "label": _band_label(score),
        "detail": f"CGPA {cgpa:.1f}" + (
            f" · {profile.backlogs_count} backlog{'s' if profile.backlogs_count != 1 else ''}"
            if profile.backlogs_count else ""
        ),
    }


def _exposure_score(profile: StudentProfile) -> dict[str, Any]:
    """Internship + project depth signal.

    Internships dominate (top-tier brand + relevance amplifies).
    Projects act as a secondary multiplier.
    """
    internships = profile.internships or []
    n_intern = len(internships)
    top_tier_count = sum(1 for i in internships if i.get("is_top_tier"))
    avg_relevance = (
        sum(float(i.get("relevance", 0.6)) for i in internships) / max(1, n_intern)
        if internships else 0.0
    )

    intern_component = min(1.0, n_intern * 0.30)            # 0,1,2,3+ -> 0,.30,.60,.90+
    if top_tier_count:
        intern_component = min(1.0, intern_component + 0.20)
    intern_component = min(1.0, intern_component * (0.7 + 0.5 * avg_relevance))

    project_component = min(1.0,
        (profile.github_projects or 0) * 0.05 +
        (profile.projects_count or 0) * 0.04
    )
    score = _clip01(0.70 * intern_component + 0.30 * project_component)

    detail_parts: list[str] = []
    if n_intern:
        detail_parts.append(f"{n_intern} internship{'s' if n_intern != 1 else ''}")
    if top_tier_count:
        detail_parts.append(f"{top_tier_count} top-tier")
    if profile.github_projects:
        detail_parts.append(f"{profile.github_projects} GitHub")
    return {
        "score": round(score, 3),
        "label": _band_label(score),
        "detail": " · ".join(detail_parts) or "no internships yet",
    }


def _skill_score(profile: StudentProfile) -> dict[str, Any]:
    """Breadth + technical depth signal."""
    skills = profile.skills or []
    certs = profile.certifications or []
    coding = int(profile.coding_problem_count or 0)
    hackathons = int(profile.hackathon_wins or 0)

    skill_breadth = min(1.0, len(skills) / 8.0)         # 8 skills -> full
    cert_depth = min(1.0, len(certs) / 4.0)             # 4 certs -> full
    coding_depth = min(1.0, coding / 500.0)             # 500 problems -> full
    hack_bonus = min(0.20, hackathons * 0.10)

    score = _clip01(
        0.40 * skill_breadth + 0.25 * cert_depth + 0.30 * coding_depth + hack_bonus
    )

    parts: list[str] = []
    if skills:
        parts.append(f"{len(skills)} skill{'s' if len(skills) != 1 else ''}")
    if certs:
        parts.append(f"{len(certs)} cert{'s' if len(certs) != 1 else ''}")
    if coding:
        parts.append(f"{coding} LC")
    if hackathons:
        parts.append(f"{hackathons} hackathon win{'s' if hackathons != 1 else ''}")
    return {
        "score": round(score, 3),
        "label": _band_label(score),
        "detail": " · ".join(parts) or "limited technical signal",
    }


def _activity_score(profile: StudentProfile) -> dict[str, Any]:
    """Portal + interview engagement signal."""
    portal = int(profile.portal_activity_30d or 0)
    interviews = int(profile.interview_invites_count or 0)

    portal_component = min(1.0, portal / 30.0)          # 30 actions/30d -> full
    interview_component = min(1.0, interviews / 5.0)     # 5 invites -> full
    score = _clip01(0.55 * portal_component + 0.45 * interview_component)

    parts: list[str] = []
    if portal:
        parts.append(f"{portal} portal actions/30d")
    if interviews:
        parts.append(f"{interviews} interview invite{'s' if interviews != 1 else ''}")
    return {
        "score": round(score, 3),
        "label": _band_label(score),
        "detail": " · ".join(parts) or "low engagement",
    }


def _institute_score(ipr_dict: dict[str, Any], profile: StudentProfile) -> dict[str, Any]:
    """Institute strength: IPR 6m placement rate + tier prior, blended."""
    placement_rate = ipr_dict.get("placement_rate", {})
    p6 = float(placement_rate.get("month_6", 0.6))
    tier_prior = {1: 0.95, 2: 0.75, 3: 0.55}.get(int(profile.institute_tier or 2), 0.65)
    # Trust IPR more when data quality is high
    quality = (ipr_dict.get("data_quality") or "baseline").lower()
    quality_w = {"high": 0.85, "medium": 0.70, "low": 0.55, "baseline": 0.30}.get(quality, 0.50)
    score = _clip01(quality_w * p6 + (1 - quality_w) * tier_prior)
    inst_name = ipr_dict.get("canonical_name") or profile.institute_name
    return {
        "score": round(score, 3),
        "label": _band_label(score),
        "detail": f"{inst_name} · Tier-{profile.institute_tier} · IPR {quality}",
    }


def _band_label(score: float) -> str:
    if score >= 0.70:
        return "Strong"
    if score >= 0.40:
        return "Medium"
    return "Weak"


def _overall_band(score_100: float) -> dict[str, str]:
    """Map readiness 0-100 to a band, matching predict.py's risk tiering."""
    if score_100 >= 70:
        return {"band": "STRONG", "tier": "LOW",    "color": "#22c55e", "emoji": "🟢"}
    if score_100 >= 40:
        return {"band": "MEDIUM", "tier": "MEDIUM", "color": "#f59e0b", "emoji": "🟡"}
    return     {"band": "WEAK",   "tier": "HIGH",   "color": "#ef4444", "emoji": "🔴"}


def compute_readiness(profile: StudentProfile, predict_result: dict[str, Any]) -> dict[str, Any]:
    """Compute readiness from a profile + already-computed predict_profile result.

    Reuses the predict_profile dict so we don't pay double for IPR lookup, SHAP,
    or LightGBM inference.
    """
    ipr = predict_result.get("ipr") or {}

    components = {
        "academic":  _academic_score(profile),
        "exposure":  _exposure_score(profile),
        "skill":     _skill_score(profile),
        "activity":  _activity_score(profile),
        "institute": _institute_score(ipr, profile),
    }

    readiness_raw = sum(_WEIGHTS[k] * components[k]["score"] for k in _WEIGHTS)
    readiness_100 = int(round(_clip01(readiness_raw) * 100))
    band = _overall_band(readiness_100)

    # Pull a soft-anchor from the model's calibrated 6m probability so the
    # composite doesn't drift far from the LightGBM signal. We weight 70/30
    # toward our explicit composite (interpretability) but pull toward the
    # model when they disagree heavily.
    p_6m = float(predict_result.get("placement_probabilities", {}).get("p_6m", 0.5))
    model_anchor = int(round(p_6m * 100))
    blended = int(round(0.70 * readiness_100 + 0.30 * model_anchor))

    # Identify the weakest sub-score for an actionable hint
    weakest = min(components.items(), key=lambda kv: kv[1]["score"])

    return {
        "score":            blended,
        "score_components": readiness_100,
        "model_anchor":     model_anchor,
        "band":             band["band"],
        "tier":             band["tier"],
        "color":            band["color"],
        "emoji":            band["emoji"],
        "components":       components,
        "weights":          _WEIGHTS,
        "weakest_area":     weakest[0],
        "weakest_label":    weakest[1]["label"],
        "narrative":        _narrative(blended, band, weakest[0], weakest[1]),
    }


def _narrative(score: int, band: dict, weakest_key: str, weakest: dict) -> str:
    """One-sentence judge-friendly summary."""
    if band["band"] == "STRONG":
        return (f"Placement Readiness {score}/100 — strong overall. "
                f"Lift the {weakest_key} signal ({weakest['label']}) to push past 80.")
    if band["band"] == "MEDIUM":
        return (f"Placement Readiness {score}/100 — closing on placement-ready. "
                f"Biggest opportunity: {weakest_key} ({weakest['label']}).")
    return (f"Placement Readiness {score}/100 — early-warning territory. "
            f"Top fix: {weakest_key} ({weakest['label']}).")
