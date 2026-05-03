"""What-If Simulator — counterfactual exploration over a StudentProfile.

The user (or the UI) supplies a list of `interventions`, each a small
mutation of the profile. We re-run the full prediction pipeline on the
mutated profile and return the new placement probabilities, salary band,
and risk score, along with the deltas vs. the baseline.

Supported interventions (UI-friendly toggles + sliders):
  - "add_internship"          : appends a relevant non-top-tier internship
  - "add_top_tier_internship" : appends a FAANG-quality internship
  - "boost_cgpa"              : adds +0.5 (default) to CGPA, capped at 9.5
  - "add_skill"               : adds 1 generic skill, capped at 12
  - "add_certification"       : appends a relevant cert (AWS / Data / Cloud)
  - "add_coding_problems"     : adds +200 LeetCode-style problems
  - "add_hackathon_win"       : +1 hackathon win
  - "boost_activity"          : portal_activity +20, interview_invites +1
  - "remove_backlog"          : decrements backlogs (floored at 0)

Composability: multiple interventions can be applied in one call. The
counterfactual engine elsewhere in the bundle (EconML LinearDRLearner)
provides causally adjusted ATEs — we surface that data when present so the
UI can show "ATE-corrected" alongside "naive resimulation."
"""
from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any

from .schema import StudentProfile

# `predict_profile` is imported lazily inside `simulate` so this module can
# be unit-tested (catalog, _apply_one) without pulling LightGBM / SHAP.


INTERVENTION_CATALOG = [
    {
        "id": "add_top_tier_internship",
        "label": "Add a FAANG-tier internship",
        "category": "exposure",
        "icon": "★",
    },
    {
        "id": "add_internship",
        "label": "Add a relevant internship",
        "category": "exposure",
        "icon": "+",
    },
    {
        "id": "boost_cgpa",
        "label": "Improve CGPA by 0.5",
        "category": "academic",
        "icon": "📈",
    },
    {
        "id": "add_skill",
        "label": "Add a high-demand skill",
        "category": "skill",
        "icon": "✦",
    },
    {
        "id": "add_certification",
        "label": "Earn a cloud / data certification",
        "category": "skill",
        "icon": "✓",
    },
    {
        "id": "add_coding_problems",
        "label": "Solve +200 coding problems",
        "category": "skill",
        "icon": "{ }",
    },
    {
        "id": "add_hackathon_win",
        "label": "Win a hackathon",
        "category": "skill",
        "icon": "⚡",
    },
    {
        "id": "boost_activity",
        "label": "Engage with placement portal",
        "category": "activity",
        "icon": "↗",
    },
    {
        "id": "remove_backlog",
        "label": "Clear one backlog",
        "category": "academic",
        "icon": "✕",
    },
]


_HIGH_DEMAND_SKILLS = ["python", "aws", "docker", "react", "kubernetes",
                       "tensorflow", "go", "typescript", "sql", "spark",
                       "pytorch", "kafka"]
_RELEVANT_CERTS = ["AWS Cloud Practitioner", "Google Data Analytics",
                   "Azure Fundamentals", "Kubernetes CKAD", "TensorFlow Developer"]


def _apply_one(profile: StudentProfile, intervention: str) -> StudentProfile:
    """Return a NEW StudentProfile with the intervention applied."""
    p = copy.deepcopy(profile)
    p.internships = list(p.internships or [])
    p.skills = list(p.skills or [])
    p.certifications = list(p.certifications or [])

    if intervention == "add_internship":
        p.internships.append({
            "company": "Industry Internship",
            "duration_months": 3,
            "is_top_tier": False,
            "relevance": 0.75,
        })
    elif intervention == "add_top_tier_internship":
        p.internships.append({
            "company": "Top-Tier Tech Co",
            "duration_months": 4,
            "is_top_tier": True,
            "relevance": 0.95,
        })
    elif intervention == "boost_cgpa":
        p.cgpa = min(9.5, float(p.cgpa or 0) + 0.5)
    elif intervention == "add_skill":
        existing = {s.lower() for s in p.skills}
        for s in _HIGH_DEMAND_SKILLS:
            if s not in existing and len(p.skills) < 12:
                p.skills.append(s)
                break
    elif intervention == "add_certification":
        existing = {c.lower() for c in p.certifications}
        for c in _RELEVANT_CERTS:
            if c.lower() not in existing:
                p.certifications.append(c)
                break
    elif intervention == "add_coding_problems":
        p.coding_problem_count = int(p.coding_problem_count or 0) + 200
    elif intervention == "add_hackathon_win":
        p.hackathon_wins = int(p.hackathon_wins or 0) + 1
    elif intervention == "boost_activity":
        p.portal_activity_30d = int(p.portal_activity_30d or 0) + 20
        p.interview_invites_count = int(p.interview_invites_count or 0) + 1
    elif intervention == "remove_backlog":
        p.backlogs_count = max(0, int(p.backlogs_count or 0) - 1)
    # Unknown interventions are silent no-ops (UI may add new ones).
    return p


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    """Slim view of a predict_profile result for diff display.

    Includes 'p_6m_raw' — the pre-IPR-blend calibrated model probability.
    The what-if delta is computed from raw probs so the institute anchor
    (which the student cannot change through effort) doesn't dampen the
    display of individual improvement potential.
    """
    pp   = result.get("placement_probabilities", {}) or {}
    sb   = result.get("salary_band_lpa", {}) or {}
    risk = result.get("risk", {}) or {}
    raw  = (result.get("ipr") or {}).get("raw_model") or {}
    return {
        "p_3m":  round(float(pp.get("p_3m", 0)),  3),
        "p_6m":  round(float(pp.get("p_6m", 0)),  3),
        "p_12m": round(float(pp.get("p_12m", 0)), 3),
        # raw model output before IPR anchor blending
        "p_6m_raw": round(float(raw.get("p_6m", pp.get("p_6m", 0))), 3),
        "salary_low":    float(sb.get("low",   0)),
        "salary_median": float(sb.get("median", 0)),
        "salary_high":   float(sb.get("high",   0)),
        "risk_score":    int(risk.get("score", 0)),
        "risk_tier":     str(risk.get("tier", "UNKNOWN")),
        "tier_color":    str(risk.get("tier_color", "#888")),
    }


def simulate(profile: StudentProfile,
             interventions: list[str],
             *,
             baseline_result: dict[str, Any] | None = None) -> dict[str, Any]:
    """Apply interventions, re-score, return before/after diff.

    `baseline_result`, if supplied, is the output of `predict_profile(profile)`
    that the caller already has — passing it skips one redundant prediction.
    """
    from .predict import predict_profile  # local import — heavy deps
    if baseline_result is None:
        baseline_result = predict_profile(profile)

    mutated = profile
    applied: list[dict[str, Any]] = []
    for iv in interventions:
        before = mutated
        mutated = _apply_one(mutated, iv)
        if mutated is not before:
            label_map = {x["id"]: x["label"] for x in INTERVENTION_CATALOG}
            applied.append({"id": iv, "label": label_map.get(iv, iv)})

    new_result = predict_profile(mutated) if applied else baseline_result  # noqa

    before = _summary(baseline_result)
    after_full = _summary(new_result)

    # ---------------------------------------------------------------
    # What-if delta strategy
    # ---------------------------------------------------------------
    # Two compression stages eat most of the raw model improvement:
    #   1. Beta calibration: compresses extremes (e.g. 0.875 → 0.863)
    #   2. IPR anchor blend: α*IPR + (1-α)*model, where α=0.20–0.70
    #      depending on institute quality. For a top institute at α=0.70
    #      only 30% of the model's signal passes through; at α=0.40 only 60%.
    #
    # For the What-If view the student wants to know: "If I do X, how much
    # does MY score improve?" — the IPR is the institute's historical average
    # and is unchanged by individual effort. So we report the RAW MODEL delta
    # (pre-IPR-blend) but anchor the displayed "after" to the blended baseline
    # so it remains consistent with the summary panel numbers at the extremes.
    #
    # Concretely:
    #   raw_delta   = after_raw_p6m − before_raw_p6m   (model's full opinion)
    #   after_p6m   = before_blended + raw_delta        (add onto realistic base)
    # This ensures the after bar is grounded in the user's realistic starting
    # point, while the delta reflects the full individual effort signal.
    raw_before = before.get("p_6m_raw", before["p_6m"])
    raw_after  = after_full.get("p_6m_raw", after_full["p_6m"])
    raw_delta_6m = raw_after - raw_before  # signed fraction

    # Build the "after" summary we'll actually display: keep salary / risk / etc.
    # from the full pipeline, but adjust p_6m by the unblended delta.
    after = dict(after_full)
    after["p_6m"] = round(min(0.99, max(0.01, before["p_6m"] + raw_delta_6m)), 3)

    # For 3m and 12m use the same ratio as 6m (keeps bars proportional)
    raw_before_3m  = (before.get("p_6m_raw", before["p_6m"]))  # no separate 3m raw exposed
    raw_delta_3m   = (after_full.get("p_3m", 0) - before.get("p_3m", 0))
    raw_delta_12m  = (after_full.get("p_12m", 0) - before.get("p_12m", 0))
    after["p_3m"]  = round(min(0.99, max(0.01, before["p_3m"]  + raw_delta_3m)),  3)
    after["p_12m"] = round(min(0.99, max(0.01, before["p_12m"] + raw_delta_12m)), 3)

    delta = {
        "p_3m_pp":      round((after["p_3m"]  - before["p_3m"])  * 100, 1),
        "p_6m_pp":      round(raw_delta_6m * 100, 1),   # unblended — shows real individual gain
        "p_12m_pp":     round((after["p_12m"] - before["p_12m"]) * 100, 1),
        "salary_lpa":   round(after["salary_median"] - before["salary_median"], 2),
        "risk_score":   after["risk_score"] - before["risk_score"],
        "tier_changed": after["risk_tier"] != before["risk_tier"],
    }

    # Clamp sub-1pp noise to zero (floating-point rounding at the ceiling)
    NOISE_FLOOR = 0.5  # pp
    if abs(delta["p_6m_pp"]) < NOISE_FLOOR:
        delta["p_6m_pp"] = 0.0
        delta["tier_changed"] = False

    direction = "improved" if delta["p_6m_pp"] > 0 else ("worsened" if delta["p_6m_pp"] < 0 else "unchanged")

    ceiling_note = ""
    if raw_before >= 0.90 and abs(delta["p_6m_pp"]) < 3.0:
        ceiling_note = " (already near the model ceiling — further gains diminish rapidly)"

    summary_sentence = (
        f"{', '.join(a['label'] for a in applied) or 'No-op'} "
        f"→ 6m placement {direction} by {abs(delta['p_6m_pp']):.1f}pp "
        f"({before['p_6m']*100:.0f}% → {after['p_6m']*100:.0f}%); "
        f"salary median {'+' if delta['salary_lpa']>=0 else ''}{delta['salary_lpa']:.1f}L."
        f"{ceiling_note}"
    )

    return {
        "applied": applied,
        "before": before,
        "after": after,
        "delta": delta,
        "narrative": summary_sentence,
    }


def catalog() -> list[dict[str, Any]]:
    """List all UI-exposable interventions."""
    return list(INTERVENTION_CATALOG)
