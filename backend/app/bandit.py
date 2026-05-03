"""Beta-Bernoulli Thompson Sampling bandit for the Co-Pilot action surface.

Replaces the hand-picked-uplift template list. Each candidate intervention
has a Beta(alpha, beta) posterior keyed on (action × segment), where the
segment is a (course-family × institute-tier × risk-tier) bucket. At
inference, we Thompson-sample one draw per action and sort by sampled mean;
the dashboard surfaces the top actions plus the posterior mean and 95%
credible interval of the uplift in percentage points.

Outcomes are POSTed to /api/record_outcome and update the posterior in
place (alpha += success, beta += 1 - success). Posteriors are persisted to
disk so the bandit state survives restarts.

Citations:
  Russo, Van Roy et al., A Tutorial on Thompson Sampling — arXiv:1707.02038.
  Chapelle & Li, An Empirical Evaluation of Thompson Sampling — NIPS 2011.
  Cortes, contextualbandits library — arXiv:1811.04383.

Regret bound for Beta-Bernoulli TS: O(sqrt(n K log K)) — sublinear.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_dist


# Six candidate actions a Co-Pilot can recommend. Adding new actions is safe —
# their posterior just starts at the prior.
ACTIONS = [
    {"id": "skill_cert",      "title": "SQL / Cloud Certification",   "detail": "Required by 78% of JDs in your target city."},
    {"id": "internship",      "title": "Apply to 3 Summer Internships","detail": "Each month of internship raises placement probability ~3pp."},
    {"id": "mock_interview",  "title": "Mock Interview Coaching",     "detail": "Schedule 4 mock rounds focused on case studies."},
    {"id": "portal_activity", "title": "Increase Portal Activity",    "detail": "Apply to 10+ matched roles per week."},
    {"id": "resume_clinic",   "title": "Resume Clinic - Domain Keywords","detail": "Add 5 high-signal keywords for your target sector."},
    {"id": "system_design",   "title": "Build 1 System-Design Project","detail": "Strong differentiator in BFSI / SaaS recruiter screens."},
]

# Warm-start priors are now seeded from the EconML DR ATE estimates (see
# counterfactual.py). We map the per-action point estimate (in pp) to a
# Beta(alpha, beta) prior with alpha + beta = 10 (informative but easy to
# override after a few real outcomes). If the bundle has no ATE artefact we
# fall back to the original hand-picked numbers.
_FALLBACK_PRIOR: dict[str, tuple[float, float]] = {
    "skill_cert":      (4.8, 5.2),   # ~12pp prior
    "internship":      (3.6, 6.4),   # ~9pp
    "mock_interview":  (2.4, 7.6),   # ~6pp
    "portal_activity": (2.0, 8.0),   # ~5pp
    "resume_clinic":   (1.6, 8.4),   # ~4pp
    "system_design":   (2.0, 8.0),   # ~5pp
}


def _load_priors_from_bundle() -> dict[str, tuple[float, float]]:
    """Pull DR-estimated ATEs from the model bundle if available."""
    try:
        from .predict import _bundle
        b = _bundle()
        ates = b.get("action_ates") or []
        if not ates:
            return _FALLBACK_PRIOR
        priors: dict[str, tuple[float, float]] = {}
        for r in ates:
            # ATE in pp -> success-rate scale by /25 (matches the old prior shape)
            p = max(0.05, min(0.95, float(r["dr_ate_pp"]) / 25.0))
            alpha = 10.0 * p
            beta = 10.0 * (1.0 - p)
            priors[r["action_id"]] = (alpha, beta)
        # Fill in any actions absent from the ATE artefact.
        for k, v in _FALLBACK_PRIOR.items():
            priors.setdefault(k, v)
        return priors
    except Exception:
        return _FALLBACK_PRIOR


_PRIOR = _load_priors_from_bundle()

_STATE_PATH = Path(__file__).resolve().parents[2] / "data" / "bandit_state.json"
_LOCK = threading.RLock()


@dataclass
class ActionRec:
    action_id: str
    title: str
    detail: str
    posterior_mean: float          # E[reward] = α / (α + β)
    ci_low: float                  # 2.5th percentile
    ci_high: float                 # 97.5th percentile
    n_trials: int                  # α + β - prior_strength (rounded)
    sampled: float                 # the Thompson draw used for ranking
    uplift_pp: float               # rounded posterior mean in pp
    uplift_ci_pp: tuple[float, float]
    impact: str                    # HIGH / MEDIUM / LOW
    expected_uplift_pp: int        # legacy field for the old UI


def _segment_key(profile_dict: dict, risk_tier: str) -> str:
    course = profile_dict.get("course_type", "BTech-CS")
    course_family = course.split("-")[0]
    tier = profile_dict.get("institute_tier", 2)
    return f"{course_family}_T{tier}_{risk_tier}"


def _load() -> dict:
    if _STATE_PATH.exists():
        try:
            return json.loads(_STATE_PATH.read_text())
        except Exception:
            return {}
    return {}


def _save(state: dict) -> None:
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STATE_PATH.write_text(json.dumps(state, indent=2))


def _get_posterior(state: dict, action_id: str, segment: str) -> tuple[float, float]:
    """Returns (alpha, beta), seeded with the action prior on first access."""
    seg_state = state.setdefault(segment, {})
    if action_id in seg_state:
        a, b = seg_state[action_id]
        return float(a), float(b)
    a0, b0 = _PRIOR.get(action_id, (1.0, 1.0))
    seg_state[action_id] = [a0, b0]
    return float(a0), float(b0)


def recommend(profile_dict: dict, risk_tier: str, top_k: int = 3) -> list[ActionRec]:
    """Return top-k actions by Thompson sampling, plus credible intervals."""
    seg = _segment_key(profile_dict, risk_tier)
    rng = np.random.default_rng()
    with _LOCK:
        state = _load()
        recs: list[ActionRec] = []
        for action in ACTIONS:
            a, b = _get_posterior(state, action["id"], seg)
            # Thompson sample
            theta = float(rng.beta(a, b))
            # Posterior mean and 95% credible interval (in success-rate units)
            mean = a / (a + b)
            ci_low = float(beta_dist.ppf(0.025, a, b))
            ci_high = float(beta_dist.ppf(0.975, a, b))
            # Convert to percentage-point uplift (× 25 keeps the same scale as the
            # old UI's hand-picked numbers).
            uplift_pp = mean * 25.0
            ci_low_pp = ci_low * 25.0
            ci_high_pp = ci_high * 25.0
            # Trial count = total beats - prior strength
            prior_a, prior_b = _PRIOR.get(action["id"], (1.0, 1.0))
            n_trials = max(0, int(round((a + b) - (prior_a + prior_b))))
            recs.append(ActionRec(
                action_id=action["id"],
                title=action["title"],
                detail=action["detail"],
                posterior_mean=mean,
                ci_low=ci_low,
                ci_high=ci_high,
                n_trials=n_trials,
                sampled=theta,
                uplift_pp=uplift_pp,
                uplift_ci_pp=(ci_low_pp, ci_high_pp),
                impact=("HIGH" if uplift_pp >= 9 else ("MEDIUM" if uplift_pp >= 4 else "LOW")),
                expected_uplift_pp=int(round(uplift_pp)),
            ))
        _save(state)
    # Sort by Thompson-sampled value (exploration + exploitation)
    recs.sort(key=lambda r: r.sampled, reverse=True)
    return recs[:top_k]


def record(segment: str, action_id: str, success: int) -> dict:
    """Update the posterior for (action, segment) with a 0/1 success."""
    success = 1 if int(success) >= 1 else 0
    with _LOCK:
        state = _load()
        a, b = _get_posterior(state, action_id, segment)
        a += success
        b += (1 - success)
        state[segment][action_id] = [a, b]
        _save(state)
    return {"segment": segment, "action_id": action_id, "alpha": a, "beta": b,
            "posterior_mean": a / (a + b)}


def state_summary() -> dict:
    """Return aggregated bandit state for the Architecture tab."""
    with _LOCK:
        state = _load()
    segments = []
    for seg, actions in state.items():
        rows = []
        for action_id, ab in actions.items():
            a, b = ab
            mean = a / (a + b)
            prior_a, prior_b = _PRIOR.get(action_id, (1.0, 1.0))
            n_trials = max(0, int(round((a + b) - (prior_a + prior_b))))
            rows.append({
                "action_id": action_id,
                "alpha": float(a),
                "beta": float(b),
                "posterior_mean": float(mean),
                "n_trials": int(n_trials),
                "ci_low": float(beta_dist.ppf(0.025, a, b)),
                "ci_high": float(beta_dist.ppf(0.975, a, b)),
            })
        rows.sort(key=lambda r: -r["posterior_mean"])
        segments.append({"segment": seg, "actions": rows})
    segments.sort(key=lambda s: -sum(r["n_trials"] for r in s["actions"]))
    return {"n_segments": len(segments), "segments": segments[:25]}
