"""Peer benchmarking — contextualises a single prediction against students
"like this one."

Approach: we hold the synthetic_students.csv training corpus in memory (it's
already loaded for drift / OOD work) and slice it on three axes:
  1. course_type  (exact match)
  2. cgpa_band    (the student's 1.0-wide CGPA window, e.g. 7.0-8.0)
  3. tier         (institute tier, exact)

If that cohort is too small (< 30 rows) we fall back to course + tier only,
then to course only. Every output carries the actual cohort size so the
caller can show how trustworthy the comparison is.

What we extract per cohort:
  - placement_rate at 3 / 6 / 12 months
  - median + p25/p75 final salary in LPA
  - "expected" months to placement (linearly interpolated from the three
    rates — same trick predict.py uses elsewhere)
  - the student's percentile against each metric

The output is built so the UI can render direct comparisons:
  "Peer median placement: 5 months / Yours: 7 months"
  "You are ahead of 64% of peers on salary"
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .schema import StudentProfile


_CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "synthetic_students.csv"


@lru_cache(maxsize=1)
def _df() -> pd.DataFrame:
    if not _CSV_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(_CSV_PATH)


def _months_to_placement(p3: float, p6: float, p12: float) -> float:
    """Expected months-to-placement assuming the three rates are CDF samples.

    We integrate the survival curve numerically using the three anchors
    (linear ramps between them), then return E[T] over [0, 12].
    """
    pts = [(0, 0.0), (3, p3), (6, p6), (12, p12)]
    # E[T] = ∫ (1 - F(t)) dt over [0, 12], plus a tail contribution for the
    # students who don't get placed within 12 months (we conservatively
    # charge them 18 months on average).
    e_t = 0.0
    for (t0, f0), (t1, f1) in zip(pts[:-1], pts[1:]):
        # ∫ (1 - F(t)) dt over [t0, t1] using trapezoid rule on (1-F)
        e_t += 0.5 * (t1 - t0) * ((1 - f0) + (1 - f1))
    # Tail: students unplaced at 12m. Charge 18m average for them.
    unplaced_12 = max(0.0, 1.0 - p12)
    e_t += unplaced_12 * 6.0
    return float(e_t)


def _percentile_below(series: pd.Series, value: float) -> float:
    """Fraction of cohort below `value`. Returns [0, 1]."""
    if len(series) == 0:
        return 0.5
    # use standard "strictly less than" percentile
    arr = series.dropna().to_numpy()
    if len(arr) == 0:
        return 0.5
    return float((arr < value).sum() / len(arr))


def _slice_peers(df: pd.DataFrame, profile: StudentProfile) -> tuple[pd.DataFrame, str, str]:
    """Pick the largest peer slice that is still narrow enough to be useful.

    Returns: (cohort_df, level_label, definition_str)
    """
    course = profile.course_type
    tier = int(profile.institute_tier or 2)
    cgpa = float(profile.cgpa or 0)
    # 1.0-wide window around CGPA, snapped to integer grid
    cgpa_lo = max(0.0, np.floor(cgpa))
    cgpa_hi = cgpa_lo + 1.0

    # Level 1: course + cgpa-band + tier
    m1 = df[
        (df["course_type"] == course) &
        (df["tier"] == tier) &
        (df["cgpa"] >= cgpa_lo) &
        (df["cgpa"] < cgpa_hi)
    ]
    if len(m1) >= 30:
        return m1, "L1", f"{course} · Tier-{tier} · CGPA {cgpa_lo:.0f}-{cgpa_hi:.0f}"

    # Level 2: course + tier
    m2 = df[
        (df["course_type"] == course) &
        (df["tier"] == tier)
    ]
    if len(m2) >= 30:
        return m2, "L2", f"{course} · Tier-{tier} (any CGPA)"

    # Level 3: course only
    m3 = df[df["course_type"] == course]
    if len(m3) >= 30:
        return m3, "L3", f"{course} (any tier, any CGPA)"

    # Level 4: full population baseline
    return df, "L4", "All graduates (national baseline)"


def benchmark_profile(profile: StudentProfile, predict_result: dict[str, Any]) -> dict[str, Any]:
    """Compare this student's predicted outcomes to their peer cohort."""
    df = _df()
    if df.empty:
        return {"available": False, "note": "synthetic_students.csv not found"}

    cohort, level, definition = _slice_peers(df, profile)

    p3 = float(cohort["placed_3m"].mean())
    p6 = float(cohort["placed_6m"].mean())
    p12 = float(cohort["placed_12m"].mean())

    salary = cohort["final_salary_lpa"].dropna()
    salary_p25 = float(salary.quantile(0.25)) if len(salary) else 0.0
    salary_med = float(salary.quantile(0.50)) if len(salary) else 0.0
    salary_p75 = float(salary.quantile(0.75)) if len(salary) else 0.0

    peer_months = _months_to_placement(p3, p6, p12)

    # Student's predicted side
    pp = predict_result.get("placement_probabilities", {})
    sp3 = float(pp.get("p_3m", 0.0))
    sp6 = float(pp.get("p_6m", 0.0))
    sp12 = float(pp.get("p_12m", 0.0))
    student_months = _months_to_placement(sp3, sp6, sp12)

    sb = predict_result.get("salary_band_lpa", {})
    student_salary_med = float(sb.get("median", 0.0))

    # Percentile rank within peers
    salary_pct = _percentile_below(salary, student_salary_med) if len(salary) else 0.5
    # speed percentile = % of peers SLOWER than the student
    # so we need to compute time-to-placement per peer row, but we don't have
    # that directly — t_placed_months IS in the CSV
    t_col = cohort["t_placed_months"].dropna()
    if len(t_col):
        speed_pct = float((t_col > student_months).sum() / len(t_col))
    else:
        speed_pct = 0.5

    # Top-25% peer placement time (fast lane)
    if len(t_col):
        fast_lane_months = float(t_col.quantile(0.25))
    else:
        fast_lane_months = peer_months

    delta_months = student_months - peer_months
    delta_salary = student_salary_med - salary_med

    return {
        "available": True,
        "level": level,
        "definition": definition,
        "cohort_size": int(len(cohort)),
        "peer": {
            "p_3m":        round(p3,  3),
            "p_6m":        round(p6,  3),
            "p_12m":       round(p12, 3),
            "median_months_to_placement": round(peer_months, 1),
            "fast_lane_months_top25":     round(fast_lane_months, 1),
            "salary_p25_lpa":   round(salary_p25, 1),
            "salary_p50_lpa":   round(salary_med, 1),
            "salary_p75_lpa":   round(salary_p75, 1),
        },
        "student": {
            "p_3m":  round(sp3,  3),
            "p_6m":  round(sp6,  3),
            "p_12m": round(sp12, 3),
            "expected_months_to_placement": round(student_months, 1),
            "median_salary_lpa": round(student_salary_med, 1),
        },
        "comparison": {
            "delta_months":  round(delta_months, 1),
            "delta_salary":  round(delta_salary, 1),
            "speed_percentile":  round(speed_pct, 2),
            "salary_percentile": round(salary_pct, 2),
            "speed_label": _speed_label(delta_months),
            "salary_label": _salary_label(delta_salary),
            "narrative": _narrative(delta_months, delta_salary, speed_pct, salary_pct),
        },
    }


def _speed_label(delta_months: float) -> str:
    if delta_months <= -1.5: return "Faster than peers"
    if delta_months <  0.5:  return "On pace with peers"
    if delta_months <  2.5:  return "Slightly behind peers"
    return "Well behind peers"


def _salary_label(delta_salary: float) -> str:
    if delta_salary >= 3:    return "Above peer median"
    if delta_salary >= -1.5: return "On peer median"
    return "Below peer median"


def _narrative(delta_months: float, delta_salary: float,
               speed_pct: float, salary_pct: float) -> str:
    speed = (
        f"placed {abs(delta_months):.1f} mo faster than peers"
        if delta_months < -0.3 else
        f"placed {delta_months:.1f} mo slower than peers"
        if delta_months > 0.3 else
        "on the peer pace"
    )
    salary = (
        f"₹{abs(delta_salary):.1f}L above peer median"
        if delta_salary > 0.5 else
        f"₹{abs(delta_salary):.1f}L below peer median"
        if delta_salary < -0.5 else
        "matching peer median"
    )
    return (f"You are projected {speed}, "
            f"with salary {salary} "
            f"(beats {int(round(salary_pct * 100))}% of peers).")
