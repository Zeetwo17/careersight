"""Population Stability Index + KS drift detection for the feature store.

Production credit-risk teams speak PSI (Siddiqi 2006, Credit Risk Scorecards),
not p-values. This module computes both:

  - PSI per feature against a stored reference distribution.
    Thresholds (industry-canonical, re-validated by Liu et al. 2024):
      < 0.10  stable
      0.10-0.25  moderate, monitor
      > 0.25  significant drift, retrain
  - Kolmogorov-Smirnov two-sample test per continuous feature
    (Bonferroni-corrected for multi-feature comparison).

The reference distribution is the training-time feature matrix; the live
distribution is whatever the lender currently has in production. For the
demo we generate a "current" distribution by perturbing the reference
with a sector-shock + cohort-shift, large enough to put a few features
into the moderate band so the dashboard has something to show.

Citations:
  Siddiqi, Credit Risk Scorecards (Wiley 2006) — PSI formula and thresholds.
  Rabanser, Günnemann, Lipton, Failing Loudly — arXiv:1810.11953 (NeurIPS 2019).
  Liu et al., Concept Drift Adaptation for Credit Scoring — arXiv:2305.18092.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .features import expand_dataframe
from .schema import ALL_FEATURE_COLUMNS


PSI_STABLE = 0.10
PSI_MODERATE = 0.25


@dataclass
class FeatureDrift:
    feature: str
    psi: float
    ks_stat: float
    ks_pvalue: float        # Bonferroni-corrected
    band: str               # "stable" / "moderate" / "drift"

    def to_dict(self) -> dict:
        return {
            "feature": self.feature,
            "psi": float(self.psi),
            "ks_stat": float(self.ks_stat),
            "ks_pvalue": float(self.ks_pvalue),
            "band": self.band,
        }


def _psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two 1-D samples."""
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if len(reference) == 0 or len(current) == 0:
        return 0.0
    # Quantile bins from reference; if reference has near-zero variance, return 0.
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    # Pad edges to cover anything outside reference range
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref_p = (ref_counts + 1e-6) / (ref_counts.sum() + 1e-6 * len(ref_counts))
    cur_p = (cur_counts + 1e-6) / (cur_counts.sum() + 1e-6 * len(cur_counts))
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def _band(psi: float) -> str:
    if psi < PSI_STABLE:
        return "stable"
    if psi < PSI_MODERATE:
        return "moderate"
    return "drift"


def compute_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> list[FeatureDrift]:
    """Return per-feature PSI + KS drift metrics, sorted by PSI descending."""
    ref = expand_dataframe(reference_df).astype(float).values
    cur = expand_dataframe(current_df).astype(float).values
    if cur.shape[1] != ref.shape[1]:
        raise ValueError("reference / current feature counts disagree")

    n_features = ref.shape[1]
    bonferroni = max(1, n_features)

    results: list[FeatureDrift] = []
    for j in range(n_features):
        ref_col, cur_col = ref[:, j], cur[:, j]
        psi = _psi(ref_col, cur_col)
        try:
            ks = stats.ks_2samp(ref_col, cur_col, alternative="two-sided", method="asymp")
            ks_stat, ks_p = float(ks.statistic), float(ks.pvalue)
        except Exception:
            ks_stat, ks_p = 0.0, 1.0
        ks_p_bonf = min(1.0, ks_p * bonferroni)
        results.append(FeatureDrift(
            feature=ALL_FEATURE_COLUMNS[j],
            psi=psi,
            ks_stat=ks_stat,
            ks_pvalue=ks_p_bonf,
            band=_band(psi),
        ))
    results.sort(key=lambda r: r.psi, reverse=True)
    return results


def synthetic_current_cohort(reference_df: pd.DataFrame, seed: int = 17) -> pd.DataFrame:
    """Generate a 'current' cohort with a realistic drift signature.

    Designed to put 4-6 features into the moderate band and 1-2 into the
    drift band, so the dashboard has visible drift colors at startup.
    """
    rng = np.random.default_rng(seed)
    df = reference_df.sample(n=min(2500, len(reference_df)), random_state=seed).copy()

    # Sector shock: BFSI hiring slowdown (matches deck's narrative)
    bfsi_mask = df["course_type"].str.startswith("MBA-Finance")
    df.loc[bfsi_mask, "sector_hiring_index"] *= rng.uniform(0.55, 0.70, bfsi_mask.sum())
    df.loc[bfsi_mask, "course_demand_index"] = df.loc[bfsi_mask, "sector_hiring_index"]

    # Macro: unemployment up
    df["macro_unemployment_rate"] = (df["macro_unemployment_rate"] + 1.8).clip(3, 15)

    # Behavioral: portal activity falling (cohort-wide demoralisation)
    df["portal_activity_30d"] = (df["portal_activity_30d"] * rng.uniform(0.6, 0.85, len(df))).round().astype(int)
    df["portal_activity_90d"] = (df["portal_activity_90d"] * rng.uniform(0.6, 0.85, len(df))).round().astype(int)

    # Skills: certification rates rising (good drift, but still drift)
    df["relevant_certs_count"] = (df["relevant_certs_count"] * rng.uniform(1.2, 1.5, len(df))).round().astype(int)

    # Tier-1 share down (real-world cohort change)
    tier1_mask = df["institute_tier"] == 1
    drop_tier1 = rng.uniform(size=tier1_mask.sum()) < 0.4
    df = df[~(tier1_mask & np.append(drop_tier1, [False] * (len(df) - tier1_mask.sum())))[:len(df)]]

    return df.reset_index(drop=True)
