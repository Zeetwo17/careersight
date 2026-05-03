"""Beta calibration for the 3 LightGBM placement classifiers.

LightGBM with `is_unbalance` / `class_weight` is systematically miscalibrated
on imbalanced binary tasks (Caplin, Martin, Marx, arXiv:2205.04613). For
class-imbalanced labels, Platt sigmoid scaling is the wrong functional form;
Beta calibration (Kull, Silva Filho, Flach — AISTATS 2017) fits a richer
3-parameter family that stays inside [0, 1] and respects the data's natural
skew.

We fit one BetaCalibration per horizon on a held-out 20% calibration fold,
then apply at inference. We also report Brier score, ECE@15 bins, and a
reliability curve so the dashboard can surface a "well-calibrated" badge.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from netcal.metrics import ECE
from netcal.scaling import BetaCalibration
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class CalibrationReport:
    """Per-horizon calibration metrics, persisted next to the LightGBM head."""
    ece_pre: float        # ECE on raw classifier probabilities
    ece_post: float       # ECE on Beta-calibrated probabilities
    brier_pre: float      # Brier on raw
    brier_post: float     # Brier on calibrated
    n_calibration: int    # Size of held-out calibration fold
    reliability_curve: list[dict]  # [{p_pred, p_true, count}, ...] for the dashboard

    def to_dict(self) -> dict:
        return {
            "ece_pre": float(self.ece_pre),
            "ece_post": float(self.ece_post),
            "brier_pre": float(self.brier_pre),
            "brier_post": float(self.brier_post),
            "n_calibration": int(self.n_calibration),
            "reliability_curve": self.reliability_curve,
        }


def fit_beta_calibrator(p_uncal: np.ndarray, y: np.ndarray) -> tuple[BetaCalibration, CalibrationReport]:
    """Fit Beta calibration on (p_uncal, y) pairs and report before/after metrics."""
    p_uncal = np.asarray(p_uncal, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=int).reshape(-1)

    beta = BetaCalibration()
    beta.fit(p_uncal, y)
    p_cal = beta.transform(p_uncal)

    ece_metric = ECE(15)  # 15 quantile bins per Guo et al. 2017
    ece_pre = float(ece_metric.measure(p_uncal, y))
    ece_post = float(ece_metric.measure(p_cal, y))
    brier_pre = float(brier_score_loss(y, p_uncal))
    brier_post = float(brier_score_loss(y, p_cal))

    # Reliability curve (10 bins, used by the dashboard)
    frac_pos, mean_pred = calibration_curve(y, p_cal, n_bins=10, strategy="quantile")
    bin_counts, _ = np.histogram(p_cal, bins=10)
    rc = [
        {"p_pred": float(mp), "p_true": float(fp), "count": int(c)}
        for mp, fp, c in zip(mean_pred, frac_pos, bin_counts)
    ]

    return beta, CalibrationReport(
        ece_pre=ece_pre, ece_post=ece_post,
        brier_pre=brier_pre, brier_post=brier_post,
        n_calibration=len(y),
        reliability_curve=rc,
    )


def apply_calibrator(beta: BetaCalibration, p_uncal: np.ndarray) -> np.ndarray:
    """Apply a fitted Beta calibrator to raw classifier probabilities."""
    p_uncal = np.asarray(p_uncal, dtype=float).reshape(-1)
    p_cal = beta.transform(p_uncal)
    # Beta calibration can produce small negatives in pathological cases — clip.
    return np.clip(p_cal, 1e-6, 1 - 1e-6)
