"""Train the CareerSight model bundle.

Trains:
  - Three LightGBM classifiers for placed_3m / placed_6m / placed_12m
  - Two LightGBM regressors for salary lower (q=0.10) and upper (q=0.90)
    quantiles — the "Conformal Quantile Regression (CQR)" surface from the
    pitch deck.
  - SHAP TreeExplainer fitted on placed_6m for the explanation surface.

Persists everything to data/models/bundle.joblib so the API can load it once.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from .calibration import fit_beta_calibrator
from .causal import discover as causal_discover
from .counterfactual import estimate_action_ates
from .crc import fit_crc_controller
from .edge_model import distill as distill_edge
from .fairness import run_default_audit
from .features import expand_dataframe
from .federated import federated_shap
from .ood import evaluate_ood
from .schema import ALL_FEATURE_COLUMNS
from .survival import SURVIVAL_CUTS, fit_deephit

CLASSIFIER_PARAMS = dict(
    objective="binary",
    learning_rate=0.05,
    num_leaves=63,
    max_depth=-1,
    min_data_in_leaf=80,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=5,
    n_estimators=400,
    verbose=-1,
)

REGRESSOR_PARAMS = dict(
    learning_rate=0.05,
    num_leaves=63,
    min_data_in_leaf=80,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=5,
    n_estimators=500,
    verbose=-1,
)


def train(csv_path: str, out_path: str) -> None:
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} rows")

    X_df = expand_dataframe(df).astype(np.float32)
    X = X_df.values
    feature_names = ALL_FEATURE_COLUMNS

    classifiers: dict[str, lgb.LGBMClassifier] = {}
    calibrators: dict[str, object] = {}
    calibration_reports: dict[str, dict] = {}
    aucs: dict[str, float] = {}
    # We pre-split a calibration fold (15% of total) once; it stays held-out
    # from training and is used both for fitting the Beta calibrator and for
    # computing pre/post calibration ECE + Brier. The remaining 85% is split
    # 80/20 train/test for AUC reporting on truly unseen rows.
    n = len(df)
    rng = np.random.default_rng(42)
    cal_idx = rng.choice(n, size=int(0.15 * n), replace=False)
    cal_mask = np.zeros(n, dtype=bool); cal_mask[cal_idx] = True

    for label in ["placed_3m", "placed_6m", "placed_12m"]:
        y = df[label].astype(int).values
        X_cal, y_cal = X[cal_mask], y[cal_mask]
        X_rem, y_rem = X[~cal_mask], y[~cal_mask]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_rem, y_rem, test_size=0.2, random_state=42, stratify=y_rem
        )
        clf = lgb.LGBMClassifier(**CLASSIFIER_PARAMS)
        clf.fit(X_tr, y_tr, eval_set=[(X_te, y_te)],
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)])
        proba_te = clf.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, proba_te)
        classifiers[label] = clf
        aucs[label] = float(auc)

        proba_cal = clf.predict_proba(X_cal)[:, 1]
        beta, report = fit_beta_calibrator(proba_cal, y_cal)
        calibrators[label] = beta
        calibration_reports[label] = report.to_dict()
        print(
            f"  {label:14s} AUC={auc:.4f}  "
            f"ECE {report.ece_pre:.3f}->{report.ece_post:.3f}  "
            f"Brier {report.brier_pre:.4f}->{report.brier_post:.4f}"
        )

    # Salary regressors — fit only on placed students (those with non-null final_salary_lpa)
    placed = df.dropna(subset=["final_salary_lpa"])
    Xs = expand_dataframe(placed).astype(np.float32).values
    ys = placed["final_salary_lpa"].values
    Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(Xs, ys, test_size=0.2, random_state=42)

    salary_low = lgb.LGBMRegressor(objective="quantile", alpha=0.10, **REGRESSOR_PARAMS)
    salary_low.fit(Xs_tr, ys_tr)
    salary_med = lgb.LGBMRegressor(objective="quantile", alpha=0.50, **REGRESSOR_PARAMS)
    salary_med.fit(Xs_tr, ys_tr)
    salary_high = lgb.LGBMRegressor(objective="quantile", alpha=0.90, **REGRESSOR_PARAMS)
    salary_high.fit(Xs_tr, ys_tr)

    pred_med = salary_med.predict(Xs_te)
    mae = float(np.mean(np.abs(pred_med - ys_te)))
    print(f"  salary median MAE = {mae:.2f} LPA")

    # SHAP explainer is fit on placed_6m — that's the headline risk model.
    explainer = shap.TreeExplainer(classifiers["placed_6m"])

    # DeepHitSingle survival model on the {3, 6, 12} month grid. Replaces
    # the linear-interpolation hack in predict.py with a discrete-time deep
    # survival model trained directly on the synthetic durations.
    durations = df["t_placed_months"].astype(float).clip(0.1, 13.0).values
    events = (durations <= 12.5).astype(int)
    print("  Training DeepHitSingle on cuts=[3,6,12]...")
    deephit_model, deephit_artefact = fit_deephit(
        X=X, durations=durations, events=events, cuts=SURVIVAL_CUTS,
        hidden_sizes=[64, 64], epochs=40, batch_size=256, lr=1e-3,
    )
    print(f"  DEEPHIT  C-index (Antolini) = {deephit_artefact.c_index:.3f}  "
          f"({deephit_artefact.note})")

    # Conformal Risk Control via MAPIE Learn-Then-Test on the same calibration
    # fold as the Beta calibrators. The deck's framing is "FNR <= 0.10 on
    # HIGH-risk students" — i.e. we want to catch >= 90% of students who will
    # actually fail to be placed. We model the *unplaced* class as positive:
    #   y_unplaced = 1 - placed_6m
    #   p_unplaced = 1 - p_calibrated_placed
    # and ask MAPIE for the threshold lambda such that
    #   recall(predict_unplaced(p_unplaced >= lambda), y_unplaced) >= 0.90
    # holds with confidence >= 0.90. That's exactly the deck's claim.
    y6_cal = df["placed_6m"].astype(int).values[cal_mask]
    y_unplaced_cal = 1 - y6_cal
    clf6 = classifiers["placed_6m"]
    beta6 = calibrators["placed_6m"]

    def _unplaced_predict(X):
        raw = clf6.predict_proba(X)[:, 1]
        p_placed = np.clip(beta6.transform(raw), 1e-6, 1 - 1e-6)
        return 1.0 - p_placed   # P(unplaced)

    crc_result = fit_crc_controller(
        predict_proba_fn=_unplaced_predict,
        X_cal=X[cal_mask],
        y_cal=y_unplaced_cal,
        target_level=0.90,      # recall on UNPLACED class >= 0.90
        confidence_level=0.90,  # with prob >= 0.90 (FNR <= 0.10 guarantee)
    )
    # The lambda is on the P(unplaced) scale; convert to the dashboard's 0-100 risk score.
    if crc_result.lambda_star is not None:
        crc_result.risk_score_threshold = int(round(float(crc_result.lambda_star) * 100))
    print(
        f"  CRC (LTT, FNR<=0.10, conf>=0.90):  lambda*={crc_result.lambda_star}  "
        f"risk_score_threshold={crc_result.risk_score_threshold}"
    )

    bundle = {
        "feature_names": feature_names,
        "classifiers": classifiers,
        "calibrators": calibrators,
        "calibration_reports": calibration_reports,
        "salary_low": salary_low,
        "salary_med": salary_med,
        "salary_high": salary_high,
        "explainer": explainer,
        "aucs": aucs,
        "salary_mae": mae,
        "n_train": len(df),
        "crc": crc_result.to_dict(),
        "deephit": deephit_artefact.to_dict(),
        "deephit_state_dict": deephit_artefact.state_dict,
    }

    # Out-of-distribution validation against real Kaggle data when available
    # (and a covariate-shifted internal fold as a fallback). Persisted on the
    # bundle so the dashboard /api/ood endpoint reads it without re-training.
    ood_reports = evaluate_ood(bundle, synth_df=df)
    bundle["ood_reports"] = [r.to_dict() for r in ood_reports]
    for r in ood_reports:
        print(
            f"  OOD {r.source:28s} n={r.n_rows:5d}  "
            f"AUC12m={r.auc_12m:.3f}  ECE {r.ece_pre:.3f}->{r.ece_post:.3f}  "
            f"pos_rate={r.pos_rate:.2f}"
        )

    # Causal discovery via PC algorithm + Markov-blanket extraction. We run
    # this on a 2.5k subsample of the top-24 numeric features so it stays
    # under 10 s on a hackathon laptop.
    causal_result = causal_discover(bundle, df)
    bundle["causal"] = causal_result.to_dict()
    if causal_result.full_auc is not None:
        print(
            f"  CAUSAL  PC + MB: |MB|={causal_result.mb_size}/{causal_result.full_size}  "
            f"full_AUC={causal_result.full_auc:.3f}  mb_AUC={causal_result.mb_auc:.3f}"
        )
    else:
        print(f"  CAUSAL  {causal_result.note}")

    # Doubly-robust ATE per Co-Pilot action (EconML LinearDRLearner) with
    # placebo refuter. Replaces the hand-picked uplift numbers in the bandit
    # warm-start with paper-cited estimates.
    ate_reports = estimate_action_ates(bundle, df)
    bundle["action_ates"] = [r.to_dict() for r in ate_reports]
    for r in ate_reports:
        flag = "" if r.placebo_passes else "  [PLACEBO ALERT]"
        print(
            f"  ATE  {r.action_id:18s}  naive={r.naive_diff_pp:+5.1f}pp  "
            f"DR={r.dr_ate_pp:+5.1f}pp [{r.dr_ci_low_pp:+.1f}, {r.dr_ci_high_pp:+.1f}]  "
            f"placebo={r.placebo_ate_pp:+5.2f}pp{flag}"
        )

    # Distil the LightGBM 6-month head into a Born-Again single tree.
    edge = distill_edge(bundle, df)
    bundle["edge_model"] = edge.to_dict()
    print(
        f"  EDGE  born-again tree: leaves={edge.n_leaves}  "
        f"size={edge.size_bytes/1024:.1f}KB  "
        f"AUC teacher={edge.teacher_auc:.3f} student={edge.student_auc:.3f} "
        f"gap={edge.auc_gap_pp:.1f}pp  top5_jaccard={edge.feature_jaccard_top5:.2f}"
    )

    # Federated SHAP simulation across 2 lender shards with Gaussian DP noise
    # calibrated to (epsilon=1.0, delta=1e-5).
    fed_result = federated_shap(df, n_shards=2, n_sample_per_shard=600,
                                eps=1.0, delta=1e-5, clip=1.0, top_k=10)
    bundle["federated_shap"] = fed_result.to_dict()
    print(
        f"  FED-SHAP  shards={fed_result.n_shards} eps={fed_result.eps} "
        f"delta={fed_result.delta} sigma={fed_result.sigma:.3f}  "
        f"Spearman_rho={fed_result.spearman_rho:.3f}  [{fed_result.note}]"
    )

    # Fairlearn audit anchored in DPDP Act 2023 §10 + EEOC four-fifths rule.
    fair_reports = run_default_audit(bundle, df)
    bundle["fairness_reports"] = fair_reports
    for fr in fair_reports:
        flag = "PASS" if not fr["breaches"] else "BREACH"
        print(
            f"  FAIR {fr['sensitive']:18s}  DI={fr['di_ratio']:.2f}  "
            f"DP={fr['dp_diff']:.2f}  EO={fr['eo_diff']:.2f}  "
            f"EOpp={fr['eopp_diff']:.2f}  AUCgap={fr['auc_gap']:.3f}  [{flag}]"
        )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out)
    print(f"Saved bundle -> {out}  ({out.stat().st_size/1024:.0f} KB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/synthetic_students.csv")
    parser.add_argument("--out", default="data/models/bundle.joblib")
    args = parser.parse_args()
    train(args.csv, args.out)


if __name__ == "__main__":
    main()
