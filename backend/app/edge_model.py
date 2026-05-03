"""Born-Again single decision tree distilled from the LightGBM teacher.

Vidal & Schiffer 2020 (arXiv:2003.11132) showed a single tree, fit with
soft labels and confidence-weighted samples, can recover 95-99% of a GBDT
ensemble's AUC on common tabular datasets at 30-100x compression. Deck
claim #7 (47KB edge model, "Risk Score on WhatsApp Without Internet")
becomes a real artefact: a JSON-serialisable decision tree shipped via
/api/edge_model, plus a tiny pure-Python evaluator embeddable in any
WhatsApp bot or PWA.

Quantisation: thresholds scaled by 2^15 and stored as int16 to halve the
JSON payload. The evaluator de-quantises on the fly.

Citations:
  Vidal & Schiffer, Born-Again Tree Ensembles — ICML 2020 (arXiv:2003.11132).
  Sagi & Rokach, Approximating XGBoost with an Interpretable Decision Tree —
    Information Sciences 572:522-542, 2021 (DOI 10.1016/j.ins.2021.05.055).
  Hinton, Vinyals, Dean — arXiv:1503.02531.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from .features import expand_dataframe


@dataclass
class EdgeModelArtefact:
    """JSON-shaped distilled tree ready to be served from /api/edge_model."""
    feature_names: list[str]
    feat: list[int]      # node feature index (-2 = leaf)
    thr_q: list[int]     # int16 quantised threshold (raw_thr * 2^15)
    lc: list[int]        # left child index (-1 if leaf)
    rc: list[int]        # right child index (-1 if leaf)
    val: list[float]     # leaf probability of placed=1
    teacher_auc: float
    student_auc: float
    auc_gap_pp: float
    feature_jaccard_top5: float
    n_leaves: int
    size_bytes: int

    def to_dict(self) -> dict:
        return {
            "feature_names": self.feature_names,
            "feat": self.feat,
            "thr_q": self.thr_q,
            "lc": self.lc,
            "rc": self.rc,
            "val": self.val,
            "teacher_auc": float(self.teacher_auc),
            "student_auc": float(self.student_auc),
            "auc_gap_pp": float(self.auc_gap_pp),
            "feature_jaccard_top5": float(self.feature_jaccard_top5),
            "n_leaves": int(self.n_leaves),
            "size_bytes": int(self.size_bytes),
        }


def distill(bundle, df: pd.DataFrame, max_leaf_nodes: int = 128) -> EdgeModelArtefact:
    """Train the Born-Again single tree."""
    X_full = expand_dataframe(df).astype(np.float32).values
    y = df["placed_6m"].astype(int).values

    teacher = bundle["classifiers"]["placed_6m"]
    soft = teacher.predict_proba(X_full)[:, 1]

    X_tr, X_te, y_tr, y_te, soft_tr, soft_te = train_test_split(
        X_full, y, soft, test_size=0.2, random_state=42, stratify=y
    )

    # Confidence-weighted samples: data points where the teacher is far from
    # 0.5 are easier to mimic, so weight them more.
    sample_w = np.abs(soft_tr - 0.5) + 1e-3

    student = DecisionTreeClassifier(
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=20,
        criterion="entropy",
        random_state=0,
    )
    # Train against the soft-label-thresholded class but with confidence weights.
    student.fit(X_tr, (soft_tr > 0.5).astype(int), sample_weight=sample_w)

    student_auc = roc_auc_score(y_te, student.predict_proba(X_te)[:, 1])
    teacher_auc = roc_auc_score(y_te, teacher.predict_proba(X_te)[:, 1])

    feature_names = bundle["feature_names"]
    feat = [int(f) for f in student.tree_.feature.tolist()]
    thr = student.tree_.threshold.tolist()
    thr_q = [int(np.clip(round(t * 32767), -32768, 32767)) if f != -2 else 0
             for t, f in zip(thr, feat)]
    lc = [int(c) for c in student.tree_.children_left.tolist()]
    rc = [int(c) for c in student.tree_.children_right.tolist()]
    # Leaf value = P(positive class). For internal nodes we still write the
    # majority-class prob so the JSON shape is regular.
    values = student.tree_.value.squeeze()
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.shape[1] == 2:
        prob_positive = values[:, 1] / values.sum(axis=1).clip(min=1)
    else:
        prob_positive = values[:, 0]
    val = [float(p) for p in prob_positive.tolist()]

    # Top-5 feature Jaccard between teacher importance and student top splits.
    teacher_imp = teacher.booster_.feature_importance(importance_type="gain")
    teacher_top = set(np.argsort(teacher_imp)[::-1][:5].tolist())
    student_use = [f for f in feat if f >= 0]
    if student_use:
        student_counts = pd.Series(student_use).value_counts()
        student_top = set(student_counts.head(5).index.astype(int).tolist())
    else:
        student_top = set()
    jacc = (len(teacher_top & student_top) / len(teacher_top | student_top)
            if (teacher_top | student_top) else 0.0)

    n_leaves = int(student.get_n_leaves())

    artefact = EdgeModelArtefact(
        feature_names=feature_names,
        feat=feat,
        thr_q=thr_q,
        lc=lc,
        rc=rc,
        val=val,
        teacher_auc=float(teacher_auc),
        student_auc=float(student_auc),
        auc_gap_pp=float((teacher_auc - student_auc) * 100),
        feature_jaccard_top5=float(jacc),
        n_leaves=n_leaves,
        size_bytes=0,
    )
    artefact.size_bytes = len(json.dumps(artefact.to_dict()).encode("utf-8"))
    return artefact


# Pure-Python evaluator used by the dashboard's offline-mode demo.
EVALUATOR_PY = '''
def predict_edge(features, model):
    """30-line edge evaluator. Returns P(placed=1) for one feature vector.

    `features` is a dict {feature_name: float}.
    `model`    is the JSON dict served by /api/edge_model.
    """
    feat = model["feat"]; thr_q = model["thr_q"]
    lc = model["lc"];     rc = model["rc"]
    val = model["val"];   names = model["feature_names"]
    i = 0
    while feat[i] >= 0:
        x = float(features.get(names[feat[i]], 0.0))
        i = lc[i] if x <= (thr_q[i] / 32767.0) else rc[i]
    return val[i]
'''
