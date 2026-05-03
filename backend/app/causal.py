"""Automated causal discovery via the PC algorithm + Markov-blanket selection.

The deck (innovation #6) claims "Auto DAG · Markov Blanket". Previously this
was vapor — no DAG was ever computed. This module:

  1. Subsamples the training data and runs causallearn's PC algorithm
     (Spirtes & Glymour, 1991) to recover a CPDAG over a numeric feature
     subset + the placement_6m target.
  2. Extracts the Markov blanket of the target (parents + children +
     spouses-of-children) — the minimal sufficient feature set under
     faithfulness.
  3. Refits a small "MB-only" LightGBM on just those features and compares
     AUC against the 57-feature baseline. A small AUC gap with far fewer
     features is the parsimony story.
  4. Persists the CPDAG adjacency + MB list + parsimony delta on the bundle.

We sub-select the 24 most-informative numeric features (by full-model gain)
to keep runtime under ~10 s. With more time we could push to all 45.

Citations:
  Spirtes, Glymour, Scheines — "Causation, Prediction, and Search" (2000).
  Yu et al., Causal Learner — arXiv:2103.06544.
  Huber — "Causal Discovery in Economics/Education" — arXiv:2407.08602 (2024).
  causal-learn docs — https://causal-learn.readthedocs.io
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from .features import expand_dataframe
from .schema import ALL_FEATURE_COLUMNS

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class CausalResult:
    target_idx: int
    feature_subset: list[str]              # the names corresponding to graph nodes
    edges: list[tuple[int, int, str]]      # (source_idx, target_idx, kind)
    markov_blanket: list[str]              # feature names in the MB of the target
    n_subsample: int
    full_auc: float | None
    mb_auc: float | None
    mb_size: int
    full_size: int
    note: str

    def to_dict(self) -> dict:
        return {
            "target_idx": int(self.target_idx),
            "feature_subset": list(self.feature_subset),
            "edges": [[int(s), int(t), str(k)] for (s, t, k) in self.edges],
            "markov_blanket": list(self.markov_blanket),
            "n_subsample": int(self.n_subsample),
            "full_auc": (None if self.full_auc is None else float(self.full_auc)),
            "mb_auc": (None if self.mb_auc is None else float(self.mb_auc)),
            "mb_size": int(self.mb_size),
            "full_size": int(self.full_size),
            "note": self.note,
        }


def _select_top_features(bundle, top_k: int = 24) -> list[str]:
    """Pick the top-k most-informative features from the trained LightGBM
    (gain importance on the 6-month head). Used to keep PC's adjacency
    search tractable on a hackathon timeline."""
    clf = bundle["classifiers"]["placed_6m"]
    booster = clf.booster_
    importance = booster.feature_importance(importance_type="gain")
    feat_names = bundle["feature_names"]
    pairs = sorted(zip(feat_names, importance), key=lambda x: -x[1])
    # Keep numeric continuous-ish features for fisherz independence test
    continuous = [
        f for f, _ in pairs
        if not f.startswith("course_") and f not in {"is_metro", "tier"}
    ]
    return continuous[:top_k]


def discover(bundle, df: pd.DataFrame, n_subsample: int = 2500, alpha: float = 0.05,
             top_k: int = 24) -> CausalResult:
    """Run PC on the chosen feature subset + target and extract the MB."""
    try:
        from causallearn.search.ConstraintBased.PC import pc
    except ImportError as exc:
        return CausalResult(
            target_idx=-1, feature_subset=[], edges=[], markov_blanket=[],
            n_subsample=0, full_auc=None, mb_auc=None,
            mb_size=0, full_size=0,
            note=f"causallearn import failed: {exc}",
        )

    chosen = _select_top_features(bundle, top_k=top_k)
    n = len(df)
    sample = df.sample(n=min(n_subsample, n), random_state=11).reset_index(drop=True)
    X_full = expand_dataframe(sample).astype(np.float32)
    X_sub = X_full[chosen].values
    y_6m = sample["placed_6m"].astype(int).values
    data = np.column_stack([X_sub, y_6m]).astype(float)
    node_names = list(chosen) + ["placement_6m"]
    target_idx = len(chosen)  # last column

    try:
        cg = pc(data, alpha=alpha, indep_test="fisherz", stable=True,
                uc_rule=0, uc_priority=2, mvpc=False, verbose=False,
                node_names=node_names)
    except Exception as exc:
        return CausalResult(
            target_idx=target_idx, feature_subset=chosen, edges=[],
            markov_blanket=[], n_subsample=len(sample),
            full_auc=None, mb_auc=None,
            mb_size=0, full_size=len(ALL_FEATURE_COLUMNS),
            note=f"PC algorithm failed: {exc}",
        )

    G = cg.G.graph  # adjacency matrix; values: -1 = tail, 1 = head, 0 = no edge

    # Extract edges in a UI-friendly form. CPDAG conventions:
    #   G[i][j] == -1 and G[j][i] == 1  -> directed i -> j
    #   G[i][j] == -1 and G[j][i] == -1 -> undirected i — j
    edges: list[tuple[int, int, str]] = []
    n_nodes = G.shape[0]
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            a, b = G[i, j], G[j, i]
            if a == -1 and b == 1:
                edges.append((i, j, "->"))
            elif a == 1 and b == -1:
                edges.append((j, i, "->"))
            elif a == -1 and b == -1:
                edges.append((i, j, "--"))
            elif a == 1 and b == 1:
                edges.append((i, j, "<->"))

    # Markov blanket: parents ∪ children ∪ spouses-of-children
    parents = {i for i in range(n_nodes) if G[i, target_idx] == -1 and G[target_idx, i] == 1}
    children = {j for j in range(n_nodes) if G[target_idx, j] == -1 and G[j, target_idx] == 1}
    spouses = set()
    for ch in children:
        for k in range(n_nodes):
            if k == target_idx:
                continue
            if G[k, ch] == -1 and G[ch, k] == 1:
                spouses.add(k)
    # Adjacent (undirected) neighbours also belong to MB under the standard
    # PC-derived definition — under faithfulness these are still relevant.
    adj = {k for k in range(n_nodes)
           if k != target_idx
           and ((G[k, target_idx] == -1 and G[target_idx, k] == -1)
                or (G[k, target_idx] == 1 and G[target_idx, k] == 1))}
    mb_idx = sorted((parents | children | spouses | adj) - {target_idx})
    mb_names = [node_names[i] for i in mb_idx]

    # Parsimony test: refit LGBM on just the MB features, compare AUC.
    full_auc = mb_auc = None
    try:
        import lightgbm as lgb
        full_X = X_full.values
        Xf_tr, Xf_te, y_tr, y_te = train_test_split(full_X, y_6m, test_size=0.2,
                                                    random_state=42, stratify=y_6m)
        # Use the existing trained classifier for full-feature AUC (test fold)
        full_auc = float(roc_auc_score(y_te, bundle["classifiers"]["placed_6m"].predict_proba(Xf_te)[:, 1]))

        if mb_names:
            X_mb = X_full[mb_names].values
            Xm_tr, Xm_te, _, _ = train_test_split(X_mb, y_6m, test_size=0.2,
                                                   random_state=42, stratify=y_6m)
            small = lgb.LGBMClassifier(
                objective="binary", num_leaves=31, n_estimators=200,
                learning_rate=0.05, min_data_in_leaf=80, verbose=-1,
            )
            small.fit(Xm_tr, y_tr)
            mb_auc = float(roc_auc_score(y_te, small.predict_proba(Xm_te)[:, 1]))
    except Exception as exc:
        return CausalResult(
            target_idx=target_idx, feature_subset=chosen, edges=edges,
            markov_blanket=mb_names, n_subsample=len(sample),
            full_auc=full_auc, mb_auc=mb_auc,
            mb_size=len(mb_names), full_size=len(ALL_FEATURE_COLUMNS),
            note=f"MB refit failed: {exc}",
        )

    parsimony_str = ""
    if full_auc is not None and mb_auc is not None:
        parsimony_str = (f"MB-only model retains "
                         f"{(mb_auc/full_auc * 100):.1f}% of full AUC "
                         f"with {len(mb_names)}/{len(ALL_FEATURE_COLUMNS)} features.")

    return CausalResult(
        target_idx=target_idx,
        feature_subset=chosen,
        edges=edges,
        markov_blanket=mb_names,
        n_subsample=len(sample),
        full_auc=full_auc,
        mb_auc=mb_auc,
        mb_size=len(mb_names),
        full_size=len(ALL_FEATURE_COLUMNS),
        note=parsimony_str or "PC discovery completed.",
    )
