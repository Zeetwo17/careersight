"""DeepHitSingle survival model on the {3, 6, 12} month placement grid.

Replaces the linear-interpolation hack in `predict._predict_calibrated` /
`predict_profile` (the fake "Neural ODE survival curve") with a discrete-time
deep survival model fit on the same {3, 6, 12} month anchors. Wiegrebe et al.
(2024, arXiv:2305.14961) show discrete-time deep models match continuous-time
Neural ODEs at this scale with 10–50× less compute, so we deliberately reject
torchdiffeq and use DeepHit (Lee et al., AAAI 2018).

Trained on (duration, event) pairs derived from the synthetic time-to-placement.
We bucket time into [0, 3), [3, 6), [6, 12), and [12, +inf) and fit a single-
risk DeepHit. At inference we return the discrete-time S(t) at the three grid
points; for the dashboard's smoother UX we still interpolate between them, but
this is now interpolation between honestly-trained anchors rather than between
three independent classifier outputs.

Citations:
  Lee et al. — DeepHit — AAAI 2018.
  Kvamme et al. — pycox — arXiv:1907.00825 (the package implementing DeepHit).
  Wiegrebe et al. — Deep Learning for Survival Analysis: A Review —
    arXiv:2305.14961 (2024).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
# Pycox produces a lot of "PendingDeprecation" noise on some torch builds.
import logging
logging.getLogger("torch").setLevel(logging.ERROR)


SURVIVAL_CUTS = np.array([3.0, 6.0, 12.0], dtype="float32")


@dataclass
class SurvivalArtefact:
    """Packed for joblib persistence; PyTorch state lives in `state_dict`."""
    cuts: list[float]
    state_dict: dict          # cpu state-dict of the MLP
    in_features: int
    hidden_sizes: list[int]
    out_features: int
    duration_index: list[float]
    c_index: float
    note: str

    def to_dict(self) -> dict:
        return {
            "cuts": list(self.cuts),
            "duration_index": list(self.duration_index),
            "c_index": float(self.c_index),
            "in_features": int(self.in_features),
            "hidden_sizes": list(self.hidden_sizes),
            "out_features": int(self.out_features),
            "note": self.note,
        }


def _build_net(in_features: int, hidden_sizes: list[int], out_features: int) -> torch.nn.Module:
    layers: list[torch.nn.Module] = []
    prev = in_features
    for h in hidden_sizes:
        layers.append(torch.nn.Linear(prev, h))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.BatchNorm1d(h))
        layers.append(torch.nn.Dropout(0.1))
        prev = h
    layers.append(torch.nn.Linear(prev, out_features))
    return torch.nn.Sequential(*layers)


def fit_deephit(X: np.ndarray, durations: np.ndarray, events: np.ndarray,
                cuts: np.ndarray = SURVIVAL_CUTS,
                hidden_sizes: list[int] | None = None,
                epochs: int = 50, batch_size: int = 256,
                lr: float = 1e-3, seed: int = 21) -> tuple:
    """Fit DeepHitSingle on the discrete-time grid and return (model, artefact)."""
    from pycox.models import DeepHitSingle
    import torchtuples as tt
    from pycox.evaluation import EvalSurv

    torch.manual_seed(seed)
    np.random.seed(seed)

    hidden_sizes = hidden_sizes or [64, 64]

    labtrans = DeepHitSingle.label_transform(cuts=cuts)
    durations_f = durations.astype("float32")
    events_int = events.astype("int")
    y_train = labtrans.fit_transform(durations_f, events_int)

    # Train/val split for early stopping
    n = len(X)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    val_n = int(0.15 * n)
    val_idx = idx[:val_n]
    tr_idx = idx[val_n:]

    Xtr = X[tr_idx].astype("float32")
    Xva = X[val_idx].astype("float32")
    ytr = (y_train[0][tr_idx], y_train[1][tr_idx])
    yva = (y_train[0][val_idx], y_train[1][val_idx])

    in_features = X.shape[1]
    out_features = labtrans.out_features
    net = _build_net(in_features, hidden_sizes, out_features)

    model = DeepHitSingle(net, tt.optim.Adam(lr=lr), alpha=0.2, sigma=0.1,
                          duration_index=labtrans.cuts)
    callbacks = [tt.callbacks.EarlyStopping(patience=8)]
    model.fit(Xtr, ytr, batch_size=batch_size, epochs=epochs,
              callbacks=callbacks, val_data=(Xva, yva), verbose=False)

    # Concordance
    surv = model.predict_surv_df(Xva)
    ev = EvalSurv(surv, durations_f[val_idx], events_int[val_idx], censor_surv="km")
    try:
        c_index = float(ev.concordance_td("antolini"))
    except Exception:
        c_index = float("nan")

    artefact = SurvivalArtefact(
        cuts=list(cuts.astype(float)),
        state_dict={k: v.detach().cpu() for k, v in net.state_dict().items()},
        in_features=in_features,
        hidden_sizes=hidden_sizes,
        out_features=out_features,
        duration_index=list(labtrans.cuts.astype(float)),
        c_index=c_index,
        note=f"DeepHitSingle on cuts={list(cuts.astype(float))}; n_train={len(tr_idx)}",
    )
    return model, artefact


def reload_model(artefact_dict: dict, state_dict: dict):
    """Rebuild a DeepHit model from a persisted artefact."""
    from pycox.models import DeepHitSingle
    import torchtuples as tt
    net = _build_net(artefact_dict["in_features"], artefact_dict["hidden_sizes"],
                     artefact_dict["out_features"])
    net.load_state_dict(state_dict)
    net.eval()
    duration_index = np.asarray(artefact_dict["duration_index"], dtype="float32")
    return DeepHitSingle(net, tt.optim.Adam(lr=1e-3), alpha=0.2, sigma=0.1,
                        duration_index=duration_index)


def survival_curve(model, x_row: np.ndarray) -> list[dict]:
    """Return [(month, p_unplaced), ...] interpolated to the dashboard's
    13-point monthly grid for visual continuity."""
    import torchtuples as tt
    if x_row.ndim == 1:
        x_row = x_row.reshape(1, -1)
    surv = model.predict_surv_df(x_row.astype("float32")).iloc[:, 0]
    times = surv.index.values.astype(float)
    vals = surv.values.astype(float)

    # Dashboard expects months 0..12. Add anchors (0, 1.0) and linearly
    # interpolate between DeepHit's discrete cuts. Beyond 12 we hold the
    # last value (right-censored).
    anchor_t = np.concatenate([[0.0], times, [13.0]])
    anchor_v = np.concatenate([[1.0], vals, [vals[-1] if len(vals) else 0.5]])
    anchor_v = np.minimum.accumulate(anchor_v)  # enforce monotonicity

    out = []
    for m in range(0, 13):
        v = float(np.interp(m, anchor_t, anchor_v))
        out.append({"month": m, "p_unplaced": round(v, 4)})
    return out
