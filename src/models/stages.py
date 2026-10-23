"""Training and prediction for the three stability stages (thermal regression, solvent classification, water random forest)
using the released architectures and hyperparameters, with the early-stopping restore FIXED (copy.deepcopy of the best state)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.models.stability_nets import SolventNet, ThermalNet, predict, train_released_recipe

# released hyperparameters (model/thermal_hyperparams.json, model/solvent_hyperparams.json, saved random forest)
THERMAL = dict(hidden=[2048, 1024, 512, 256, 128, 64, 32], lr=1e-4, batch_size=512, patience=100, epochs=1000)
SOLVENT = dict(hidden=[512, 256, 128, 64, 32, 16], lr=1e-3, batch_size=256, patience=200, epochs=1000, dropout=0.2)
WATER_RF = dict(n_estimators=200, max_depth=None, min_samples_split=2, min_samples_leaf=1, bootstrap=False, max_features="sqrt")


@dataclass
class Fitted:
    kind: str
    model: object
    scaler: Optional[StandardScaler]
    info: dict


def fit_thermal(Xtr, ytr, Xva, yva, seed: int, device: str = "cuda") -> Fitted:
    sc = StandardScaler().fit(Xtr_all := np.vstack([Xtr, Xva]))
    net = ThermalNet(Xtr.shape[1], THERMAL["hidden"], 1)
    h = train_released_recipe(net, sc.transform(Xtr), ytr, sc.transform(Xva), yva, nn.MSELoss(), THERMAL["lr"], THERMAL["batch_size"], THERMAL["epochs"], THERMAL["patience"], seed, device, faithful_best_state=False)
    return Fitted("thermal", net, sc, {k: h[k] for k in ("epochs_run", "best_epoch", "best_val", "final_val", "restore_changed_nothing")})


def fit_solvent(Xtr, ytr, Xva, yva, seed: int, device: str = "cuda") -> Fitted:
    sc = StandardScaler().fit(np.vstack([Xtr, Xva]))
    net = SolventNet(Xtr.shape[1], SOLVENT["hidden"], 1, SOLVENT["dropout"])
    h = train_released_recipe(net, sc.transform(Xtr), ytr, sc.transform(Xva), yva, nn.BCEWithLogitsLoss(), SOLVENT["lr"], SOLVENT["batch_size"], SOLVENT["epochs"], SOLVENT["patience"], seed, device, faithful_best_state=False)
    return Fitted("solvent", net, sc, {k: h[k] for k in ("epochs_run", "best_epoch", "best_val", "final_val", "restore_changed_nothing")})


def fit_water(Xtr, ytr, seed: int, n_jobs: int = 8) -> Fitted:
    rf = RandomForestClassifier(random_state=seed, n_jobs=n_jobs, **WATER_RF).fit(Xtr, ytr)
    return Fitted("water", rf, None, {})


def scores(f: Fitted, X: np.ndarray, device: str = "cuda") -> dict:
    """thermal: predicted degC (`score`); solvent: logit and probability; water: class probabilities, P(stable)=P(2)+P(3)."""
    if f.kind == "thermal":
        t = predict(f.model, f.scaler.transform(X), device)
        return {"pred": t, "score": t}
    if f.kind == "solvent":
        z = predict(f.model, f.scaler.transform(X), device)
        p = 1 / (1 + np.exp(-z))
        return {"logit": z, "p_stable": p, "score": p}
    pr = f.model.predict_proba(X)
    full = np.zeros((len(X), 4)); full[:, f.model.classes_.astype(int)] = pr
    out = {f"p{c}": full[:, c] for c in range(4)}
    out["p_stable"] = full[:, 2] + full[:, 3]; out["score"] = out["p_stable"]
    return out
