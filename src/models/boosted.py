"""Tree-model candidates for the three stability stages (Phase 3): XGBoost, ExtraTrees, RandomForest, with optional class weighting and
feature selection fitted INSIDE the training data handed to `fit_predict`. Output keys match `src.models.stages.scores`
(thermal: pred/score; solvent: p_stable/score; water: p0..p3/p_stable/score), so conformal code can use either."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor

N_CLASSES = {"thermal": 0, "solvent": 2, "water": 4}


def config_grid(task: str) -> List[dict]:
    """Small fixed grid; selection is by grouped CV on the training groups only (scripts/p3_01_select.py)."""
    cls = task != "thermal"
    out = []
    for fs in (None, 50):
        for depth in (3, 5, 7):
            for mcw in (1, 5):
                for v in ((False, True) if cls else ("reg:squarederror", "reg:absoluteerror")):
                    c = dict(kind="xgb", depth=depth, mcw=mcw, fs=fs)
                    c.update(balanced=v) if cls else c.update(obj=v)
                    out.append(c)
        for kind in ("et", "rf"):
            for leaf in (1, 3):
                for bal in ((False, True) if cls else (False,)):
                    out.append(dict(kind=kind, leaf=leaf, fs=fs, balanced=bal))
    return out


def cfg_name(c: dict) -> str:
    return "_".join(f"{k}={v}" for k, v in c.items())


def _xgb(task, c, seed, n_jobs):
    p = dict(n_estimators=400, learning_rate=0.05, max_depth=c["depth"], min_child_weight=c["mcw"], subsample=0.8, colsample_bytree=0.5,
             tree_method="hist", n_jobs=n_jobs, random_state=seed)
    if task == "thermal":
        return XGBRegressor(objective=c["obj"], **p)
    return XGBClassifier(**p)


def _make(task, c, seed, n_jobs):
    if c["kind"] == "xgb":
        return _xgb(task, c, seed, n_jobs)
    p = dict(n_estimators=500, min_samples_leaf=c["leaf"], max_features="sqrt", n_jobs=n_jobs, random_state=seed)
    cls = {"et": (ExtraTreesRegressor, ExtraTreesClassifier), "rf": (RandomForestRegressor, RandomForestClassifier)}[c["kind"]]
    return cls[0](**p) if task == "thermal" else cls[1](**p)


def select_features(task: str, Xtr, ytr, k: int, seed: int, n_jobs: int = 8) -> np.ndarray:
    """Indices of the k features with highest XGBoost gain importance, fitted on the given training rows only."""
    c = dict(kind="xgb", depth=4, mcw=1, obj="reg:squarederror")
    m = _xgb(task, c, seed, n_jobs).set_params(n_estimators=150)
    m.fit(Xtr, _enc(task, ytr))
    return np.sort(np.argsort(-m.feature_importances_)[:k])


def _enc(task, y):
    return np.asarray(y, float) if task == "thermal" else np.asarray(y, int)


def fit_predict(task: str, c: dict, Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, seed: int = 0, n_jobs: int = 8) -> Dict[str, np.ndarray]:
    cols = select_features(task, Xtr, ytr, c["fs"], seed, n_jobs) if c.get("fs") else np.arange(Xtr.shape[1])
    m = _make(task, c, seed, n_jobs)
    y = _enc(task, ytr)
    sw = compute_sample_weight("balanced", y) if c.get("balanced") else None
    classes = np.arange(N_CLASSES[task]) if task != "thermal" else None
    if task != "thermal" and not np.array_equal(np.unique(y), classes):      # a class missing from the training rows: remap to present classes
        present = np.unique(y); remap = {int(v): i for i, v in enumerate(present)}
        m.fit(Xtr[:, cols], np.array([remap[int(v)] for v in y]), sample_weight=sw)
        pr = np.zeros((len(Xte), N_CLASSES[task])); pr[:, present] = m.predict_proba(Xte[:, cols])
    elif task == "thermal":
        m.fit(Xtr[:, cols], y); t = m.predict(Xte[:, cols]).astype(float)
        return {"pred": t, "score": t}
    else:
        m.fit(Xtr[:, cols], y, sample_weight=sw); pr = m.predict_proba(Xte[:, cols])
    if task == "solvent":
        return {"p_stable": pr[:, 1], "score": pr[:, 1]}
    out = {f"p{i}": pr[:, i] for i in range(4)}
    out["p_stable"] = pr[:, 2] + pr[:, 3]; out["score"] = out["p_stable"]
    return out


def average(outs: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Unweighted mean of members' outputs (probabilities / predicted degC)."""
    return {k: np.mean([o[k] for o in outs], axis=0) for k in outs[0]}
