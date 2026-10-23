"""Metrics with bootstrap intervals over groups."""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, mean_absolute_error, r2_score, roc_auc_score


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error of a binary probability (equal-width bins)."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    idx = np.clip(np.digitize(p, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).sum() for b in range(bins) if (idx == b).any()) / len(p))


def reliability_bins(p: np.ndarray, y: np.ndarray, bins: int = 10):
    p, y = np.asarray(p, float), np.asarray(y, float)
    idx = np.clip(np.digitize(p, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
    return [(int((idx == b).sum()), float(p[idx == b].mean()), float(y[idx == b].mean())) for b in range(bins) if (idx == b).any()]


def safe_auc(y, s) -> float:
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def cluster_bootstrap(metric: Callable[[np.ndarray], float], groups: np.ndarray, n_boot: int = 1000, seed: int = 0, level: float = 0.95) -> Tuple[float, float, float]:
    """Resample whole groups with replacement; `metric(idx)` receives the integer row indices of the resampled rows.
    Returns (point estimate on the original sample, lower, upper) percentile interval."""
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == k)[0] for k in range(len(uniq))]
    rng = np.random.RandomState(seed)
    point = metric(np.arange(len(groups)))
    vals = []
    for _ in range(n_boot):
        pick = rng.randint(0, len(uniq), len(uniq))
        idx = np.concatenate([members[k] for k in pick])
        v = metric(idx)
        if np.isfinite(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(point), float(lo), float(hi)
