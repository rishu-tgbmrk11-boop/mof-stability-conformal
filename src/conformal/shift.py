"""Family-shift validation helpers: split MOFs by feature-space CLUSTERS (calibration and test clusters disjoint), plus a distance-aware (Mondrian) variant of the
recall rule. The Mondrian variant is a heuristic for shift; no coverage proof is claimed under cluster shift (within-bin exchangeability is what it relies on)."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.conformal.recall import conformal_threshold


def embed(X: np.ndarray, n_components: int = 20, seed: int = 0) -> np.ndarray:
    """Unsupervised embedding used only for clustering (no labels): z-scored features -> PCA."""
    Z = StandardScaler().fit_transform(X)
    return PCA(n_components=n_components, random_state=seed).fit_transform(Z)


def cluster_split(E: np.ndarray, n_clusters: int, seed: int, fractions=(0.6, 0.2, 0.2)) -> Dict[str, np.ndarray]:
    """k-means clusters, shuffled; whole clusters go to test first, then calibration, then train until each split reaches its share of MOFs.
    Returns integer index arrays for 'train', 'cal', 'test' and the cluster label of every MOF."""
    lab = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(E)
    rng = np.random.RandomState(seed)
    order = rng.permutation(n_clusters)
    n = len(lab); want = {"test": fractions[2] * n, "cal": fractions[1] * n}
    where = {}; filled = {"test": 0, "cal": 0}; cur = "test"
    for c in order:
        if cur is not None and filled[cur] >= want[cur]:
            cur = "cal" if cur == "test" else None
        s = cur or "train"; where[c] = s
        if s in filled: filled[s] += int((lab == c).sum())
    split = np.array([where[c] for c in lab])
    return {s: np.where(split == s)[0] for s in ("train", "cal", "test")} | {"cluster": lab}


def knn_distance(Z_train: np.ndarray, Z: np.ndarray, k: int = 5) -> np.ndarray:
    """Mean distance of each row of Z to its k nearest rows of Z_train (both in the same standardised space)."""
    return NearestNeighbors(n_neighbors=k).fit(Z_train).kneighbors(Z)[0].mean(1)


def mondrian_thresholds(cal_score: np.ndarray, cal_stable: np.ndarray, cal_dist: np.ndarray, alpha: float, n_bins: int = 3) -> Dict[str, object]:
    """Distance-aware recall rule: bins of the distance-to-training statistic (edges = calibration quantiles); one split-conformal threshold per bin from the
    calibration positives in that bin (-inf = keep everything when a bin has too few positives for this alpha)."""
    edges = np.quantile(cal_dist, np.linspace(0, 1, n_bins + 1)[1:-1])
    b = np.searchsorted(edges, cal_dist, side="right")
    thr = [conformal_threshold(cal_score[(b == j) & cal_stable.astype(bool)], alpha) for j in range(n_bins)]
    return {"edges": edges, "thr": np.array(thr)}


def mondrian_keep(m: Dict[str, object], score: np.ndarray, dist: np.ndarray) -> np.ndarray:
    b = np.searchsorted(m["edges"], dist, side="right")
    return score >= m["thr"][b]
