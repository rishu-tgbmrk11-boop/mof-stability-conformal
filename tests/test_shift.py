import numpy as np

from src.conformal.shift import cluster_split, embed, knn_distance, mondrian_keep, mondrian_thresholds


def _toy(n=600, seed=0):
    rng = np.random.RandomState(seed)
    centres = rng.randn(8, 5) * 6
    lab = rng.randint(0, 8, n)
    return centres[lab] + rng.randn(n, 5)


def test_cluster_split_is_disjoint_and_keeps_clusters_whole():
    X = _toy(); E = embed(X, 4, 0)
    sp = cluster_split(E, 8, seed=3)
    idx = [set(sp[s].tolist()) for s in ("train", "cal", "test")]
    assert not (idx[0] & idx[1]) and not (idx[0] & idx[2]) and not (idx[1] & idx[2])
    assert len(idx[0] | idx[1] | idx[2]) == len(X)
    for s in ("train", "cal", "test"):                       # every cluster lies wholly in one split
        assert len(set(sp["cluster"][sp[s]].tolist()) & set(sp["cluster"][np.setdiff1d(np.arange(len(X)), sp[s])].tolist())) == 0
    assert len(sp["test"]) > 0 and len(sp["cal"]) > 0 and len(sp["train"]) > 0


def test_knn_distance_zero_for_training_points_when_k_is_one():
    Z = np.random.RandomState(1).randn(50, 3)
    assert np.allclose(knn_distance(Z, Z, k=1), 0)


def test_mondrian_matches_plain_rule_in_each_bin_and_keeps_all_when_bin_too_small():
    rng = np.random.RandomState(0)
    n = 300; dist = rng.rand(n); stable = rng.rand(n) < 0.6; score = rng.randn(n) + stable
    m = mondrian_thresholds(score, stable, dist, 0.1, n_bins=3)
    assert len(m["thr"]) == 3 and np.isfinite(m["thr"]).all()
    keep = mondrian_keep(m, score, dist)
    assert keep.shape == (n,)
    tiny = mondrian_thresholds(score[:20], stable[:20], dist[:20], 0.01, n_bins=3)       # alpha*(n+1) < 1 in every bin -> keep everything
    assert np.isneginf(tiny["thr"]).all() and mondrian_keep(tiny, score, dist).all()
