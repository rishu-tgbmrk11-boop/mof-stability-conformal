import numpy as np

from src.conformal.recall import conformal_k, conformal_threshold, conformal_threshold_pac, recall_curve, stage_outcome, validity_check


def test_k_and_threshold_formula():
    assert conformal_k(19, 0.05) == 1 and conformal_k(99, 0.1) == 10 and conformal_k(10, 0.05) == 0
    pos = np.arange(1, 100)           # 99 calibration scores 1..99
    assert conformal_threshold(pos, 0.1) == 10.0                # k = floor(0.1 * 100) = 10 -> the 10th smallest
    assert conformal_threshold(pos[:10], 0.05) == float("-inf")  # alpha (n+1) < 1: keep everything


def test_recall_guarantee_holds_on_average_for_exchangeable_scores():
    rng = np.random.RandomState(0)
    for alpha in (0.05, 0.1, 0.3):
        recs = []
        for _ in range(400):
            cal = rng.randn(60) + 1.0; te = rng.randn(2000) + 1.0     # stable scores, exchangeable
            t = conformal_threshold(cal, alpha)
            recs.append((te >= t).mean())
        assert np.mean(recs) >= 1 - alpha - 0.01, (alpha, np.mean(recs))


def test_curve_columns_and_monotone_kept_fraction():
    rng = np.random.RandomState(1)
    s = np.r_[rng.randn(300) + 2, rng.randn(300)]; y = np.r_[np.ones(300), np.zeros(300)]
    tc = recall_curve(s[::2], y[::2], s[1::2], y[1::2], [0.05, 0.1, 0.2, 0.4])
    assert list(tc.alpha) == [0.05, 0.1, 0.2, 0.4]
    assert np.all(np.diff(tc.kept_fraction) <= 1e-12)          # larger alpha -> stricter threshold -> fewer MOFs kept
    assert np.all(tc.recall.values <= 1.0 + 1e-12)


def test_validity_check_runs_on_groups():
    rng = np.random.RandomState(2)
    g = np.repeat(np.arange(200), 3); y = (rng.rand(600) < 0.5).astype(int); s = y * 1.5 + rng.randn(600)
    v = validity_check(s, y, g, 0.1, n_rep=60, seed=0)
    assert v["mean_recall"] >= 0.85 and v["n_rep"] == 60


def test_stage_outcome_counts():
    o = stage_outcome(np.array([0.1, 0.5, 0.9, 0.95]), np.array([0, 1, 1, 1]), 0.5)
    assert o["recall"] == 1.0 and o["kept_fraction"] == 0.75 and abs(o["precision"] - 1.0) < 1e-12


def test_pac_threshold_is_more_conservative_and_gives_high_probability_guarantee():
    rng = np.random.RandomState(3)
    n, alpha, delta = 100, 0.1, 0.1
    ok, plain_ok = 0, 0
    for _ in range(600):
        cal = rng.rand(n)                                  # uniform scores: the population miss probability of "keep >= t" is exactly t
        t_pac = conformal_threshold_pac(cal, alpha, delta); t = conformal_threshold(cal, alpha)
        assert t_pac <= t                                  # lower threshold = keeps more = safer
        ok += (t_pac <= alpha); plain_ok += (t <= alpha)
    assert ok / 600 >= 1 - delta - 0.04                    # recall >= 1-alpha in at least ~90 % of calibration draws
    assert plain_ok / 600 < 0.7                            # plain split conformal only guarantees it on average (about half the draws)
