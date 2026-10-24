import numpy as np

from src.conformal.joint import cvplus_pvalue, cvplus_slack, guaranteed_recall, keep_rule, min_alpha, rank_normalise
from src.conformal.recall import conformal_threshold


def test_rank_normalise_is_monotone_and_in_unit_interval():
    ref = np.random.RandomState(0).randn(200); u = rank_normalise(ref)
    s = np.linspace(-4, 4, 50); v = u(s)
    assert (np.diff(v) >= 0).all() and v.min() >= 0 and v.max() <= 1
    assert abs(u(np.median(ref)) - 0.5) < 0.01


def test_cvplus_reduces_to_split_conformal_when_models_agree():
    rng = np.random.RandomState(1); cal = rng.randn(40); alpha = 0.1
    test = np.sort(rng.randn(300))
    ts = np.repeat(test[:, None], 5, axis=1)                            # every fold model gives the same score
    keep = keep_rule(cvplus_pvalue(cal, rng.randint(0, 5, 40), ts), alpha)
    thr = conformal_threshold(cal, alpha)
    assert (keep == (test >= thr)).all()


def test_trivial_below_min_alpha_and_guarantee_arithmetic():
    cal = np.random.RandomState(2).randn(17)
    p = cvplus_pvalue(cal, np.arange(17), np.full((5, 17), -100.0))     # test scores far below every calibration score
    assert keep_rule(p, 0.5 * min_alpha(17)).all()                      # alpha < 1/(n+1): keep everything
    assert not keep_rule(p, min_alpha(17)).any()                        # alpha = 1/(n+1): a hopeless MOF is dropped
    assert abs(min_alpha(17) - 1 / 18) < 1e-12 and abs(guaranteed_recall(1 / 18) - (1 - 2 / 18)) < 1e-12
    assert cvplus_slack(17, 17) == 0.0 and cvplus_slack(17, 5) > 0


def test_cvplus_marginal_coverage_in_simulation():
    rng = np.random.RandomState(3); n, alpha, miss = 17, 0.1, []
    for _ in range(3000):
        cal = rng.randn(n) + 1; x = rng.randn(1) + 1
        miss.append(not keep_rule(cvplus_pvalue(cal, np.arange(n), np.repeat(x[:, None], n, 1)), alpha)[0])
    assert np.mean(miss) <= 2 * alpha                                    # guaranteed; in the iid no-model case it is about alpha
