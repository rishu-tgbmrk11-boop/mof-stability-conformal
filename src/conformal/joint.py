"""Joint cascade score and cross-conformal (CV+ / jackknife+ style) recall control on the all-three-stable MOFs.

Joint score: c(x) = min over the three stages of the rank-normalised stage score, u_k(s) = ECDF of stage-k scores on that stage's calibration data (mid-rank, in [0, 1]).
Calibration positives: the n MOFs that are stable at ALL three stages. Positive i has an out-of-fold score c_i from the model j(i) that was trained without i's group.
For a new MOF x, every fold model j gives a score c^(j)(x). CV+ p-value (Barber, Candes, Ramdas, Tibshirani 2021, adapted to a one-sided 'positive-like' test):
    p(x) = (1 + #{i : c_i <= c^(j(i))(x)}) / (n + 1);   keep x iff p(x) > alpha.
Ties count in favour of keeping (conservative for recall).
Guarantee (exchangeability of the positives and a model that treats them symmetrically; leave-one-positive-out = jackknife+): P(x is dropped | x all-stable) <= 2 alpha,
i.e. recall >= 1 - 2 alpha; for K-fold CV+ with equal folds there is an extra slack min{2(1-1/K)/(n/K+1), (1-K/n)/(K+1)}. Non-trivial filtering needs alpha >= 1/(n+1)."""
from __future__ import annotations

import numpy as np


def rank_normalise(ref: np.ndarray):
    """Return u(s) = mid-rank of s among `ref` scores, in [0, 1] (monotone non-decreasing in s)."""
    ref = np.sort(np.asarray(ref, float)); n = len(ref)

    def u(s):
        s = np.asarray(s, float)
        return (np.searchsorted(ref, s, side="left") + np.searchsorted(ref, s, side="right")) / (2.0 * n)
    return u


def cvplus_pvalue(cal_scores: np.ndarray, cal_fold: np.ndarray, test_scores: np.ndarray) -> np.ndarray:
    """cal_scores[i]: out-of-fold score of calibration positive i; cal_fold[i]: index of the model that scored it;
    test_scores[x, j]: score of test point x from model j. Returns p(x) = (1 + #{i: cal_scores[i] <= test_scores[x, cal_fold[i]]}) / (n + 1)."""
    cal_scores, cal_fold, test_scores = np.asarray(cal_scores, float), np.asarray(cal_fold, int), np.asarray(test_scores, float)
    n = len(cal_scores)
    cnt = (cal_scores[None, :] <= test_scores[:, cal_fold]).sum(1)
    return (1.0 + cnt) / (n + 1.0)


def keep_rule(p: np.ndarray, alpha: float) -> np.ndarray:
    return np.asarray(p) > alpha


def min_alpha(n_pos: int) -> float:
    """Smallest alpha with a non-trivial rule (below it every MOF is kept)."""
    return 1.0 / (n_pos + 1)


def guaranteed_recall(alpha: float) -> float:
    """Jackknife+ / CV+ headline guarantee 1 - 2 alpha (without the CV+ slack term)."""
    return 1.0 - 2.0 * alpha


def cvplus_slack(n: int, k: int) -> float:
    """Extra slack of K-fold CV+ with equal folds (Barber et al. 2021, Thm 4): min{2 (1 - 1/K) / (n/K + 1), (1 - K/n) / (K + 1)}; 0 for K = n."""
    return float(max(0.0, min(2 * (1 - 1 / k) / (n / k + 1), (1 - k / n) / (k + 1))))
