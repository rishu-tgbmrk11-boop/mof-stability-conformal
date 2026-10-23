"""Split conformal RECALL control for one screening stage.

Rule: keep a MOF if its stage score is >= t. With n calibration scores of TRULY STABLE MOFs, take t = the k-th smallest of them, k = floor(alpha (n + 1)).
If a new stable MOF's score is exchangeable with the calibration stable scores, P(score < t) = k / (n + 1) <= alpha, so
E[recall] >= 1 - alpha (the expectation is over the draw of the calibration set; ties only make the rule more conservative).
If k = 0 no filtering is possible at that alpha (t = -inf: keep everything).

Exchangeability holds at the level of the units that were split: here whole GROUPS were assigned to calibration or test, so the guarantee is
marginal over random assignments of groups, not conditional on one assignment."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


def conformal_k(n_pos: int, alpha: float) -> int:
    return int(np.floor(alpha * (n_pos + 1) + 1e-12))


def conformal_threshold(cal_pos_scores: Sequence[float], alpha: float) -> float:
    """Threshold t for the rule 'keep if score >= t'. Returns -inf when alpha*(n+1) < 1."""
    s = np.sort(np.asarray(cal_pos_scores, float))
    k = conformal_k(len(s), alpha)
    return float("-inf") if k < 1 else float(s[k - 1])


def conformal_threshold_pac(cal_pos_scores: Sequence[float], alpha: float, delta: float) -> float:
    """Training-conditional (PAC) variant: with probability >= 1 - delta over the calibration draw, the realised recall is >= 1 - alpha.
    For exchangeable continuous scores the population miss probability of 'keep if score >= s_(k)' is Beta(k, n-k+1); take the largest k with
    Beta.cdf(alpha; k, n-k+1) >= 1 - delta (k = 0, i.e. keep everything, if none qualifies). More conservative than `conformal_threshold`."""
    from scipy.stats import beta

    s = np.sort(np.asarray(cal_pos_scores, float))
    n = len(s)
    ks = [k for k in range(1, n + 1) if beta.cdf(alpha, k, n - k + 1) >= 1 - delta]
    return float("-inf") if not ks else float(s[max(ks) - 1])


def stage_outcome(scores: np.ndarray, stable: np.ndarray, t: float) -> Dict[str, float]:
    keep = np.asarray(scores, float) >= t
    st = np.asarray(stable).astype(bool)
    n_pos = int(st.sum())
    return dict(threshold=float(t), recall=float((keep & st).sum() / n_pos) if n_pos else float("nan"), kept_fraction=float(keep.mean()),
                precision=float((keep & st).sum() / keep.sum()) if keep.any() else float("nan"), n_pos=n_pos, n_kept=int(keep.sum()))


def recall_curve(cal_scores: np.ndarray, cal_stable: np.ndarray, test_scores: np.ndarray, test_stable: np.ndarray, alphas: Iterable[float]) -> pd.DataFrame:
    pos = np.asarray(cal_scores)[np.asarray(cal_stable).astype(bool)]
    rows = []
    for a in alphas:
        t = conformal_threshold(pos, a)
        rows.append(dict(alpha=a, target_recall=1 - a, n_cal_pos=len(pos), k=conformal_k(len(pos), a), **stage_outcome(test_scores, test_stable, t)))
    return pd.DataFrame(rows)


def validity_check(scores: np.ndarray, stable: np.ndarray, groups: np.ndarray, alpha: float, n_rep: int = 500, cal_frac: float = 0.5, seed: int = 0, threshold_fn=None) -> Dict[str, float]:
    """With the scores FIXED (models already trained), repeatedly split the pooled held-out groups at random into calibration and test, calibrate, and
    measure recall on the test positives. Split conformal predicts mean recall >= 1 - alpha."""
    threshold_fn = threshold_fn or conformal_threshold
    scores, stable, groups = np.asarray(scores, float), np.asarray(stable).astype(bool), np.asarray(groups)
    uniq = np.unique(groups)
    members = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.RandomState(seed)
    rec, kept = [], []
    for _ in range(n_rep):
        perm = rng.permutation(uniq)
        cal_g = set(perm[: int(round(cal_frac * len(perm)))])
        cal_idx = np.concatenate([members[g] for g in perm if g in cal_g]); te_idx = np.concatenate([members[g] for g in perm if g not in cal_g])
        pos = scores[cal_idx][stable[cal_idx]]
        if len(pos) == 0 or stable[te_idx].sum() == 0:
            continue
        o = stage_outcome(scores[te_idx], stable[te_idx], threshold_fn(pos, alpha))
        rec.append(o["recall"]); kept.append(o["kept_fraction"])
    rec = np.array(rec)
    return dict(alpha=alpha, target=1 - alpha, mean_recall=float(rec.mean()), p05=float(np.percentile(rec, 5)), p50=float(np.percentile(rec, 50)), p95=float(np.percentile(rec, 95)),
                frac_reps_at_or_above_target=float((rec >= 1 - alpha).mean()), mean_kept_fraction=float(np.mean(kept)), n_rep=int(len(rec)))
