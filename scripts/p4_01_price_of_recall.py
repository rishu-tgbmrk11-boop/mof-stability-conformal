"""Phase 4, step 1: 'price of recall' = fraction of all MOFs that must be kept to reach a target recall of truly stable MOFs.
(a) REAL stages (group-aware pooled calibration+test MOFs, Phase 2 and Phase 3 models): empirical kept fraction at recall 0.80/0.90/0.95 and AUC, with bootstrap over GROUPS.
(b) SYNTHETIC scores with controlled AUC 0.55..0.95: binormal model (negatives N(0,1), positives N(d,1), AUC = Phi(d/sqrt 2)) on the stage's real labels and
    real group ids (scores drawn independently of group, so no within-group dependence), with group-bootstrap intervals, and the analytic infinite-sample curve.
Empirical rule: threshold t = the floor((1-r) n_pos)-th smallest positive score (0-based), kept = mean(score >= t). Estimated without calibration/test distinction, i.e. the price
if the threshold were known exactly; the conformal single-split numbers are in notes/phase2.md and notes/phase3.md.
python -I scripts/p4_01_price_of_recall.py  -> data/processed/phase4/price_of_recall.json"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

OUT = "data/processed/phase4/"
RECALLS, NBOOT = [0.80, 0.90, 0.95], 1000
AUC_GRID = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
TASKS = ("thermal", "solvent", "water")


def kept_at(score, stable, r):
    pos = np.sort(score[stable == 1])
    if len(pos) == 0:
        return float("nan")
    t = pos[int(np.floor((1 - r) * len(pos) + 1e-9))]
    return float((score >= t).mean())


def boot(score, stable, gid, fn, seed):
    uniq, inv = np.unique(gid, return_inverse=True)
    members = [np.where(inv == k)[0] for k in range(len(uniq))]
    rng = np.random.RandomState(seed); vals = []
    for _ in range(NBOOT):
        idx = np.concatenate([members[k] for k in rng.randint(0, len(uniq), len(uniq))])
        v = fn(score[idx], stable[idx])
        if np.isfinite(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [float(fn(score, stable)), float(lo), float(hi)]


def summarise(score, stable, gid, seed):
    d = {"n": int(len(score)), "n_pos": int(stable.sum()), "n_groups": int(len(np.unique(gid))), "prevalence": float(stable.mean()),
         "auc": boot(score, stable, gid, lambda s, y: roc_auc_score(y, s) if 0 < y.sum() < len(y) else float("nan"), seed)}
    for r in RECALLS:
        d[f"kept_{r}"] = boot(score, stable, gid, lambda s, y, r=r: kept_at(s, y, r), seed + 1)
    return d


res = {"recalls": RECALLS, "auc_grid": AUC_GRID, "n_boot": NBOOT, "real": {}, "synthetic": {}}
for t in TASKS:
    base = {}
    for phase, d in (("phase2", "data/processed/phase2/"), ("phase3", "data/processed/phase3/")):
        df = pd.read_parquet(d + f"pred_group_{t}_s0.parquet"); df = df[df.split.isin(["cal", "test"])].reset_index(drop=True)
        res["real"][f"{phase}_{t}"] = summarise(df.score.values.astype(float), df.stable.values.astype(int), df.group_id.values, 10)
        base[phase] = df
    ref = base["phase3"]                                         # same MOFs / labels / groups in both phases
    assert (ref.name.values == base["phase2"].name.values).all() and (ref.stable.values == base["phase2"].stable.values).all()
    y, gid = ref.stable.values.astype(int), ref.group_id.values
    pi = float(y.mean()); syn = {"prevalence": pi, "n": int(len(y)), "n_groups": int(len(np.unique(gid))), "points": [], "analytic": []}
    rng = np.random.RandomState(2026)
    for auc in AUC_GRID:
        dd = np.sqrt(2) * norm.ppf(auc)
        s = rng.randn(len(y)) + dd * y
        row = {"auc_target": auc, "auc_sample": float(roc_auc_score(y, s))}
        for r in RECALLS:
            row[f"kept_{r}"] = boot(s, y, gid, lambda a, b, r=r: kept_at(a, b, r), 20)
        syn["points"].append(row)
    for auc in np.round(np.arange(0.50, 0.9951, 0.005), 3):
        dd = np.sqrt(2) * norm.ppf(auc)
        syn["analytic"].append({"auc": float(auc), **{f"kept_{r}": float(pi * r + (1 - pi) * (1 - norm.cdf(dd + norm.ppf(1 - r)))) for r in RECALLS}})
    res["synthetic"][t] = syn
    print(f"[{t}] n {syn['n']}, groups {syn['n_groups']}, prevalence {pi:.3f}")
    for k in ("phase2", "phase3"):
        r_ = res["real"][f"{k}_{t}"]; print(f"   real {k}: AUC {r_['auc'][0]:.3f} [{r_['auc'][1]:.3f}, {r_['auc'][2]:.3f}] | kept@0.80/0.90/0.95: " + " / ".join(f"{r_[f'kept_{r}'][0]:.3f}" for r in RECALLS))
    for p in syn["points"]:
        print(f"   synthetic AUC {p['auc_target']:.2f} (sample {p['auc_sample']:.3f}): kept " + " / ".join(f"{p[f'kept_{r}'][0]:.3f} [{p[f'kept_{r}'][1]:.3f},{p[f'kept_{r}'][2]:.3f}]" for r in RECALLS))
json.dump(res, open(OUT + "price_of_recall.json", "w"), indent=1)
print("DONE")
