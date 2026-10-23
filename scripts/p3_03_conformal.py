"""Phase 3, step 3: per-stage split conformal recall control (plain and PAC) for the Phase 3 models, group-aware split only.
Same code as scripts/p2_03_conformal.py with predictions from data/processed/phase3/. python -I scripts/p3_03_conformal.py"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd

from src.conformal.recall import conformal_threshold, conformal_threshold_pac, recall_curve, stage_outcome, validity_check
from src.eval.metrics import cluster_bootstrap
from src.features import stability_data as sd

OUT = "data/processed/phase3/"
ALPHAS = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
VALID_ALPHAS = [0.05, 0.10, 0.20]
res = {}
paper = {"thermal": lambda d: d["pred"].values >= 350, "solvent": lambda d: d["p_stable"].values > 0.9, "water": lambda d: d[["p0", "p1", "p2", "p3"]].values.argmax(1) >= 2}
for mi, mode in enumerate(("group",)):
    for ti, task in enumerate(sd.TASKS):
        df = pd.read_parquet(OUT + f"pred_{mode}_{task}_s0.parquet")
        cal, test = df[df.split == "cal"].reset_index(drop=True), df[df.split == "test"].reset_index(drop=True)
        bg = test["group_id"].values if mode == "group" else np.arange(len(test))
        curve = recall_curve(cal.score.values, cal.stable.values, test.score.values, test.stable.values, ALPHAS)
        rows = []
        for _, r in curve.iterrows():
            t = r["threshold"]
            rc = cluster_bootstrap(lambda i: stage_outcome(test.score.values[i], test.stable.values[i], t)["recall"], bg, 1000, seed=1)
            kf = cluster_bootstrap(lambda i: stage_outcome(test.score.values[i], test.stable.values[i], t)["kept_fraction"], bg, 1000, seed=2)
            rows.append(dict(**{k: (None if (isinstance(v, float) and not np.isfinite(v)) else v) for k, v in r.items()}, recall_lo=rc[1], recall_hi=rc[2], kept_lo=kf[1], kept_hi=kf[2]))
        pr = paper[task](test); st = test.stable.values.astype(bool)
        base = dict(recall=float((pr & st).sum() / st.sum()), kept_fraction=float(pr.mean()), precision=float((pr & st).sum() / max(1, pr.sum())))
        pool = pd.concat([cal, test]); pg = pool["group_id"].values if mode == "group" else np.arange(len(pool))
        valid = [validity_check(pool.score.values, pool.stable.values, pg, a, n_rep=500, seed=3) for a in VALID_ALPHAS]
        pos_cal = cal.score.values[cal.stable.values.astype(bool)]; pac = []
        for a in VALID_ALPHAS:
            t = conformal_threshold_pac(pos_cal, a, 0.1); o = stage_outcome(test.score.values, test.stable.values, t)
            v = validity_check(pool.score.values, pool.stable.values, pg, a, n_rep=500, seed=3, threshold_fn=lambda ps, al: conformal_threshold_pac(ps, al, 0.1))
            pac.append(dict(alpha=a, delta=0.1, threshold=t, test_recall=o["recall"], test_kept=o["kept_fraction"], valid=v))
        res[f"{mode}_{task}"] = dict(n_cal=int(len(cal)), n_cal_pos=int(cal.stable.sum()), n_test=int(len(test)), n_test_pos=int(test.stable.sum()), curve=rows, paper_rule=base, validity=valid, pac=pac)
        c = pd.DataFrame(rows)
        print(f"\n=== {mode} split, {task}: calibration {len(cal)} MOFs ({int(cal.stable.sum())} stable), test {len(test)} ({int(test.stable.sum())} stable); paper rule on the same test set: recall {base['recall']:.3f}, kept {base['kept_fraction']:.3f}")
        print(c[["alpha", "target_recall", "k", "threshold", "recall", "recall_lo", "recall_hi", "kept_fraction", "kept_lo", "kept_hi", "precision"]].round(3).to_string(index=False))
        print("   validity (500 random re-partitions of the pooled held-out groups into calibration/test, scores fixed):")
        for v in valid:
            print(f"   alpha {v['alpha']:.2f}: target recall {v['target']:.2f} | mean realised recall {v['mean_recall']:.3f} (5-95% of partitions: {v['p05']:.3f}-{v['p95']:.3f}) | partitions at/above target: {v['frac_reps_at_or_above_target']:.1%} | mean kept {v['mean_kept_fraction']:.3f}")
print("\n=== PAC (training-conditional, delta = 0.1) variant: recall >= 1-alpha with probability >= 0.9 over the calibration draw")
for k, v in res.items():
    for r in v["pac"]:
        print(f"{k:16s} alpha {r['alpha']:.2f}: threshold {r['threshold']:.3f} | test recall {r['test_recall']:.3f}, kept {r['test_kept']:.3f} | over 500 re-partitions: mean recall {r['valid']['mean_recall']:.3f}, partitions at/above target {r['valid']['frac_reps_at_or_above_target']:.1%}, mean kept {r['valid']['mean_kept_fraction']:.3f}")
json.dump(res, open(OUT + "conformal.json", "w"), indent=1, default=float)
