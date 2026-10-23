"""Phase 2, step 3: split conformal recall control per stage (calibration set held out by group), recall and kept-fraction curves.
python -I scripts/p2_03_conformal.py   (needs p2_01 and p2_02)"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.conformal.recall import conformal_threshold, conformal_threshold_pac, recall_curve, stage_outcome, validity_check
from src.eval.metrics import cluster_bootstrap
from src.features import stability_data as sd

OUT = "data/processed/phase2/"
ALPHAS = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
VALID_ALPHAS = [0.05, 0.10, 0.20]
res = {}
paper = {"thermal": lambda d: d["pred"].values >= 350, "solvent": lambda d: d["p_stable"].values > 0.9, "water": lambda d: d[["p0", "p1", "p2", "p3"]].values.argmax(1) >= 2}
fig, axes = plt.subplots(2, 3, figsize=(14, 7.5)), None
for mi, mode in enumerate(("group", "random")):
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
        c = pd.DataFrame(rows); ax = axes[0][mi * 0 + ti] if False else None
        print(f"\n=== {mode} split, {task}: calibration {len(cal)} MOFs ({int(cal.stable.sum())} stable), test {len(test)} ({int(test.stable.sum())} stable); paper rule on the same test set: recall {base['recall']:.3f}, kept {base['kept_fraction']:.3f}")
        print(c[["alpha", "target_recall", "k", "threshold", "recall", "recall_lo", "recall_hi", "kept_fraction", "kept_lo", "kept_hi", "precision"]].round(3).to_string(index=False))
        print("   validity (500 random re-partitions of the pooled held-out groups into calibration/test, scores fixed):")
        for v in valid:
            print(f"   alpha {v['alpha']:.2f}: target recall {v['target']:.2f} | mean realised recall {v['mean_recall']:.3f} (5-95% of partitions: {v['p05']:.3f}-{v['p95']:.3f}) | partitions at/above target: {v['frac_reps_at_or_above_target']:.1%} | mean kept {v['mean_kept_fraction']:.3f}")
print("\n=== PAC (training-conditional, delta = 0.1) variant: recall >= 1-alpha with probability >= 0.9 over the calibration draw")
for k, v in res.items():
    for r in v["pac"]:
        print(f"{k:16s} alpha {r['alpha']:.2f}: threshold {r['threshold']:.3f} | test recall {r['test_recall']:.3f}, kept {r['test_kept']:.3f} | over 500 re-partitions: mean recall {r['valid']['mean_recall']:.3f}, partitions at/above target {r['valid']['frac_reps_at_or_above_target']:.1%}, mean kept {r['valid']['mean_kept_fraction']:.3f}")
# figure
fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
for mi, mode in enumerate(("group", "random")):
    for ti, task in enumerate(sd.TASKS):
        c = pd.DataFrame(res[f"{mode}_{task}"]["curve"]); ax = axes[mi][ti]
        ax.plot(c.target_recall, c.target_recall, "k--", lw=1, label="target recall 1-alpha")
        ax.fill_between(c.target_recall, c.recall_lo, c.recall_hi, alpha=0.2, color="C0"); ax.plot(c.target_recall, c.recall, "o-", color="C0", label="test recall (95% CI over " + ("groups" if mode == "group" else "MOFs") + ")")
        ax.fill_between(c.target_recall, c.kept_lo, c.kept_hi, alpha=0.2, color="C3"); ax.plot(c.target_recall, c.kept_fraction, "s-", color="C3", label="fraction of test MOFs kept")
        b = res[f"{mode}_{task}"]["paper_rule"]; ax.plot([b["recall"]], [b["kept_fraction"]], "k*", ms=11, label="paper's fixed rule (recall, kept)")
        ax.set_title(f"{task} - {mode} split"); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3)
        if mi == 1: ax.set_xlabel("target recall (1 - alpha)")
        if ti == 0: ax.set_ylabel("recall / kept fraction")
axes[0][0].legend(fontsize=7, loc="lower right")
plt.tight_layout(); plt.savefig(OUT + "figs/conformal_curves.png", dpi=110)
json.dump(res, open(OUT + "conformal.json", "w"), indent=1, default=float)
print("\nfigure: data/processed/phase2/figs/conformal_curves.png")
