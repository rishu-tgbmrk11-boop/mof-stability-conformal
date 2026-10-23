"""Phase 3, step 2: refit the CV-selected configurations on all training groups (5 seeds), predict everything, and compare with the Phase 2 baseline on the
SAME group-aware test MOFs: metrics with group-bootstrap intervals, plus paired differences (new - baseline) with group-bootstrap intervals.
The test split is read only here, after selection (p3_01) is frozen.
python -I scripts/p3_02_final_eval.py"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score

from src.eval.metrics import cluster_bootstrap, safe_auc
from src.features import stability_data as sd
from src.models import boosted

OUT, P2 = "data/processed/phase3/", "data/processed/phase2/"
SEEDS, NBOOT = [0, 1, 2, 3, 4], 1000
groups = pd.read_parquet(P2 + "groups.parquet").set_index("name")
sel = json.load(open(OUT + "selected.json"))


def fns(task, df):
    y, st = df["y"].values, df["stable"].values
    if task == "thermal":
        return {"AUC_stable": lambda i: safe_auc(st[i], df.score.values[i]), "MAE_degC": lambda i: mean_absolute_error(y[i], df.pred.values[i]),
                "acc_stable_at_350": lambda i: accuracy_score(st[i], (df.pred.values[i] >= 350).astype(int))}
    if task == "solvent":
        p = df.p_stable.values
        return {"AUC": lambda i: safe_auc(y[i], p[i]), "accuracy": lambda i: accuracy_score(y[i], (p[i] >= .5).astype(int))}
    P = df[["p0", "p1", "p2", "p3"]].values
    return {"AUC_stable": lambda i: safe_auc(st[i], df.p_stable.values[i]), "accuracy_4class": lambda i: accuracy_score(y[i], P[i].argmax(1)),
            "acc_stable_argmax_ge2": lambda i: accuracy_score(st[i], (P[i].argmax(1) >= 2).astype(int))}


report = {}
for task in sd.TASKS:
    d = sd.load_task(task); sp = groups.loc[d.index, "split"].values; gid = groups.loc[d.index, "group_id"].values
    X = sd.feature_matrix(d, task); tr = np.where(sp == "train")[0]
    members = [sel[task]["single"]] if sel[task]["choice"] == "single" else sel[task]["ensemble"]
    per_seed = []
    for seed in SEEDS:
        o = boosted.average([boosted.fit_predict(task, c, X[tr], d["y"].values[tr], X, seed) for c in members])
        df = pd.DataFrame({"name": d.index, "split": sp, "group_id": gid, "y": d["y"].values, "stable": d["stable"].values, **o})
        df.to_parquet(OUT + f"pred_group_{task}_s{seed}.parquet")
        te = df[df.split == "test"].reset_index(drop=True)
        per_seed.append({k: f(np.arange(len(te))) for k, f in fns(task, te).items()})
        print(f"[{task} seed {seed}] " + ", ".join(f"{k} {v:.3f}" for k, v in per_seed[-1].items()), flush=True)
    new = pd.read_parquet(OUT + "pred_group_%s_s0.parquet" % task); new = new[new.split == "test"].set_index("name")
    base = pd.read_parquet(P2 + f"pred_group_{task}_s0.parquet"); base = base[base.split == "test"].set_index("name").loc[new.index]
    assert (base.stable.values == new.stable.values).all()
    new, base = new.reset_index(), base.reset_index(); bg = new.group_id.values
    fn, fb = fns(task, new), fns(task, base)
    ci_new = {k: cluster_bootstrap(f, bg, NBOOT, seed=0) for k, f in fn.items()}
    ci_base = {k: cluster_bootstrap(f, bg, NBOOT, seed=0) for k, f in fb.items()}
    ci_diff = {k: cluster_bootstrap(lambda i, k=k: fn[k](i) - fb[k](i), bg, NBOOT, seed=0) for k in fn}
    arr = pd.DataFrame(per_seed)
    report[task] = dict(choice=sel[task]["choice"], members=members, n_test=int(len(new)), n_test_groups=int(len(np.unique(bg))), new=ci_new, baseline=ci_base, diff=ci_diff,
                        seeds_mean=arr.mean().to_dict(), seeds_sd=arr.std(ddof=1).to_dict())
    print(f"== {task} ({sel[task]['choice']}):" + "".join(f"\n   {k}: new {ci_new[k][0]:.3f} [{ci_new[k][1]:.3f}, {ci_new[k][2]:.3f}] | base {ci_base[k][0]:.3f} [{ci_base[k][1]:.3f}, {ci_base[k][2]:.3f}] | diff {ci_diff[k][0]:+.3f} [{ci_diff[k][1]:+.3f}, {ci_diff[k][2]:+.3f}]" for k in ci_new), flush=True)
json.dump(report, open(OUT + "final_eval.json", "w"), indent=1, default=float)
print("DONE")
