"""Phase 2, step 2: retrain the three stability stages honestly (fixed early-stopping restore, group-aware splits) and evaluate with
bootstrap intervals over groups; the same pipeline is run with a MOF-level random split for comparison.
python -I scripts/p2_02_train_eval.py   (run from the repo root; writes data/processed/phase2/pred_*.parquet and metrics.json)"""
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, brier_score_loss, mean_absolute_error, r2_score, roc_auc_score

from src.eval.metrics import cluster_bootstrap, ece, safe_auc
from src.features import stability_data as sd
from src.features.groups import assign_groups
from src.models import stages

OUT = "data/processed/phase2/"
SEEDS = [0, 1, 2, 3, 4]
NBOOT = 1000
dev = "cuda" if torch.cuda.is_available() else "cpu"
groups = pd.read_parquet(OUT + "groups.parquet").set_index("name")
data = {t: sd.load_task(t) for t in sd.TASKS}
metrics, info = {}, {}


def val_split(train_names, mode, seed=0):
    """10 % of the training MOFs held out for early stopping: whole groups in group mode, random MOFs in random mode."""
    gi = groups.loc[train_names]
    if mode == "group":
        sub = pd.DataFrame({"name": train_names, "group_id": gi["group_id"].values})
        sub["group_size"] = sub.groupby("group_id")["name"].transform("size")
        return (assign_groups(sub, {"fit": 0.9, "val": 0.1}, seed=seed).values == "val")
    rng = np.random.RandomState(seed)
    m = np.zeros(len(train_names), bool); m[rng.permutation(len(train_names))[: int(round(0.1 * len(train_names)))]] = True
    return m


def metric_fns(task, test, cal):
    """Return {metric_name: f(idx) -> value} on the test rows; `cal` is used only to fit the isotonic map for the thermal ECE."""
    y, st = test["y"].values, test["stable"].values
    if task == "thermal":
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(cal["score"].values, cal["stable"].values)
        ps = iso.predict(test["score"].values)
        return {"R2": lambda i: r2_score(y[i], test["pred"].values[i]), "MAE_degC": lambda i: mean_absolute_error(y[i], test["pred"].values[i]),
                "AUC_stable": lambda i: safe_auc(st[i], test["score"].values[i]), "ECE_stable_isotonic": lambda i: ece(ps[i], st[i]),
                "acc_stable_at_350": lambda i: accuracy_score(st[i], (test["pred"].values[i] >= 350).astype(int))}
    if task == "solvent":
        p = test["p_stable"].values
        return {"accuracy": lambda i: accuracy_score(y[i], (p[i] >= .5).astype(int)), "AUC": lambda i: safe_auc(y[i], p[i]), "ECE": lambda i: ece(p[i], y[i]), "Brier": lambda i: brier_score_loss(y[i], p[i])}
    P = test[["p0", "p1", "p2", "p3"]].values; ps = test["p_stable"].values
    def auc4(i):
        try:
            return float(roc_auc_score(y[i], P[i], multi_class="ovr", labels=[0, 1, 2, 3]))
        except ValueError:
            return float("nan")
    return {"accuracy_4class": lambda i: accuracy_score(y[i], P[i].argmax(1)), "AUC_4class_macro": auc4, "AUC_stable": lambda i: safe_auc(st[i], ps[i]),
            "ECE_stable": lambda i: ece(ps[i], st[i]), "Brier_stable": lambda i: brier_score_loss(st[i], ps[i]), "acc_stable_argmax_ge2": lambda i: accuracy_score(st[i], (P[i].argmax(1) >= 2).astype(int))}


for mode, col in (("group", "split"), ("random", "split_random")):
    for task in sd.TASKS:
        d = data[task]; sp_ = groups.loc[d.index, col].values; gid = groups.loc[d.index, "group_id"].values
        tr = np.where(sp_ == "train")[0]; vm = val_split(d.index[tr], mode)
        fit_idx, val_idx = tr[~vm], tr[vm]
        X = sd.feature_matrix(d, task)
        per_seed = []
        for seed in SEEDS:
            t0 = time.time()
            if task == "water":
                f = stages.fit_water(X[tr], d["y"].values[tr], seed)         # forest uses all training MOFs (no validation set needed)
            else:
                fit = stages.fit_thermal if task == "thermal" else stages.fit_solvent
                f = fit(X[fit_idx], d["y"].values[fit_idx], X[val_idx], d["y"].values[val_idx], seed, dev)
            sc = stages.scores(f, X, dev)
            df = pd.DataFrame({"name": d.index, "split": sp_, "group_id": gid, "y": d["y"].values, "stable": d["stable"].values, **sc})
            df.loc[val_idx, "split"] = "val"
            df.to_parquet(OUT + f"pred_{mode}_{task}_s{seed}.parquet")
            test, cal = df[df.split == "test"].reset_index(drop=True), df[df.split == "cal"].reset_index(drop=True)
            fns = metric_fns(task, test, cal)
            per_seed.append({k: f_(np.arange(len(test))) for k, f_ in fns.items()})
            info[f"{mode}_{task}_s{seed}"] = {**f.info, "n_fit": int(len(fit_idx)) if task != "water" else int(len(tr)), "n_val": int(len(val_idx)) if task != "water" else 0, "secs": round(time.time() - t0, 1)}
            print(f"[{mode:6s} {task:8s} seed {seed}] " + ", ".join(f"{k} {v:.3f}" for k, v in per_seed[-1].items()) + f"  | epochs {f.info.get('epochs_run', '-')} best {f.info.get('best_epoch', '-')} | {time.time() - t0:.0f}s", flush=True)
        # bootstrap intervals over groups for seed 0
        df = pd.read_parquet(OUT + f"pred_{mode}_{task}_s0.parquet")
        test, cal = df[df.split == "test"].reset_index(drop=True), df[df.split == "cal"].reset_index(drop=True)
        bg = test["group_id"].values if mode == "group" else np.arange(len(test))
        fns = metric_fns(task, test, cal)
        ci = {k: cluster_bootstrap(f_, bg, NBOOT, seed=0) for k, f_ in fns.items()}
        arr = pd.DataFrame(per_seed)
        metrics[f"{mode}_{task}"] = dict(n_test=int(len(test)), n_test_groups=int(len(np.unique(bg))), seed0_ci={k: list(v) for k, v in ci.items()}, seeds_mean=arr.mean().to_dict(), seeds_sd=arr.std(ddof=1).to_dict(), per_seed=per_seed)
        print(f"== {mode} {task}: n_test {len(test)}, clusters {len(np.unique(bg))}; seed-0 95% CI over {'groups' if mode == 'group' else 'MOFs'}: " + "; ".join(f"{k} {v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]" for k, v in ci.items()), flush=True)
json.dump(dict(metrics=metrics, info=info, seeds=SEEDS, n_boot=NBOOT), open(OUT + "metrics.json", "w"), indent=1, default=float)
print("DONE")
