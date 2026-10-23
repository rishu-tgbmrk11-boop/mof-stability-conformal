"""Phase 2, step 4: three-stage cascade with a union bound, evaluated on the MOFs that carry all three labels using group cross-fitting.
For each repetition the groups are split into K=5 folds; for each fold the three stages are trained (train groups) and calibrated (calibration groups) on the OTHER
folds' groups and predict the held-out fold, so every MOF gets out-of-fold scores and thresholds. Cascade rule: pass iff all three stages keep the MOF, with
per-stage levels alpha_k = alpha / 3 (union bound).
python -I scripts/p2_04_cascade.py   (needs p2_01; ~10 minutes)"""
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
import torch
from scipy.stats import beta

from src.conformal.recall import conformal_threshold
from src.eval.metrics import cluster_bootstrap
from src.features import stability_data as sd
from src.features.groups import assign_groups
from src.models import stages

OUT = "data/processed/phase2/"
K, REPS = 5, [0, 1, 2, 3, 4]
ALPHAS = [0.05, 0.10, 0.20, 0.30]
ALL_LEVELS = sorted(set(ALPHAS) | {round(a / 3, 6) for a in ALPHAS})
dev = "cuda" if torch.cuda.is_available() else "cpu"
groups = pd.read_parquet(OUT + "groups.parquet").set_index("name")
gtab = groups.drop_duplicates("group_id")[["group_id", "group_size"]].reset_index(drop=True)
data = {t: sd.load_task(t) for t in sd.TASKS}
paper = {"thermal": lambda d: d["pred"].values >= 350, "solvent": lambda d: d["p_stable"].values > 0.9, "water": lambda d: d[["p0", "p1", "p2", "p3"]].values.argmax(1) >= 2}


def cp_interval(k, n, level=0.95):
    lo = 0.0 if k == 0 else beta.ppf((1 - level) / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - (1 - level) / 2, k + 1, n - k)
    return float(lo), float(hi)


def val_mask(names, seed):
    gi = groups.loc[names]
    sub = pd.DataFrame({"name": names, "group_id": gi["group_id"].values}); sub["group_size"] = sub.groupby("group_id")["name"].transform("size")
    return assign_groups(sub, {"fit": 0.9, "val": 0.1}, seed=seed).values == "val"


def one_repetition(rep):
    fold_of = assign_groups(gtab, {f"f{i}": 1 / K for i in range(K)}, seed=rep).values
    fold_by_g = dict(zip(gtab.group_id, fold_of))
    oof = {t: [] for t in sd.TASKS}
    for fi in range(K):
        f = f"f{fi}"
        rest = gtab[fold_of != f].reset_index(drop=True)
        role_g = dict(zip(rest.group_id, assign_groups(rest, {"train": 0.75, "cal": 0.25}, seed=1000 * rep + fi).values))
        for task in sd.TASKS:
            d = data[task]; gid = groups.loc[d.index, "group_id"].values
            held = np.array([fold_by_g[g] == f for g in gid]); role = np.array([role_g.get(g, "held") for g in gid])
            tr = np.where(role == "train")[0]; vm = val_mask(d.index[tr], 1000 * rep + fi)
            X = sd.feature_matrix(d, task)
            if task == "water":
                m = stages.fit_water(X[tr], d["y"].values[tr], 0)
            else:
                fit = stages.fit_thermal if task == "thermal" else stages.fit_solvent
                m = fit(X[tr[~vm]], d["y"].values[tr[~vm]], X[tr[vm]], d["y"].values[tr[vm]], 0, dev)
            idx = np.where(held | (role == "cal"))[0]
            sc = stages.scores(m, X[idx], dev)
            df = pd.DataFrame({"name": d.index[idx], "group_id": gid[idx], "role": np.where(held[idx], "held", "cal"), "stable": d["stable"].values[idx], "y": d["y"].values[idx], **sc})
            pos = df[(df.role == "cal") & (df.stable == 1)].score.values
            th = {a: conformal_threshold(pos, a) for a in ALL_LEVELS}
            ho = df[df.role == "held"].copy()
            for a in ALL_LEVELS:
                ho[f"t_{a}"] = th[a]
            ho["fold"] = f; ho["n_cal_pos"] = len(pos)
            oof[task].append(ho)
    return {t: pd.concat(v, ignore_index=True) for t, v in oof.items()}


def evaluate(rep, oof):
    out = {"rep": rep, "stage": {}, "cascade": {}}
    for task in sd.TASKS:                                              # per-stage coverage over ALL out-of-fold MOFs
        o = oof[task]; st = o.stable.values.astype(bool)
        out["stage"][task] = {str(a): dict(recall=float(((o.score.values >= o[f"t_{a}"].values) & st).sum() / st.sum()), kept=float((o.score.values >= o[f"t_{a}"].values).mean()), n_pos=int(st.sum()), n=int(len(o))) for a in ALPHAS}
    idx = {t: oof[t].set_index("name") for t in sd.TASKS}
    names = sorted(set(idx["thermal"].index) & set(idx["solvent"].index) & set(idx["water"].index))
    tab = pd.DataFrame(index=names)
    for t in sd.TASKS:
        tab[f"stable_{t}"] = idx[t].loc[names, "stable"].values; tab[f"score_{t}"] = idx[t].loc[names, "score"].values
        tab["group_id"] = groups.loc[names, "group_id"].values
        for a in ALL_LEVELS:
            tab[f"pass_{t}_{a}"] = idx[t].loc[names, "score"].values >= idx[t].loc[names, f"t_{a}"].values
        if t == "thermal":
            tab["pass_paper_thermal"] = idx[t].loc[names, "pred"].values >= 350
        elif t == "solvent":
            tab["pass_paper_solvent"] = idx[t].loc[names, "p_stable"].values > 0.9
        else:
            tab["pass_paper_water"] = idx[t].loc[names, ["p0", "p1", "p2", "p3"]].values.argmax(1) >= 2
    tab["all_stable"] = tab[[f"stable_{t}" for t in sd.TASKS]].all(axis=1)
    out["n_labelled"], out["n_all_stable"] = int(len(tab)), int(tab.all_stable.sum())
    pos, neg = tab[tab.all_stable], tab[~tab.all_stable]
    def rule(name, passes):
        p_all = passes.all(axis=1)
        return dict(recall=float(p_all[tab.all_stable].mean()), n_pass_pos=int(p_all[tab.all_stable].sum()), kept=float(p_all.mean()), false_pass_rate=float(p_all[~tab.all_stable].mean()),
                    stage_recall={t: float(passes[t][tab.all_stable].mean()) for t in sd.TASKS})
    for a in ALPHAS:
        ak = round(a / 3, 6)
        out["cascade"][f"union_{a}"] = rule("union", pd.DataFrame({t: tab[f"pass_{t}_{ak}"] for t in sd.TASKS}))
        out["cascade"][f"each_alpha_{a}"] = rule("each", pd.DataFrame({t: tab[f"pass_{t}_{a}"] for t in sd.TASKS}))
    out["cascade"]["paper_rules"] = rule("paper", pd.DataFrame({t: tab[f"pass_paper_{t}"] for t in sd.TASKS}))
    return out, tab


allres, tabs = [], {}
for rep in REPS:
    t0 = time.time(); oof = one_repetition(rep); res, tab = evaluate(rep, oof); allres.append(res); tabs[rep] = tab
    c = res["cascade"]
    print(f"[rep {rep}] {time.time() - t0:.0f}s | labelled {res['n_labelled']}, all-stable {res['n_all_stable']} | cascade recall on all-stable (union bound, alpha=0.05/0.10/0.20/0.30): " + " / ".join(f"{c[f'union_{a}']['recall']:.3f} ({c[f'union_{a}']['n_pass_pos']}/{res['n_all_stable']})" for a in ALPHAS) + f" | paper rules {c['paper_rules']['recall']:.3f}", flush=True)
tabs[0].to_parquet(OUT + "cascade_oof_rep0.parquet")
json.dump(dict(results=allres, alphas=ALPHAS, K=K, reps=REPS), open(OUT + "cascade.json", "w"), indent=1, default=float)
print("DONE")
