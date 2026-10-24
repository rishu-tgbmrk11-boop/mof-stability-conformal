"""Phase 5, step 5: group-vs-random gaps as RANGES over many splits. Split seed s gives one random GROUP split (`random_group_assign`) and one MOF-level random
split (`random_assign`), both 60/20/20 over the union of the three tables; the same model family is trained on each and scored on the test part.
  family 'rf' : random forest (pre-specified, 300 trees; thermal = regressor, solvent/water = classifier), N = 100 splits
  family 'nn' : released networks with the FIXED restore (thermal, solvent) + released forest (water), N = 10 splits (training seed = split seed)
usage: python -I scripts/p5_05_split_ranges.py rf|nn   (repo root)"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, roc_auc_score

from src.features import stability_data as sd
from src.features.groups import random_assign, random_group_assign

fam = sys.argv[1]
N = {"rf": 100, "nn": 10}[fam]
OUT = "data/processed/phase5/"
FRAC = {"train": 0.6, "cal": 0.2, "test": 0.2}
groups = pd.read_parquet("data/processed/phase2/groups.parquet")[["name", "group_id", "group_size"]].set_index("name")
data = {t: sd.load_task(t) for t in sd.TASKS}
if fam == "nn":
    import torch
    from src.models import stages
    dev = "cuda" if torch.cuda.is_available() else "cpu"


def safe_auc(y, s):
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


rows = []
for s in range(N):
    sp = {"group": random_group_assign(groups.reset_index(), FRAC, seed=s).values, "random": random_assign(groups.index.tolist(), FRAC, seed=s).values}
    sp = {k: pd.Series(v, index=groups.index) for k, v in sp.items()}
    for mode in ("group", "random"):
        for t in sd.TASKS:
            d = data[t]; lab = sp[mode].loc[d.index].values; tr, te = np.where(lab == "train")[0], np.where(lab == "test")[0]
            X = sd.feature_matrix(d, t); y = d["y"].values; st = d["stable"].values
            if fam == "rf":
                if t == "thermal":
                    m = RandomForestRegressor(n_estimators=300, max_features="sqrt", n_jobs=8, random_state=s).fit(X[tr], d["label_raw"].values[tr]); p = m.predict(X[te])
                    met = dict(auc_stable=safe_auc(st[te], p), mae_degC=mean_absolute_error(d["label_raw"].values[te], p), r2=r2_score(d["label_raw"].values[te], p), accuracy_stable=accuracy_score(st[te], p >= 350))
                else:
                    m = RandomForestClassifier(n_estimators=300, max_features="sqrt", n_jobs=8, random_state=s).fit(X[tr], y[tr]); P = m.predict_proba(X[te])
                    if t == "solvent":
                        met = dict(auc_stable=safe_auc(st[te], P[:, 1]), accuracy_stable=accuracy_score(st[te], P[:, 1] >= .5))
                    else:
                        full = np.zeros((len(te), 4)); full[:, m.classes_.astype(int)] = P; ps = full[:, 2] + full[:, 3]
                        met = dict(auc_stable=safe_auc(st[te], ps), accuracy_stable=accuracy_score(st[te], ps >= .5), accuracy_4class=accuracy_score(y[te], full.argmax(1)))
            else:
                if t == "water":
                    f = stages.fit_water(X[tr], y[tr], s)
                else:                                                   # 10% of the training MOFs for early stopping: whole groups (group mode) or random MOFs (random mode)
                    names_tr = d.index[tr]; rng = np.random.RandomState(s)
                    if mode == "group":
                        sub = groups.loc[names_tr].reset_index(); sub["group_size"] = sub.groupby("group_id")["name"].transform("size")
                        vm = random_group_assign(sub, {"fit": 0.9, "val": 0.1}, seed=s).values == "val"
                    else:
                        vm = np.zeros(len(tr), bool); vm[rng.permutation(len(tr))[: int(round(0.1 * len(tr)))]] = True
                    fit = stages.fit_thermal if t == "thermal" else stages.fit_solvent
                    f = fit(X[tr[~vm]], y[tr[~vm]], X[tr[vm]], y[tr[vm]], s, dev)
                o = stages.scores(f, X[te], dev)
                if t == "thermal":
                    met = dict(auc_stable=safe_auc(st[te], o["pred"]), mae_degC=mean_absolute_error(y[te], o["pred"]), r2=r2_score(y[te], o["pred"]), accuracy_stable=accuracy_score(st[te], o["pred"] >= 350))
                elif t == "solvent":
                    met = dict(auc_stable=safe_auc(st[te], o["p_stable"]), accuracy_stable=accuracy_score(st[te], o["p_stable"] >= .5))
                else:
                    P4 = np.stack([o[f"p{c}"] for c in range(4)], 1)
                    met = dict(auc_stable=safe_auc(st[te], o["p_stable"]), accuracy_stable=accuracy_score(st[te], o["p_stable"] >= .5), accuracy_4class=accuracy_score(y[te], P4.argmax(1)))
            rows.append(dict(split_seed=s, mode=mode, stage=t, n_test=len(te), n_train=len(tr), **met))
    if (s + 1) % 10 == 0 or fam == "nn":
        print(f"split {s + 1}/{N} done", flush=True)
df = pd.DataFrame(rows); df.to_parquet(OUT + f"split_ranges_{fam}.parquet")
print(f"\nfamily {fam}: N = {N} splits (split seeds 0..{N - 1}); per stage and mode: mean [2.5%, 97.5% percentile across splits] (min-max); gap = random - group, paired by split seed")
for t in sd.TASKS:
    for k in [c for c in df.columns if c not in ("split_seed", "mode", "stage", "n_test", "n_train") and df[(df.stage == t)][c].notna().any()]:
        a = df[(df.stage == t) & (df["mode"] == "random")].set_index("split_seed")[k]; b = df[(df.stage == t) & (df["mode"] == "group")].set_index("split_seed")[k]; g = a - b
        f = lambda x: f"{x.mean():.3f} [{x.quantile(.025):.3f}, {x.quantile(.975):.3f}] ({x.min():.3f}-{x.max():.3f})"
        print(f"  {t:8s} {k:16s} random {f(a)} | group {f(b)} | gap {f(g)}; gap>0 in {int((g > 0).sum())}/{len(g)}" if k != "mae_degC" else f"  {t:8s} {k:16s} random {f(a)} | group {f(b)} | gap(group-random) {f(-g)}; group worse in {int((g < 0).sum())}/{len(g)}")
