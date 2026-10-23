"""Audit step 4: retrain from the released data. python -I scripts/audit_04_retrain.py   (writes data/processed/audit/retrain.json)"""
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from src.features.paper_splits import paper_random_split, paper_val_split
from src.models import safe_pickle as sp
from src.models.stability_nets import SolventNet, ThermalNet, predict, train_released_recipe

R = "data/raw/optimof/"
OUT = "data/processed/audit/"
dev = "cuda" if torch.cuda.is_available() else "cpu"
L = lambda p: sp.safe_joblib_load(R + p)
res = {}
SEEDS = [0, 1, 2, 3, 4]


def prep(task):
    d = L(f"data/{task}/{task}_split_data.pkl")
    Xtr, Xte, ytr, yte = paper_random_split(d, task)
    sc = StandardScaler().fit(Xtr)                      # scaler fitted on the whole training part, as in the released scripts
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    Xa, Xv, ya, yv = paper_val_split(pd.DataFrame(Xtr_s), ytr.reset_index(drop=True))   # 10 % validation carved from the training part
    if task == "solvent":
        ya, yv, ytr, yte = (np.where(v.values < 0, 0, v.values) for v in (ya, yv, ytr, yte))
    else:
        ya, yv, ytr, yte = ya.values, yv.values, ytr.values, yte.values
    return Xa.values, ya, Xv.values, yv, Xtr_s, ytr, Xte_s, yte


def metrics(task, net, Xte, yte, Xtr, ytr):
    o, ot = predict(net, Xte, dev), predict(net, Xtr, dev)
    if task == "thermal":
        return dict(test_mae=mean_absolute_error(yte, o), test_rmse=mean_squared_error(yte, o) ** .5, test_r2=r2_score(yte, o), train_r2=r2_score(ytr, ot), train_mae=mean_absolute_error(ytr, ot))
    p, pt = 1 / (1 + np.exp(-o)), 1 / (1 + np.exp(-ot))
    return dict(test_acc=accuracy_score(yte, p >= .5), test_auc=roc_auc_score(yte, p), train_acc=accuracy_score(ytr, pt >= .5))


configs = {
    "thermal": {
        "released_recipe": dict(make=lambda: ThermalNet(170, [2048, 1024, 512, 256, 128, 64, 32], 1), loss=nn.MSELoss(), lr=1e-4, bs=512, epochs=1000, pat=100),
        "SI_TableS5": dict(make=lambda: SolventNet(170, [128, 512, 2048, 256, 1024], 1, dropout_prob=0.124), loss=nn.MSELoss(), lr=2e-3, bs=512, epochs=100, pat=100),
    },
    "solvent": {
        "released_recipe": dict(make=lambda: SolventNet(170, [512, 256, 128, 64, 32, 16], 1, dropout_prob=0.2), loss=nn.BCEWithLogitsLoss(), lr=1e-3, bs=256, epochs=1000, pat=200),
        "SI_TableS5": dict(make=lambda: SolventNet(170, [128, 256, 1024, 2048, 128, 256], 1, dropout_prob=0.025), loss=nn.BCEWithLogitsLoss(), lr=7.64e-5, bs=256, epochs=100, pat=100),
    },
}
for task in ("thermal", "solvent"):
    Xa, ya, Xv, yv, Xtr, ytr, Xte, yte = prep(task)
    print(f"\n##### {task}: train {len(Xa)} / val {len(Xv)} / test {len(Xte)}  (device {dev})", flush=True)
    res[task] = {}
    for cname, c in configs[task].items():
        for fix in ((True, False) if cname == "released_recipe" else (True,)):
            tag = f"{cname}{'' if fix else '+fixed_restore'}"
            rows = []
            for s in SEEDS:
                t0 = time.time()
                net = c["make"]()
                h = train_released_recipe(net, Xa, ya, Xv, yv, c["loss"], c["lr"], c["bs"], c["epochs"], c["pat"], seed=s, device=dev, faithful_best_state=fix)
                m = metrics(task, net, Xte, yte, Xtr, ytr)
                m.update(seed=s, epochs_run=h["epochs_run"], best_epoch=h["best_epoch"], best_val=h["best_val"], final_val=h["final_val"], restore_noop=h["restore_changed_nothing"], secs=round(time.time() - t0, 1))
                rows.append(m)
            df = pd.DataFrame(rows)
            res[task][tag] = rows
            keys = [k for k in df.columns if k.startswith(("test_", "train_"))]
            print(f"  [{tag}] mean over seeds {SEEDS}:", {k: round(df[k].mean(), 3) for k in keys}, "| sd:", {k: round(df[k].std(), 3) for k in keys if k.startswith("test_")}, "| epochs_run", df.epochs_run.tolist(), "best_epoch", df.best_epoch.tolist(), "| restore was a no-op in", int(df.restore_noop.sum()), "of", len(df), flush=True)
json.dump(res, open(OUT + "retrain.json", "w"), indent=1, default=float)

# ---- water: re-run the released grid searches on the released split
d = L("data/water_and_haz/water_split_data.pkl"); d = d.assign(water=d["water"] - 1)
Xtr, Xte, ytr, yte = paper_random_split(d, "water")
print(f"\n##### water: train {len(Xtr)}, test {len(Xte)}", flush=True)
t0 = time.time()
gs = GridSearchCV(RandomForestClassifier(random_state=0), {"n_estimators": [100, 150, 200, 250, 300], "max_depth": [None, 10, 20], "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4], "bootstrap": [True, False]}, cv=5, scoring="accuracy", n_jobs=-1).fit(Xtr, ytr)
pr = gs.best_estimator_.predict_proba(Xte)
res["water_rf_regrid"] = dict(best_params=gs.best_params_, cv_accuracy=gs.best_score_, test_acc=accuracy_score(yte, pr.argmax(1)), test_auc=roc_auc_score(yte, pr, multi_class="ovr"), secs=round(time.time() - t0))
print("  RF re-grid:", res["water_rf_regrid"], flush=True)
rfs = RandomForestClassifier(n_estimators=250, max_depth=20, min_samples_split=8, random_state=0).fit(Xtr, ytr); prs = rfs.predict_proba(Xte)
res["water_rf_SI_config"] = dict(test_acc=accuracy_score(yte, prs.argmax(1)), test_auc=roc_auc_score(yte, prs, multi_class="ovr"))
print("  RF with SI Table S5 settings (250 trees, depth 20, min_samples_split 8):", res["water_rf_SI_config"], flush=True)
t0 = time.time()
gx = GridSearchCV(xgb.XGBClassifier(objective="multi:softprob", random_state=0, n_jobs=1), {"n_estimators": [100, 150, 200, 250, 300], "max_depth": [3, 5, 7], "learning_rate": [0.01, 0.1, 0.3], "subsample": [0.8, 1.0], "colsample_bytree": [0.8, 1.0]}, cv=5, scoring="accuracy", n_jobs=8).fit(Xtr, ytr)
px = gx.best_estimator_.predict_proba(Xte)
res["water_xgb_regrid"] = dict(best_params=gx.best_params_, cv_accuracy=gx.best_score_, test_acc=accuracy_score(yte, px.argmax(1)), test_auc=roc_auc_score(yte, px, multi_class="ovr"), secs=round(time.time() - t0))
print("  XGB re-grid:", res["water_xgb_regrid"], "| repo docstring: acc 0.6758, AUC 0.8492", flush=True)
json.dump(res, open(OUT + "retrain.json", "w"), indent=1, default=float)
print("DONE")
