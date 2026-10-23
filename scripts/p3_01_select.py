"""Phase 3, step 1: model selection by 5-fold GROUP cross-validation on the TRAINING groups only (calibration and test MOFs are never read here).
Candidates: XGBoost / ExtraTrees / RandomForest x class weighting x in-fold feature selection (src/models/boosted.py).
Selection metric: pooled out-of-fold AUC for 'truly stable'. Also evaluates the mean of the top-3 configurations ('small ensemble').
python -I scripts/p3_01_select.py   (writes data/processed/phase3/cv_results.json, selected.json)"""
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupKFold

from src.features import stability_data as sd
from src.models import boosted

OUT = "data/processed/phase3/"
groups = pd.read_parquet("data/processed/phase2/groups.parquet").set_index("name")
SEED, TOPK = 0, 3


def cv_metrics(task, y, stable, o):
    m = {"AUC_stable": float(roc_auc_score(stable, o["score"]))}
    if task == "thermal":
        m["MAE_degC"] = float(mean_absolute_error(y, o["pred"]))
    elif task == "solvent":
        m["accuracy"] = float(accuracy_score(y, (o["p_stable"] >= .5).astype(int)))
    else:
        m["accuracy_4class"] = float(accuracy_score(y, np.column_stack([o[f"p{i}"] for i in range(4)]).argmax(1)))
    return m


res, selected = {}, {}
for task in sd.TASKS:
    d = sd.load_task(task); sp = groups.loc[d.index, "split"].values
    tr = np.where(sp == "train")[0]                       # ONLY training groups
    assert not (sp[tr] != "train").any()
    X = sd.feature_matrix(d, task)[tr]; y = d["y"].values[tr]; st = d["stable"].values[tr]; g = groups.loc[d.index[tr], "group_id"].values
    folds = list(GroupKFold(n_splits=5).split(X, y, g))
    oofs, rows = {}, []
    for c in boosted.config_grid(task):
        t0 = time.time(); keys = None; parts = []
        oof = {}
        for tri, vai in folds:
            o = boosted.fit_predict(task, c, X[tri], y[tri], X[vai], SEED)
            for k, v in o.items():
                oof.setdefault(k, np.zeros(len(tr)))[vai] = v
        oofs[boosted.cfg_name(c)] = oof
        rows.append(dict(config=boosted.cfg_name(c), cfg=c, **cv_metrics(task, y, st, oof)))
        print(f"[{task}] {boosted.cfg_name(c):60s} " + " ".join(f"{k} {v:.3f}" for k, v in rows[-1].items() if k not in ("config", "cfg")) + f" | {time.time() - t0:.0f}s", flush=True)
    rows.sort(key=lambda r: -r["AUC_stable"])
    top = rows[:TOPK]
    ens = boosted.average([oofs[r["config"]] for r in top])
    ens_m = cv_metrics(task, y, st, ens)
    best_single = rows[0]
    choice = "ensemble" if ens_m["AUC_stable"] > best_single["AUC_stable"] else "single"
    selected[task] = dict(choice=choice, single=best_single["cfg"], ensemble=[r["cfg"] for r in top], cv_single=cv_metrics(task, y, st, oofs[best_single["config"]]), cv_ensemble=ens_m)
    res[task] = dict(n_train=int(len(tr)), n_train_groups=int(len(np.unique(g))), configs=rows)
    print(f"== {task}: best single {best_single['config']} CV AUC {best_single['AUC_stable']:.3f}; top-{TOPK} ensemble CV {ens_m}; choice: {choice}", flush=True)
json.dump(res, open(OUT + "cv_results.json", "w"), indent=1, default=float)
json.dump(selected, open(OUT + "selected.json", "w"), indent=1, default=float)
print("DONE")
