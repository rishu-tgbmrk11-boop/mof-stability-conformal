"""Phase 5, step 1: the SI's split = train_test_split(test_size=0.2, random_state=0) on the released CSV ROWS. Retrain per stage with
(a) the released recipe as coded (early-stopping restore is a no-op), (b) the released recipe with the fixed restore, (c)/(d) the SI Table S5 settings
(as coded / fixed restore; batch size and patience are not in the SI, released values used). Water: released forest and SI Table S5 forest.
All variants are fixed in advance; nothing is selected on the test split. python -I scripts/p5_01_csv_split_retrain.py   (repo root)"""
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, precision_recall_fscore_support, precision_score, r2_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.features import stability_data as sd
from src.models.stability_nets import SolventNet, ThermalNet, predict, train_released_recipe

OUT = "data/processed/phase5/"
SEEDS = [0, 1, 2, 3, 4]            # training seeds; split seed (random_state) = 0 as in the released scripts / SI
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.set_per_process_memory_fraction(0.8) if dev == "cuda" else None

# ---- the split, asserted against the SI's counts (SI pages S16-S19)
data = {t: sd.load_task(t) for t in sd.TASKS}
split = {}
for t, d in data.items():
    tr, te = train_test_split(np.arange(len(d)), test_size=0.2, random_state=0)
    split[t] = (tr, te)
yte_s = data["solvent"].iloc[split["solvent"][1]]["y"].values
wc = np.bincount(data["water"].iloc[split["water"][1]]["y"].values, minlength=4)
assert len(split["thermal"][1]) == 627 and len(split["solvent"][1]) == 436 and len(split["water"][1]) == 219
assert (yte_s.sum(), (1 - yte_s).sum()) == (275, 161) and wc.tolist() == [17, 68, 112, 22], (yte_s.sum(), wc)
print("split reproduces the SI: thermal n=627, solvent n=436 (275 stable / 161 not), water n=219 classes", wc.tolist(), flush=True)


def prep(t):
    d, (tr, te) = data[t], split[t]
    cols = sd.feature_columns()[t]
    Xtr, Xte = d.iloc[tr][cols].values.astype(float), d.iloc[te][cols].values.astype(float)
    ytr, yte = d.iloc[tr]["y"].values, d.iloc[te]["y"].values
    sc = StandardScaler().fit(Xtr)                                  # fitted on the whole training part, as in the released scripts
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    Xa, Xv, ya, yv = train_test_split(Xtr_s, ytr, test_size=0.1, random_state=0)   # 10 % validation from the training part, as released
    return dict(Xa=Xa, ya=ya, Xv=Xv, yv=yv, Xtr=Xtr_s, ytr=ytr, Xte=Xte_s, yte=yte, Xtr_raw=Xtr, Xte_raw=Xte, names_te=d.index[te].values, names_tr=d.index[tr].values)


CFG = {
    "thermal": {
        "released": dict(make=lambda: ThermalNet(170, [2048, 1024, 512, 256, 128, 64, 32], 1), loss=nn.MSELoss, lr=1e-4, bs=512, epochs=1000, pat=100),
        "SI_S5": dict(make=lambda: SolventNet(170, [128, 512, 2048, 256, 1024], 1, dropout_prob=0.124), loss=nn.MSELoss, lr=2e-3, bs=512, epochs=100, pat=100),
    },
    "solvent": {
        "released": dict(make=lambda: SolventNet(170, [512, 256, 128, 64, 32, 16], 1, dropout_prob=0.2), loss=nn.BCEWithLogitsLoss, lr=1e-3, bs=256, epochs=1000, pat=200),
        "SI_S5": dict(make=lambda: SolventNet(170, [128, 256, 1024, 2048, 128, 256], 1, dropout_prob=0.025), loss=nn.BCEWithLogitsLoss, lr=7.64e-5, bs=256, epochs=100, pat=100),
    },
}
results, preds = {}, []
for t in ("thermal", "solvent"):
    P = prep(t)
    print(f"\n##### {t}: train {len(P['Xtr'])} (fit {len(P['Xa'])} / val {len(P['Xv'])}), test {len(P['Xte'])}, device {dev}", flush=True)
    for cname, c in CFG[t].items():
        for restore in ("as_coded", "fixed"):
            tag = f"{cname}|{restore}"; rows = []
            for s in SEEDS:
                t0 = time.time(); net = c["make"]()
                h = train_released_recipe(net, P["Xa"], P["ya"], P["Xv"], P["yv"], c["loss"](), c["lr"], c["bs"], c["epochs"], c["pat"], seed=s, device=dev, faithful_best_state=(restore == "as_coded"))
                o = predict(net, P["Xte"], dev); ot = predict(net, P["Xtr"], dev)
                if t == "thermal":
                    m = dict(r2=r2_score(P["yte"], o), rmse=mean_squared_error(P["yte"], o) ** .5, mae=mean_absolute_error(P["yte"], o), train_r2=r2_score(P["ytr"], ot)); score = o
                else:
                    p = 1 / (1 + np.exp(-o)); yh = (p >= .5).astype(int); tn, fp, fn, tp = confusion_matrix(P["yte"], yh).ravel()
                    m = dict(accuracy=accuracy_score(P["yte"], yh), precision=precision_score(P["yte"], yh), recall=recall_score(P["yte"], yh), f1=f1_score(P["yte"], yh), auc=roc_auc_score(P["yte"], p),
                             tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), train_accuracy=accuracy_score(P["ytr"], (1 / (1 + np.exp(-ot)) >= .5).astype(int))); score = p
                m.update(seed=s, epochs_run=h["epochs_run"], best_epoch=h["best_epoch"], restore_noop=bool(h["restore_changed_nothing"]), secs=round(time.time() - t0, 1)); rows.append(m)
                preds.append(pd.DataFrame(dict(stage=t, variant=tag, seed=s, name=P["names_te"], y=P["yte"], score=score)))
            df = pd.DataFrame(rows); results[f"{t}|{tag}"] = rows
            keys = [k for k in df.columns if k not in ("seed", "secs", "restore_noop")]
            print(f"  [{tag}] mean (sd) over seeds {SEEDS}:", {k: f"{df[k].mean():.3f} ({df[k].std():.3f})" for k in keys}, "restore_noop", df.restore_noop.tolist(), flush=True)
            json.dump(results, open(OUT + "csv_retrain.json", "w"), indent=1, default=float)

# ---- water random forest (released best parameters; SI Table S5 parameters)
P = prep("water"); Xtr, Xte, ytr, yte = P["Xtr_raw"], P["Xte_raw"], P["ytr"], P["yte"]       # forest is trained on unscaled features, as released
RF = {"released": dict(n_estimators=200, max_depth=None, min_samples_split=2, min_samples_leaf=1, bootstrap=False, max_features="sqrt"),
      "SI_S5": dict(n_estimators=250, max_depth=20, min_samples_split=8)}
print(f"\n##### water: train {len(Xtr)}, test {len(Xte)}", flush=True)
for cname, kw in RF.items():
    rows = []
    for s in SEEDS:
        rf = RandomForestClassifier(random_state=s, n_jobs=8, **kw).fit(Xtr, ytr)
        pr = np.zeros((len(Xte), 4)); pr[:, rf.classes_.astype(int)] = rf.predict_proba(Xte); yh = pr.argmax(1)
        pp, rr, ff, _ = precision_recall_fscore_support(yte, yh, labels=[0, 1, 2, 3], zero_division=0)
        m = dict(accuracy=accuracy_score(yte, yh), auc_macro_ovr=roc_auc_score(yte, pr, multi_class="ovr", labels=[0, 1, 2, 3]), test_counts=np.bincount(yte, minlength=4).tolist(),
                 precision=pp.round(3).tolist(), recall=rr.round(3).tolist(), f1=ff.round(3).tolist(), seed=s)
        rows.append(m); preds.append(pd.DataFrame(dict(stage="water", variant=f"{cname}|forest", seed=s, name=P["names_te"], y=yte, score=pr[:, 2] + pr[:, 3], pred_class=yh)))
    results[f"water|{cname}"] = rows
    print(f"  [{cname}] accuracy {np.mean([r['accuracy'] for r in rows]):.3f} ({np.std([r['accuracy'] for r in rows], ddof=1):.3f}); macro AUC {np.mean([r['auc_macro_ovr'] for r in rows]):.3f}; "
          f"mean precision {np.mean([r['precision'] for r in rows], 0).round(2).tolist()}, recall {np.mean([r['recall'] for r in rows], 0).round(2).tolist()}", flush=True)
json.dump(results, open(OUT + "csv_retrain.json", "w"), indent=1, default=float)
pd.concat(preds).to_parquet(OUT + "csv_retrain_test_preds.parquet")
print("DONE")
