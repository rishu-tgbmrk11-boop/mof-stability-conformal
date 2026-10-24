"""Phase 6 (3): group-aware metrics under alternative grouping definitions, same protocol as Phase 5 section 5 (random forest 300 trees, one random GROUP split per split seed,
60/20/20 by MOF count over the union of the three tables, test part scored). The MOF-level random baseline and the proxy-group rows are those of
data/processed/phase5/split_ranges_rf.parquet (same seeds), so gaps are paired by split seed. Seeds 0..N-1 (N = 30).
usage: python -I scripts/p6_03c_group_variants.py NAME=path/to/groups.parquet [NAME=path ...]     (parquet columns: name, group_id, group_size)"""
import os
import sys
import time

sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, roc_auc_score

from src.features import stability_data as sd
from src.features.groups import random_group_assign

N = 30
FRAC = {"train": 0.6, "cal": 0.2, "test": 0.2}
OUT = "data/processed/phase6/"
data = {t: sd.load_task(t) for t in sd.TASKS}
auc = lambda y, s: float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")
for arg in sys.argv[1:]:
    name, path = arg.split("=", 1)
    g = pd.read_parquet(path)[["name", "group_id", "group_size"]].set_index("name")
    assert g.index.is_unique and len(g) == 4963
    rows = []; t0 = time.time()
    for s in range(N):
        sp = pd.Series(random_group_assign(g.reset_index(), FRAC, seed=s).values, index=g.index)
        for t in sd.TASKS:
            d = data[t]; lab = sp.loc[d.index].values; tr, te = np.where(lab == "train")[0], np.where(lab == "test")[0]
            X = sd.feature_matrix(d, t); y = d["y"].values; st = d["stable"].values
            if t == "thermal":
                m = RandomForestRegressor(n_estimators=300, max_features="sqrt", n_jobs=8, random_state=s).fit(X[tr], d["label_raw"].values[tr]); p = m.predict(X[te])
                met = dict(auc_stable=auc(st[te], p), mae_degC=mean_absolute_error(d["label_raw"].values[te], p), r2=r2_score(d["label_raw"].values[te], p), accuracy_stable=accuracy_score(st[te], p >= 350))
            else:
                m = RandomForestClassifier(n_estimators=300, max_features="sqrt", n_jobs=8, random_state=s).fit(X[tr], y[tr]); P = m.predict_proba(X[te])
                if t == "solvent":
                    met = dict(auc_stable=auc(st[te], P[:, 1]), accuracy_stable=accuracy_score(st[te], P[:, 1] >= .5))
                else:
                    full = np.zeros((len(te), 4)); full[:, m.classes_.astype(int)] = P; ps = full[:, 2] + full[:, 3]
                    met = dict(auc_stable=auc(st[te], ps), accuracy_stable=accuracy_score(st[te], ps >= .5), accuracy_4class=accuracy_score(y[te], full.argmax(1)))
            rows.append(dict(split_seed=s, mode="group", stage=t, n_test=len(te), n_train=len(tr), grouping=name, **met))
    pd.DataFrame(rows).to_parquet(OUT + f"group_variant_{name}.parquet")
    print(f"{name}: {N} splits done in {time.time() - t0:.0f}s; largest group {int(g.group_size.max())}, groups {g.group_id.nunique()}", flush=True)
print("DONE")
