"""Phase 6 (3): the SI-split models of Phase 5 (p5_01 test predictions, released recipe as coded / fixed restore, released forest) scored on the group-CLEAN part of the CSV test set
under alternative grouping definitions: a test MOF is clean when no MOF of its group (under that grouping) is in that stage's training part. Seed-averaged metric, 95% percentile
bootstrap over test groups (500 resamples, seed 0). usage: python -I scripts/p6_03e_csv_models_alt_groups.py NAME=groups.parquet [...]"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.features import stability_data as sd

P = pd.read_parquet("data/processed/phase5/csv_retrain_test_preds.parquet")
KEEP = {"thermal": ["released|as_coded", "released|fixed"], "solvent": ["released|as_coded", "released|fixed"], "water": ["released|forest"]}
NB = 500


def met(stage, d):
    if stage == "thermal":
        return dict(r2=r2_score(d.y, d.score), mae=mean_absolute_error(d.y, d.score), auc_stable=roc_auc_score(d.y >= 350, d.score) if 0 < (d.y >= 350).sum() < len(d) else np.nan)
    if stage == "solvent":
        return dict(accuracy=accuracy_score(d.y, d.score >= .5), auc=roc_auc_score(d.y, d.score) if 0 < d.y.sum() < len(d) else np.nan)
    st = (d.y >= 2).astype(int)
    return dict(accuracy_4class=accuracy_score(d.y, d.pred_class), auc_stable=roc_auc_score(st, d.score) if 0 < st.sum() < len(d) else np.nan)


res = {}
for arg in sys.argv[1:]:
    name, path = arg.split("=", 1)
    g = pd.read_parquet(path).set_index("name")["group_id"]; res[name] = {}
    for stage in sd.TASKS:
        d = sd.load_task(stage); tr, te = train_test_split(np.arange(len(d)), test_size=0.2, random_state=0)
        train_groups = set(g.loc[d.index[tr]]); clean = {n for n in d.index[te] if g[n] not in train_groups}
        q = P[P.stage == stage].copy(); q["group_id"] = g.loc[q.name].values; q["clean"] = q.name.isin(clean)
        res[name][stage] = dict(n_test=len(te), n_clean=len(clean), variants={})
        for var in KEEP[stage]:
            vq = q[q.variant == var]; mean_m = lambda s: pd.DataFrame([met(stage, x) for _, x in s.groupby("seed")]).mean()
            allm, cl, lk = mean_m(vq), mean_m(vq[vq.clean]), mean_m(vq[~vq.clean])
            rng = np.random.RandomState(0); by = {k: v for k, v in vq.groupby("group_id")}; gids = np.array(sorted(by)); boot = []
            for _ in range(NB):
                s = pd.concat([by[k] for k in rng.choice(gids, len(gids))]); sc = s[s.clean]
                if len(sc): boot.append(mean_m(sc))
            ci = {k: [float(np.nanpercentile([b[k] for b in boot], 2.5)), float(np.nanpercentile([b[k] for b in boot], 97.5))] for k in cl.index}
            res[name][stage]["variants"][var] = dict(all=allm.to_dict(), clean=cl.to_dict(), leaky=lk.to_dict(), clean_ci=ci)
        v0 = res[name][stage]["variants"][KEEP[stage][0]]
        print(f"{name:24s} {stage:8s}: clean {len(clean)}/{len(te)}; " + "; ".join(f"{k} all {v0['all'][k]:.3f} leaky {v0['leaky'][k]:.3f} clean {v0['clean'][k]:.3f} [{v0['clean_ci'][k][0]:.3f}, {v0['clean_ci'][k][1]:.3f}]" for k in v0["all"]), flush=True)
json.dump(res, open("data/processed/phase6/csv_models_alt_groups.json", "w"), indent=1, default=float)
