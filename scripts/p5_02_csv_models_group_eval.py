"""Phase 5, step 1b: group-aware evaluation of the SAME CSV-split models (p5_01). Each test MOF is 'group-clean' if no MOF of its proxy group (refcode family + DOI,
`phase2/groups.parquet`) is in that stage's training part, otherwise 'leaky'. Metrics per training seed, averaged over the 5 seeds; 95% percentile interval from
1,000 bootstrap resamples of test GROUPS (seed 0), each resample re-computing the metric per seed and averaging. python -I scripts/p5_02_csv_models_group_eval.py"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.features import stability_data as sd

OUT = "data/processed/phase5/"
P = pd.read_parquet(OUT + "csv_retrain_test_preds.parquet")
g = pd.read_parquet("data/processed/phase2/groups.parquet").set_index("name")["group_id"]
NB = 1000
res = {}


def met(stage, d):
    if stage == "thermal":
        return dict(r2=r2_score(d.y, d.score), mae=mean_absolute_error(d.y, d.score), auc_stable=roc_auc_score(d.y >= 350, d.score) if 0 < (d.y >= 350).sum() < len(d) else np.nan)
    if stage == "solvent":
        return dict(accuracy=accuracy_score(d.y, d.score >= .5), auc=roc_auc_score(d.y, d.score) if 0 < d.y.sum() < len(d) else np.nan)
    st = (d.y >= 2).astype(int)
    return dict(accuracy_4class=accuracy_score(d.y, d.pred_class), auc_stable=roc_auc_score(st, d.score) if 0 < st.sum() < len(d) else np.nan)


for stage in sd.TASKS:
    d = sd.load_task(stage); tr, te = train_test_split(np.arange(len(d)), test_size=0.2, random_state=0)
    train_groups = set(g.loc[d.index[tr]])
    clean_names = {n for n in d.index[te] if g[n] not in train_groups}
    q = P[P.stage == stage].copy(); q["group_id"] = g.loc[q.name].values; q["clean"] = q.name.isin(clean_names)
    nn_te = len(te); n_clean = len(clean_names)
    print(f"\n##### {stage}: test {nn_te} MOFs; group-clean {n_clean} MOFs in {q[q.clean].group_id.nunique()} groups; leaky {nn_te - n_clean}", flush=True)
    res[stage] = dict(n_test=nn_te, n_clean=n_clean, variants={})
    for var, vq in q.groupby("variant"):
        out = {}
        for sub, name in ((None, "all"), (True, "clean"), (False, "leaky")):
            s = vq if sub is None else vq[vq.clean == sub]
            out[name] = pd.DataFrame([met(stage, x) for _, x in s.groupby("seed")]).mean().to_dict()
        # bootstrap over groups for the clean subset and for the clean-minus-leaky difference
        rng = np.random.RandomState(0); sg = vq.groupby("group_id"); gids = np.array(sorted(vq.group_id.unique())); by = {k: v for k, v in sg}
        boot_clean, boot_diff = [], []
        for _ in range(NB):
            pick = rng.choice(gids, len(gids)); s = pd.concat([by[k] for k in pick]); sc_, sl_ = s[s.clean], s[~s.clean]
            if sc_.empty or sl_.empty: continue
            mc = pd.DataFrame([met(stage, x) for _, x in sc_.groupby("seed")]).mean(); ml = pd.DataFrame([met(stage, x) for _, x in sl_.groupby("seed")]).mean()
            boot_clean.append(mc); boot_diff.append(mc - ml)
        ci = lambda b: {k: [float(np.nanpercentile([x[k] for x in b], 2.5)), float(np.nanpercentile([x[k] for x in b], 97.5))] for k in b[0].index}
        out["clean_ci"], out["clean_minus_leaky_ci"] = ci(boot_clean), ci(boot_diff)
        res[stage]["variants"][var] = out
        key = [k for k in out["all"]]
        print(f"  [{var}]  " + " | ".join(f"{k}: all {out['all'][k]:.3f}, leaky {out['leaky'][k]:.3f}, clean {out['clean'][k]:.3f} [{out['clean_ci'][k][0]:.3f}, {out['clean_ci'][k][1]:.3f}]" for k in key), flush=True)
json.dump(res, open(OUT + "csv_models_group_eval.json", "w"), indent=1, default=float)
