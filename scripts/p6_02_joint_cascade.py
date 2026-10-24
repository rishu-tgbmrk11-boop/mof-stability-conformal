"""Phase 6 (2): joint cascade score calibrated on the all-three-stable MOFs by jackknife+-style cross-conformal (CV+) vs the union-bound version, evaluated by nested
group cross-fitting. Fixed in advance (nothing tuned on test MOFs): stage models = random forests (100 trees; thermal regressor on temperature, solvent / water classifiers,
score = predicted temperature / P(stable) / P(class 2)+P(class 3)); outer = 10 random group folds of ALL MOFs (test = the labelled-in-all-three MOFs of the fold, which are
excluded from every model and from calibration); inner = one fold per calibration positive's group (so each all-stable MOF is scored by a model trained without it and without
its group), the remaining groups are spread at random over the same folds. 3 repetitions (outer seeds 0, 1, 2; forests seeded 1000 * seed + inner fold).
For each test MOF the script stores the CV+ p-values (joint score, and the three per-stage scores for the union-bound version); thresholds alpha are applied in the notes script.
python -I scripts/p6_02_joint_cascade.py   (repo root; writes joint_cascade_group_pvalues.parquet, which also holds the group-level p-values; its p_joint equals the earlier joint_cascade_pvalues.parquet)"""
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.conformal.joint import cvplus_pvalue, rank_normalise
from src.features import stability_data as sd

OUT = "data/processed/phase6/"
REPS, K_OUT, NTREE = [0, 1, 2], 10, 100
groups = pd.read_parquet("data/processed/phase2/groups.parquet").set_index("name")["group_id"]
data = {t: sd.load_task(t) for t in sd.TASKS}
X = {t: sd.feature_matrix(data[t], t) for t in sd.TASKS}
L = sorted(set(data["thermal"].index) & set(data["solvent"].index) & set(data["water"].index))
all_stable = {n: bool(all(data[t].loc[n, "stable"] == 1 for t in sd.TASKS)) for n in L}
print(f"labelled in all three: {len(L)}; all-stable: {sum(all_stable.values())}; groups of the all-stable MOFs: {groups.loc[[n for n in L if all_stable[n]]].nunique()}", flush=True)
gid = {t: groups.loc[data[t].index].values for t in sd.TASKS}
rows = []


def fit_predict(t, tr, preds_idx_list, seed):
    """Train the stage-t forest on rows `tr` and return scores for each index array in preds_idx_list."""
    d = data[t]
    if t == "thermal":
        m = RandomForestRegressor(n_estimators=NTREE, max_features="sqrt", n_jobs=8, random_state=seed).fit(X[t][tr], d["label_raw"].values[tr])
        f = m.predict
    else:
        m = RandomForestClassifier(n_estimators=NTREE, max_features="sqrt", n_jobs=8, random_state=seed).fit(X[t][tr], d["y"].values[tr])
        if t == "solvent":
            f = lambda Z: m.predict_proba(Z)[:, list(m.classes_).index(1)]
        else:
            f = lambda Z: m.predict_proba(Z)[:, [list(m.classes_).index(c) for c in (2, 3) if c in m.classes_]].sum(1)
    return [f(X[t][i]) for i in preds_idx_list]


for rep in REPS:
    t0 = time.time(); rng = np.random.RandomState(rep)
    gids = np.array(sorted(groups.unique())); perm = rng.permutation(gids); out_fold = dict(zip(perm, np.arange(len(perm)) % K_OUT))
    for fo in range(K_OUT):
        test_names = [n for n in L if out_fold[groups[n]] == fo]
        if not test_names:
            continue
        in_D = {t: np.array([out_fold[g] != fo for g in gid[t]]) for t in sd.TASKS}
        pos_names = [n for n in L if all_stable[n] and out_fold[groups[n]] != fo]
        pos_groups = sorted(set(groups.loc[pos_names])); F = len(pos_groups); nP = len(pos_names)
        r2 = np.random.RandomState(10_000 * rep + fo)
        fold_of_group = {g: i for i, g in enumerate(pos_groups)}
        for g in gids:
            if out_fold[g] != fo and g not in fold_of_group:
                fold_of_group[g] = int(r2.randint(F))
        fold = {t: np.array([fold_of_group.get(g, -1) for g in gid[t]]) for t in sd.TASKS}
        S_test = {t: np.zeros((len(test_names), F)) for t in sd.TASKS}; oof = {t: np.full(len(data[t]), np.nan) for t in sd.TASKS}
        tidx = {t: np.array([data[t].index.get_loc(n) for n in test_names]) for t in sd.TASKS}
        for j in range(F):
            for t in sd.TASKS:
                tr = np.where(in_D[t] & (fold[t] != j))[0]; ev = np.where(in_D[t] & (fold[t] == j))[0]
                s_ev, s_te = fit_predict(t, tr, [ev, tidx[t]], seed=1000 * rep + j)
                oof[t][ev] = s_ev; S_test[t][:, j] = s_te
        U = {t: rank_normalise(oof[t][in_D[t]]) for t in sd.TASKS}
        # joint calibration: the all-stable positives in D, scored by the model that excluded their group
        c_cal, f_cal = [], []
        for n in pos_names:
            c_cal.append(min(float(U[t](oof[t][data[t].index.get_loc(n)])) for t in sd.TASKS)); f_cal.append(fold_of_group[groups[n]])
        c_test = np.min([U[t](S_test[t]) for t in sd.TASKS], axis=0)
        p_joint = cvplus_pvalue(np.array(c_cal), np.array(f_cal), c_test)
        # GROUP-LEVEL calibration (added in the group-level follow-up): one calibration unit per group of all-stable MOFs. The group score is the MIN (primary, conservative: a lower
        # calibration score can only keep more MOFs) or the MEAN (sensitivity) of the joint scores of its positives; each group is its own fold, so n = F groups.
        cc, ff = np.array(c_cal), np.array(f_cal)
        c_gmin = np.array([cc[ff == i].min() for i in range(F)]); c_gmean = np.array([cc[ff == i].mean() for i in range(F)])
        p_gmin = cvplus_pvalue(c_gmin, np.arange(F), c_test); p_gmean = cvplus_pvalue(c_gmean, np.arange(F), c_test)
        # union-bound version: per-stage cross-conformal on that stage's own stable MOFs in D (each scored by the model that excluded its fold)
        p_stage = {}
        for t in sd.TASKS:
            m = in_D[t] & (data[t]["stable"].values == 1)
            p_stage[t] = cvplus_pvalue(oof[t][m], fold[t][m], S_test[t])
        for x, n in enumerate(test_names):
            rows.append(dict(rep=rep, outer_fold=fo, name=n, group_id=int(groups[n]), all_stable=all_stable[n], p_joint=float(p_joint[x]), p_group_min=float(p_gmin[x]), p_group_mean=float(p_gmean[x]), n_cal_groups=F, p_thermal=float(p_stage["thermal"][x]), p_solvent=float(p_stage["solvent"][x]),
                             p_water=float(p_stage["water"][x]), c_joint_test_mean=float(c_test[x].mean()), n_cal_pos=nP, n_inner_folds=F, n_stage_pos={t: int((in_D[t] & (data[t]["stable"].values == 1)).sum()) for t in sd.TASKS}.__repr__()))
    print(f"[rep {rep}] done in {time.time() - t0:.0f}s; test MOFs so far {len(rows)}", flush=True)
    pd.DataFrame(rows).to_parquet(OUT + "joint_cascade_group_pvalues.parquet")
df = pd.DataFrame(rows); df.to_parquet(OUT + "joint_cascade_group_pvalues.parquet")
print("DONE", len(df))
