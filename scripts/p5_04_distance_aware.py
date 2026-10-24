"""Phase 5, step 3b: distance-aware (Mondrian) recall rule on the SAME cluster-held-out splits and models as p5_03 (same seeds; plain results re-computed and
asserted equal to p5_03). Configuration fixed before looking at any result: statistic = mean distance to the 5 nearest TRAINING MOFs (z-scored features, training
statistics), 3 bins with edges at the calibration terciles, one split-conformal threshold per bin. No tuning. No coverage proof is claimed under cluster shift.
python -I scripts/p5_04_distance_aware.py"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from src.conformal.recall import conformal_threshold, stage_outcome
from src.conformal.shift import cluster_split, embed, knn_distance, mondrian_keep, mondrian_thresholds
from src.features import stability_data as sd

OUT = "data/processed/phase5/"
R, K, ALPHAS, NBINS, KNN = 20, 10, [0.05, 0.1, 0.2, 0.3], 3, 5
prev = json.load(open(OUT + "family_shift.json"))["results"]
res = {}
for t in sd.TASKS:
    d = sd.load_task(t); X = d[sd.feature_columns()[t]].values.astype(float); stable = d["stable"].values.astype(bool); y = d["y"].values
    E = embed(X, 20, 0); rows = []
    for r in range(R):
        sp = cluster_split(E, K, seed=r); tr, ca, te = sp["train"], sp["cal"], sp["test"]
        if t == "thermal":
            m = RandomForestRegressor(n_estimators=300, max_features="sqrt", n_jobs=8, random_state=r).fit(X[tr], d["label_raw"].values[tr]); f = m.predict
        else:
            m = RandomForestClassifier(n_estimators=300, max_features="sqrt", n_jobs=8, random_state=r).fit(X[tr], y[tr])
            f = (lambda M: (lambda Z: M.predict_proba(Z)[:, 1]))(m) if t == "solvent" else (lambda M: (lambda Z: M.predict_proba(Z)[:, 2:].sum(1)))(m)
        s_ca, s_te = f(X[ca]), f(X[te]); sc = StandardScaler().fit(X[tr]); Ztr = sc.transform(X[tr])
        d_ca, d_te = knn_distance(Ztr, sc.transform(X[ca]), KNN), knn_distance(Ztr, sc.transform(X[te]), KNN)
        rec = dict(rep=r, frac_test_beyond_cal_max_dist=float((d_te > d_ca.max()).mean()), median_dist_cal=float(np.median(d_ca)), median_dist_test=float(np.median(d_te)))
        for a in ALPHAS:
            o = stage_outcome(s_te, stable[te], conformal_threshold(s_ca[stable[ca]], a))
            assert abs(o["recall"] - prev[t][r][f"recall_{a}"]) < 1e-9, "plain result differs from p5_03"
            mo = mondrian_thresholds(s_ca, stable[ca], d_ca, a, NBINS); keep = mondrian_keep(mo, s_te, d_te)
            rec[f"plain_recall_{a}"], rec[f"plain_kept_{a}"] = o["recall"], o["kept_fraction"]
            rec[f"mond_recall_{a}"] = float((keep & stable[te]).sum() / stable[te].sum()); rec[f"mond_kept_{a}"] = float(keep.mean())
        rows.append(rec)
    res[t] = rows; df = pd.DataFrame(rows)
    print(f"\n{t}: median kNN distance to train: cal {df.median_dist_cal.mean():.2f}, test {df.median_dist_test.mean():.2f}; share of test MOFs farther than every calibration MOF: {df.frac_test_beyond_cal_max_dist.mean():.3f}", flush=True)
    for a in ALPHAS:
        p, q = df[f"plain_recall_{a}"], df[f"mond_recall_{a}"]
        print(f"  target {1 - a:.2f}: plain recall {p.mean():.3f} (sd {p.std():.3f}, min {p.min():.3f}, share>=target {np.mean(p >= 1 - a):.2f}, kept {df[f'plain_kept_{a}'].mean():.3f}) | distance-aware recall {q.mean():.3f} (sd {q.std():.3f}, min {q.min():.3f}, share>=target {np.mean(q >= 1 - a):.2f}, kept {df[f'mond_kept_{a}'].mean():.3f}) | paired diff in recall {np.mean(q - p):+.3f}", flush=True)
json.dump(dict(R=R, K=K, n_bins=NBINS, knn=KNN, alphas=ALPHAS, results=res), open(OUT + "distance_aware.json", "w"), indent=1, default=float)
