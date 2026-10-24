"""Phase 5, step 3: family-shift validation. MOFs of a stage are clustered in feature space (unsupervised: z-score -> PCA 20 -> k-means, K = 10); whole clusters
go to test (20% of MOFs), other whole clusters to calibration (20%), the rest to training. A random-forest stage model is trained on the training clusters,
the per-stage split-conformal recall rule is calibrated on the calibration clusters and its recall measured on the test clusters. Repeated R = 20 times
(k-means seed and cluster order change). Control: the same models with calibration and test both taken from the TEST clusters (random halves; in-cluster
exchangeable). Models are fixed in advance (RF, 300 trees); nothing is tuned on any test split. python -I scripts/p5_03_family_shift.py"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.conformal.recall import conformal_threshold, stage_outcome
from src.conformal.shift import cluster_split, embed, knn_distance
from src.features import stability_data as sd

OUT = "data/processed/phase5/"
R, K, ALPHAS = 20, 10, [0.05, 0.1, 0.2, 0.3]
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
        s_ca, s_te = f(X[ca]), f(X[te]); rec = dict(rep=r, n_train=len(tr), n_cal=len(ca), n_test=len(te), n_cal_pos=int(stable[ca].sum()), n_test_pos=int(stable[te].sum()), auc_test=roc_auc_score(stable[te], s_te) if 0 < stable[te].sum() < len(te) else float("nan"),
                                                       prev_cal=float(stable[ca].mean()), prev_test=float(stable[te].mean()), test_clusters=sorted(set(sp["cluster"][te].tolist())), cal_clusters=sorted(set(sp["cluster"][ca].tolist())))
        # control: halves of the test clusters
        rng = np.random.RandomState(1000 + r); perm = rng.permutation(len(te)); h1, h2 = perm[: len(te) // 2], perm[len(te) // 2:]
        for a in ALPHAS:
            o = stage_outcome(s_te, stable[te], conformal_threshold(s_ca[stable[ca]], a)); rec[f"recall_{a}"] = o["recall"]; rec[f"kept_{a}"] = o["kept_fraction"]
            oc = stage_outcome(s_te[h2], stable[te][h2], conformal_threshold(s_te[h1][stable[te][h1]], a)); rec[f"ctrl_recall_{a}"] = oc["recall"]; rec[f"ctrl_kept_{a}"] = oc["kept_fraction"]
        Z = StandardScaler().fit(X[tr]); rec["dist_cal"] = float(knn_distance(Z.transform(X[tr]), Z.transform(X[ca])).mean()); rec["dist_test"] = float(knn_distance(Z.transform(X[tr]), Z.transform(X[te])).mean())
        rows.append(rec)
    res[t] = rows; df = pd.DataFrame(rows)
    print(f"\n{t}: R={R}, K={K}; test MOFs {df.n_test.mean():.0f} (stable {df.n_test_pos.mean():.0f}), cal positives {df.n_cal_pos.mean():.0f}; test AUC {df.auc_test.mean():.3f} (sd {df.auc_test.std():.3f}); mean kNN distance to train: cal {df.dist_cal.mean():.2f}, test {df.dist_test.mean():.2f}", flush=True)
    for a in ALPHAS:
        g = df[f"recall_{a}"]; c = df[f"ctrl_recall_{a}"]
        print(f"  target {1 - a:.2f}: cluster-held-out recall mean {g.mean():.3f} (sd {g.std():.3f}, min {g.min():.3f}, share of reps >= target {np.mean(g >= 1 - a):.2f}), kept {df[f'kept_{a}'].mean():.3f} | same-cluster control recall {c.mean():.3f} (share >= target {np.mean(c >= 1 - a):.2f})", flush=True)
json.dump(dict(R=R, K=K, alphas=ALPHAS, results=res), open(OUT + "family_shift.json", "w"), indent=1, default=float)
