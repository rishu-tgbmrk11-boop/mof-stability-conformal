"""Audit step 5: train/test leakage checks, calibration and false-negative rates of the SAVED (released) models.
python -I scripts/audit_05_leakage_calibration.py   (needs audit_03 to have produced the *_saved_test_predictions.parquet files)"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.features.paper_splits import paper_random_split
from src.models import safe_pickle as sp

R = "data/raw/optimof/"
OUT = "data/processed/audit/"
L = lambda p: sp.safe_joblib_load(R + p)
res = {}
split = {"thermal": L("data/thermal/thermal_split_data.pkl"), "solvent": L("data/solvent/solvent_split_data.pkl"), "water": L("data/water_and_haz/water_split_data.pkl")}
doi = pd.read_csv(R + "data/solvent/solvent_all_data.csv").set_index("refcode")["doi"]
fam = lambda idx, k=4: pd.Series([str(i)[:k] for i in idx], index=idx)

print("=" * 100, "\nA. LEAKAGE CHECKS (released random 80/20 split)\n" + "=" * 100)
splits = {}
for task, d in split.items():
    dd = d.assign(water=d["water"] - 1) if task == "water" else d
    Xtr, Xte, ytr, yte = paper_random_split(dd, task)
    splits[task] = (Xtr, Xte, ytr, yte)
    ftr, fte = set(fam(Xtr.index)), fam(Xte.index)
    share4 = fte.isin(ftr).mean()
    ftr5, fte5 = set(fam(Xtr.index, 5)), fam(Xte.index, 5)
    line = f"{task}: test MOFs sharing a 4-letter refcode family with >=1 training MOF: {share4:.1%}; 5-letter family: {fte5.isin(ftr5).mean():.1%}"
    if task == "solvent":
        dtr = set(doi.loc[Xtr.index]); line += f"; sharing a DOI with a training MOF: {doi.loc[Xte.index].isin(dtr).mean():.1%}"
        res["solvent_share_doi"] = float(doi.loc[Xte.index].isin(dtr).mean())
    print(line); res[f"{task}_share_family4"] = float(share4); res[f"{task}_share_family5"] = float(fte5.isin(ftr5).mean())

print("\n-- feature-space nearest neighbours: test MOF -> closest training MOF (standardised features, Euclidean)")
for task, (Xtr, Xte, ytr, yte) in splits.items():
    sc = StandardScaler().fit(Xtr); A, B = sc.transform(Xtr), sc.transform(Xte)
    nn = NearestNeighbors(n_neighbors=60).fit(A); dist, ind = nn.kneighbors(B)
    ftr = fam(Xtr.index).values; fte = fam(Xte.index).values
    d1 = dist[:, 0]
    d_other = np.array([next((dist[i, j] for j in range(60) if ftr[ind[i, j]] != fte[i]), np.nan) for i in range(len(B))])
    same_fam_nn = np.mean([ftr[ind[i, 0]] == fte[i] for i in range(len(B))])
    print(f"{task}: nearest-train distance median {np.median(d1):.2f} | nearest train MOF from a DIFFERENT 4-letter family: median {np.nanmedian(d_other):.2f} ({np.nanmedian(d_other) / np.median(d1):.2f}x farther) | the single closest training MOF is in the same family for {same_fam_nn:.1%} of test MOFs")
    res[f"{task}_nn"] = dict(median_nn=float(np.median(d1)), median_nn_other_family=float(np.nanmedian(d_other)), closest_same_family=float(same_fam_nn))
    res[f"{task}_nn_dist"] = d1.tolist()

print("\n-- does the saved model do better on test MOFs that have a close training neighbour? (error by nearest-training-distance tertile)")
pred = {t: pd.read_parquet(OUT + f"{t}_saved_test_predictions.parquet").set_index("name") for t in ("thermal", "solvent", "water")}
for task in ("thermal", "solvent", "water"):
    Xtr, Xte, ytr, yte = splits[task]; d1 = np.array(res[f"{task}_nn_dist"]); pr = pred[task].loc[Xte.index]
    q = pd.qcut(d1, 3, labels=["closest third", "middle third", "farthest third"])
    if task == "thermal":
        t = pd.DataFrame({"q": q, "abs_err": np.abs(pr.truth.values - pr.pred.values)}).groupby("q", observed=True).abs_err.agg(["mean", "count"]); print("thermal MAE (deg C):\n" + t.round(2).to_string())
    elif task == "solvent":
        t = pd.DataFrame({"q": q, "ok": ((pr.p_stable.values >= .5).astype(int) == pr.truth.values)}).groupby("q", observed=True).ok.agg(["mean", "count"]); print("solvent accuracy:\n" + t.round(3).to_string())
    else:
        hat = pr[[c for c in pr.columns if c.startswith("p")]].values.argmax(1)
        t = pd.DataFrame({"q": q, "ok": hat == pr.truth.values}).groupby("q", observed=True).ok.agg(["mean", "count"]); print("water accuracy:\n" + t.round(3).to_string())
    res[f"{task}_by_nn_tertile"] = {str(k): float(v) for k, v in t["mean"].items()}

print("\n-- random vs group-aware 5-fold CV with the same fast model (random forest), all labelled MOFs of each table")
def cv(task, groups):
    d = split[task]; X = d.drop(columns=[task]).values; y = d[task].values
    if task == "solvent": y = (y > 0).astype(int)
    if task == "water": y = (y - 1).astype(int)
    out = {}
    for name, splitter in (("random KFold", KFold(5, shuffle=True, random_state=0)), ) + ((("group by " + groups[0], GroupKFold(5)),) if groups else ()):
        sc = []
        for tr, te in (splitter.split(X, y, groups[1]) if (groups and name != "random KFold") else splitter.split(X, y)):
            if task == "thermal":
                m = RandomForestRegressor(300, random_state=0, n_jobs=-1).fit(X[tr], y[tr]); p = m.predict(X[te]); sc.append((r2_score(y[te], p), mean_absolute_error(y[te], p)))
            else:
                m = RandomForestClassifier(300, random_state=0, n_jobs=-1).fit(X[tr], y[tr]); sc.append((accuracy_score(y[te], m.predict(X[te])), np.nan))
        out[name] = tuple(np.nanmean(np.array(sc), axis=0))
    return out
for task in ("thermal", "solvent", "water"):
    d = split[task]; g4 = fam(d.index).values
    r = cv(task, ("4-letter refcode family", g4)); metric = "R2 / MAE" if task == "thermal" else "accuracy"
    for k, v in r.items(): print(f"{task:8s} {k:34s} {metric}: {v[0]:.3f}" + (f" / {v[1]:.1f}" if task == "thermal" else ""))
    res[f"cv_{task}"] = {k: list(map(float, v)) for k, v in r.items()}
dg = cv("solvent", ("DOI", doi.loc[split["solvent"].index].values))
for k, v in dg.items():
    if k.startswith("group"): print(f"solvent  {k:34s} accuracy: {v[0]:.3f}")
res["cv_solvent_doi"] = {k: list(map(float, v)) for k, v in dg.items()}

print("\n-- which MOFs are in the TEST set of one task but the TRAIN set of another (matters for any stacked evaluation)")
names = {t: (set(splits[t][0].index), set(splits[t][1].index)) for t in splits}
for a in ("thermal", "solvent", "water"):
    for b in ("thermal", "solvent", "water"):
        if a < b:
            print(f"  {a} test & {b} train: {len(names[a][1] & names[b][0])}   |   {b} test & {a} train: {len(names[b][1] & names[a][0])}   |   test in both: {len(names[a][1] & names[b][1])}")
allthree = set(split["thermal"].index) & set(split["solvent"].index) & set(split["water"].index)
tst_all = {t: names[t][1] for t in splits}
print(f"  MOFs labelled in all three tables: {len(allthree)}; of these in the TEST set of: thermal {len(allthree & tst_all['thermal'])}, solvent {len(allthree & tst_all['solvent'])}, water {len(allthree & tst_all['water'])}; in the test set of ALL THREE: {len(allthree & tst_all['thermal'] & tst_all['solvent'] & tst_all['water'])}")
res["allthree"] = dict(n=len(allthree), test_thermal=len(allthree & tst_all['thermal']), test_solvent=len(allthree & tst_all['solvent']), test_water=len(allthree & tst_all['water']), test_all3=len(allthree & tst_all['thermal'] & tst_all['solvent'] & tst_all['water']))

print("\n" + "=" * 100 + "\nB. CALIBRATION AND FALSE-NEGATIVE RATES of the released models on their own test sets\n" + "=" * 100)
def reliability(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1); idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1); rows = []
    for b in range(bins):
        m = idx == b
        if m.any(): rows.append((f"{edges[b]:.1f}-{edges[b + 1]:.1f}", int(m.sum()), float(p[m].mean()), float(y[m].mean())))
    ece = sum(n * abs(pm - ym) for _, n, pm, ym in rows) / len(p)
    return rows, ece
def fn_table(score, truth, thresholds, label):
    print(f"  {label}: truly stable {int(truth.sum())} of {len(truth)}")
    out = []
    for t in thresholds:
        pas = score >= t; tp = int((pas & (truth == 1)).sum()); fn = int((~pas & (truth == 1)).sum()); fp = int((pas & (truth == 0)).sum())
        out.append(dict(threshold=t, recall=tp / max(1, tp + fn), fnr=fn / max(1, tp + fn), precision=tp / max(1, tp + fp), pass_fraction=float(pas.mean())))
        print(f"    pass if score >= {t:g}:  recall {out[-1]['recall']:.3f}  FNR {out[-1]['fnr']:.3f}  precision {out[-1]['precision']:.3f}  MOFs passing {pas.mean():.1%}")
    # threshold giving >=95 % recall
    order = np.sort(score[truth == 1]); k = int(np.floor(0.05 * len(order))); t95 = order[k]
    pas = score >= t95; print(f"    threshold for >=95% recall on this test set: {t95:.3f} -> recall {(pas & (truth == 1)).sum() / truth.sum():.3f}, MOFs passing {pas.mean():.1%}, precision {(pas & (truth == 1)).sum() / pas.sum():.3f}")
    return out, float(t95)
# solvent
ps = pred["solvent"]; p, y = ps.p_stable.values, ps.truth.values; rows, ece = reliability(p, y)
print(f"\nSOLVENT (binary, n={len(y)}): Brier {brier_score_loss(y, p):.4f}, log-loss {log_loss(y, p):.4f}, ECE(10 bins) {ece:.4f}, AUC {roc_auc_score(y, p):.4f}")
print("  reliability (bin, n, mean predicted p, observed fraction stable):"); [print(f"    {b:9s} n={n:3d}  p={pm:.3f}  observed={ym:.3f}") for b, n, pm, ym in rows]
out, t95 = fn_table(p, y, [0.5, 0.7, 0.9], "paper screening threshold is > 0.9"); res["solvent_cal"] = dict(brier=brier_score_loss(y, p), logloss=log_loss(y, p), ece=ece, rel=rows, fn=out, t95=t95)
# water: stable = classes 2,3
pw = pred["water"]; pc = pw[["p0.0", "p1.0", "p2.0", "p3.0"]].values; ys = (pw.truth.values >= 2).astype(int); pst = pc[:, 2] + pc[:, 3]; rows, ece = reliability(pst, ys)
print(f"\nWATER (stable = class 2 or 3, n={len(ys)}): Brier {brier_score_loss(ys, pst):.4f}, ECE(10 bins) {ece:.4f}, AUC {roc_auc_score(ys, pst):.4f}")
print("  reliability of P(stable) = P(class 2)+P(class 3):"); [print(f"    {b:9s} n={n:3d}  p={pm:.3f}  observed={ym:.3f}") for b, n, pm, ym in rows]
hat = pc.argmax(1); pas = hat >= 2; tp = int((pas & (ys == 1)).sum()); fn = int((~pas & (ys == 1)).sum()); fp = int((pas & (ys == 0)).sum())
print(f"  paper rule (predicted class in {{2,3}}): recall {tp / (tp + fn):.3f}, FNR {fn / (tp + fn):.3f}, precision {tp / (tp + fp):.3f}, MOFs passing {pas.mean():.1%}")
out, t95 = fn_table(pst, ys, [0.3, 0.5, 0.7], "P(stable) thresholds"); res["water_cal"] = dict(brier=brier_score_loss(ys, pst), ece=ece, rel=rows, paper_rule=dict(recall=tp / (tp + fn), fnr=fn / (tp + fn), precision=tp / (tp + fp)), fn=out, t95=t95)
# thermal: stable = >= 350 C
pt = pred["thermal"]; yt = (pt.truth.values >= 350).astype(int); sc_ = pt.pred.values
sl, ic = np.polyfit(sc_, pt.truth.values, 1)
print(f"\nTHERMAL (stable = decomposition T >= 350 C, n={len(yt)}): calibration of the regression: truth = {sl:.3f} * prediction + {ic:.1f}  (a calibrated model has slope 1, intercept 0); mean residual {np.mean(pt.truth.values - sc_):+.2f} C")
dec = pd.qcut(sc_, 10, duplicates="drop"); g = pd.DataFrame({"bin": dec, "pred": sc_, "truth": pt.truth.values, "stable": yt}).groupby("bin", observed=True).agg(n=("pred", "size"), mean_pred=("pred", "mean"), mean_truth=("truth", "mean"), frac_stable=("stable", "mean"))
print("  by prediction decile:\n" + g.round(2).to_string())
out, t95 = fn_table(sc_, yt, [300, 330, 350], "predicted T thresholds (deg C); paper screens at >= 350"); res["thermal_cal"] = dict(slope=float(sl), intercept=float(ic), fn=out, t95=t95)
json.dump(res, open(OUT + "leakage_calibration.json", "w"), indent=1, default=float)
