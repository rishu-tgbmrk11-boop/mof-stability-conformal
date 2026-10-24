"""Phase 6 (1a): evaluate BOTH saved thermal networks (model/thermal_model.pkl, model/saved_models/thermal_model_52mae.pkl; loaded with the restricted unpickler)
on the full CSV random_state=0 test set (the SI's 627 MOFs), and separately on the part of it that was / was not in the network's own training set
(training set = released split table `thermal_split_data.pkl`, same call train_test_split(test_size=0.2, random_state=0) on ITS row order).
python -I scripts/p6_01a_saved_thermal_on_csv_test.py"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.models import safe_pickle as sp
from src.models.stability_nets import ThermalNet, predict

R = "data/raw/optimof/"
OUT = "data/processed/phase6/"
pk = sp.safe_joblib_load(R + "data/thermal/thermal_split_data.pkl")            # released split table: index = refcode, 170 features + 'thermal'
csv = pd.read_csv(R + "data/thermal/thermal_all_data.csv").set_index("refcode")
cols = [c for c in pk.columns if c != "thermal"]
_, te_csv = train_test_split(np.arange(len(csv)), test_size=0.2, random_state=0)
csv_test = csv.index[te_csv]
_, te_pk = train_test_split(np.arange(len(pk)), test_size=0.2, random_state=0)
pk_test_names = set(pk.index[te_pk]); pk_train_names = set(pk.index) - pk_test_names
print(f"CSV test set: {len(csv_test)} MOFs; T sd = {csv.loc[csv_test, 'T'].std(ddof=0):.2f} (ddof=0), {csv.loc[csv_test, 'T'].std():.2f} (ddof=1); implied R2 for RMSE 67: {1 - 67 ** 2 / csv.loc[csv_test, 'T'].var(ddof=0):.3f}")
print(f"released-split test set: {len(pk_test_names)} MOFs; overlap with the CSV test set: {len(set(csv_test) & pk_test_names)} (CSV test MOFs that are in the released split's TRAIN part: {len(set(csv_test) & pk_train_names)})")
fd = np.abs(pk.loc[csv_test, cols].values - csv.loc[csv_test, cols].values).max()
print(f"feature values of the CSV test rows vs the released split table (same refcodes): max |diff| = {fd:.3g}; label equal: {bool(np.allclose(pk.loc[csv_test, 'thermal'].values, csv.loc[csv_test, 'T'].values))}")
sc = sp.safe_joblib_load(R + "model/preprocess/thermal_scalar.pkl")
nets = {k: sp.safe_torch_load(R + p, stubs={("__main__", "ThermalModel"): ThermalNet}) for k, p in (("model/thermal_model.pkl", "model/thermal_model.pkl"), ("saved_models/thermal_model_52mae.pkl", "model/saved_models/thermal_model_52mae.pkl"))}
X = sc.transform(pk.loc[csv_test, cols].values); y = pk.loc[csv_test, "thermal"].values
in_train = np.array([n in pk_train_names for n in csv_test])
Xp = sc.transform(pk.iloc[te_pk][cols].values); yp = pk.iloc[te_pk]["thermal"].values
res = {"n_csv_test": len(y), "n_in_train": int(in_train.sum()), "n_not_in_train": int((~in_train).sum()), "T_sd_ddof0": float(np.std(y)), "models": {}}
print(f"\n{'model':40s} {'subset':28s} {'n':>4s} {'R2':>7s} {'RMSE':>7s} {'MAE':>7s}")
for k, net in nets.items():
    p = predict(net, X, "cpu"); pp = predict(net, Xp, "cpu"); r = {}
    for name, m, yy, pr in (("full CSV test", np.ones(len(y), bool), y, p), ("  in the network's train part", in_train, y, p), ("  NOT in its train part", ~in_train, y, p)):
        mm = m; r[name.strip()] = dict(n=int(mm.sum()), r2=r2_score(yy[mm], pr[mm]), rmse=mean_squared_error(yy[mm], pr[mm]) ** .5, mae=mean_absolute_error(yy[mm], pr[mm]))
        print(f"{k:40s} {name:28s} {mm.sum():4d} {r[name.strip()]['r2']:7.3f} {r[name.strip()]['rmse']:7.2f} {r[name.strip()]['mae']:7.2f}")
    r["released-split test set"] = dict(n=len(yp), r2=r2_score(yp, pp), rmse=mean_squared_error(yp, pp) ** .5, mae=mean_absolute_error(yp, pp))
    print(f"{k:40s} {'released-split test set':28s} {len(yp):4d} {r['released-split test set']['r2']:7.3f} {r['released-split test set']['rmse']:7.2f} {r['released-split test set']['mae']:7.2f}")
    res["models"][k] = r
print("\nSI: R2 0.41, RMSE 67, MAE 47 (n = 627)")
json.dump(res, open(OUT + "saved_thermal_on_csv_test.json", "w"), indent=1, default=float)
