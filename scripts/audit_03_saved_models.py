"""Audit step 3: reproduce the released random splits, verify them against the released test files, and evaluate the SAVED models.
python -I scripts/audit_03_saved_models.py"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score

from src.features.paper_splits import paper_random_split
from src.models import safe_pickle as sp
from src.models.stability_nets import SolventNet, ThermalNet, predict

R = "data/raw/optimof/"
OUT = "data/processed/audit/"
res = {}
load = lambda p: sp.safe_joblib_load(R + p)
split = {"thermal": load("data/thermal/thermal_split_data.pkl"), "solvent": load("data/solvent/solvent_split_data.pkl"), "water": load("data/water_and_haz/water_split_data.pkl")}
scal = {"thermal": load("model/preprocess/thermal_scalar.pkl"), "solvent": load("model/preprocess/solvent_scalar.pkl")}
stubs = {("__main__", "ThermalModel"): ThermalNet, ("__main__", "SolventModel"): SolventNet}
nets = {k: sp.safe_torch_load(R + p, stubs=stubs) for k, p in (("thermal", "model/thermal_model.pkl"), ("thermal_saved", "model/saved_models/thermal_model_52mae.pkl"), ("solvent", "model/solvent_model.pkl"), ("solvent_saved", "model/saved_models/solvent_model_78acc.pkl"))}
rf_obj = sp.safe_joblib_load(R + "model/water_rf_model.pkl", stubs={("__main__", "WaterStabilityRF"): sp.Stub})
rf_obj2 = sp.safe_joblib_load(R + "model/saved_models/water_rf_model_66acc.pkl", stubs={("__main__", "WaterStabilityRF"): sp.Stub})

print("== 1. two copies of each saved network: same weights?")
for k in ("thermal", "solvent"):
    a, b = nets[k].state_dict(), nets[k + "_saved"].state_dict()
    same = all(torch.equal(a[x], b[x]) for x in a)
    print(f"  {k}: model/{k}_model.pkl vs saved_models copy: weights identical = {same}; parameters {sum(p.numel() for p in nets[k].parameters()):,}; layer sizes {[l.out_features for l in nets[k].layers]}")
    res[f"{k}_copies_identical"] = bool(same)
print("  water RF: model/ vs saved_models/ copy identical bytes (checked earlier): True")

print("\n== 2. reproduce the splits (train_test_split(test_size=0.2, random_state=0)) and compare with the released test files")
for n in ("thermal", "solvent"):
    d = split[n]
    Xtr, Xte, ytr, yte = paper_random_split(d, n)
    sc = scal[n]
    saved = pd.read_pickle if False else sp.safe_read_pickle(R + f"data/{n}/{n}_test_data.pkl")
    Xs = sc.transform(Xte); ysv = saved[n].values; yte_cmp = yte.values if n == "thermal" else np.where(yte.values < 0, 0, yte.values)
    feats = saved.drop(columns=[n]).values
    print(f"  {n}: train {len(Xtr)}, test {len(Xte)} (saved test file rows {len(saved)})")
    print(f"     my test labels == saved test labels: {np.array_equal(yte_cmp, ysv)}; my scaled test features vs saved: max |diff| {np.abs(Xs - feats).max():.3g}")
    m_tr, m_all = Xtr.values.mean(0), d.drop(columns=[n]).values.mean(0)
    print(f"     saved scaler mean_ matches TRAIN-only mean: max |diff| {np.abs(sc.mean_ - m_tr).max():.3g}; matches ALL-data mean: max |diff| {np.abs(sc.mean_ - m_all).max():.3g}  (-> scaler fitted on training part only: {np.abs(sc.mean_ - m_tr).max() < 1e-6})")
    res[f"{n}_split_reproduced"] = bool(np.array_equal(yte_cmp, ysv) and np.abs(Xs - feats).max() < 1e-6)
    res[f"{n}_scaler_train_only"] = bool(np.abs(sc.mean_ - m_tr).max() < 1e-6)
# water: the pickled RF object carries its own train/test frames
dw = split["water"]
Xtr, Xte, ytr, yte = paper_random_split(dw.assign(water=dw["water"] - 1), "water")
print(f"  water: train {len(Xtr)}, test {len(Xte)}")
tst_idx_saved = list(rf_obj.test_features.index)
print(f"     saved RF object's own test MOFs == my reproduced test MOFs (same order): {tst_idx_saved == list(Xte.index)}; same set: {set(tst_idx_saved) == set(Xte.index)}")
print(f"     saved RF object's own train MOFs == my train MOFs: {set(rf_obj.train_features.index) == set(Xtr.index)}")
res["water_split_reproduced"] = set(tst_idx_saved) == set(Xte.index)
vc = yte.value_counts().sort_index().to_dict()
print(f"     water test class counts (0-based): {vc}; repo docstring (25, 65, 115, 14); SI text (17, 68, 112, 22)")
res["water_test_class_counts"] = {int(k): int(v) for k, v in vc.items()}
wt = sp.safe_read_pickle(R + "data/water_and_haz/water_rf_test_data.pkl")
print(f"     released water_rf_test_data.pkl shape {wt.shape} (a concat of labels and features along axis 0 -> not usable as a test table); columns {wt.shape[1]}")

print("\n== 3. saved-model performance on the reproduced test sets")
# thermal
Xtr, Xte, ytr, yte = paper_random_split(split["thermal"], "thermal"); p = predict(nets["thermal"], scal["thermal"].transform(Xte))
mae, rmse, r2 = mean_absolute_error(yte, p), mean_squared_error(yte, p) ** 0.5, r2_score(yte, p)
print(f"  thermal (n={len(yte)}): MAE {mae:.2f}, RMSE {rmse:.2f}, R2 {r2:.3f}   | SI: MAE 47, RMSE 67, R2 0.41 ; saved filename: 52mae")
pd.DataFrame({"name": Xte.index, "truth": yte.values, "pred": p}).to_parquet(OUT + "thermal_saved_test_predictions.parquet")
res["thermal"] = dict(n=int(len(yte)), mae=mae, rmse=rmse, r2=r2)
# solvent
Xtr, Xte, ytr, yte = paper_random_split(split["solvent"], "solvent"); yb = (yte.values > 0).astype(int)
logit = predict(nets["solvent"], scal["solvent"].transform(Xte)); pr = 1 / (1 + np.exp(-logit)); hat = (pr >= 0.5).astype(int)
tn, fp, fn, tp = confusion_matrix(yb, hat).ravel()
print(f"  solvent (n={len(yb)}): accuracy {accuracy_score(yb, hat):.4f}, precision {precision_score(yb, hat):.4f}, recall {recall_score(yb, hat):.4f}, F1 {f1_score(yb, hat):.4f}, AUC {roc_auc_score(yb, pr):.4f}; TN/FP/FN/TP = {tn}/{fp}/{fn}/{tp}")
print(f"     SI: accuracy 0.775, precision 0.817, recall 0.829, F1 0.823, AUC 0.826; TN/FP/FN/TP = 110/51/47/228 ; saved filename: 78acc")
pd.DataFrame({"name": Xte.index, "truth": yb, "p_stable": pr}).to_parquet(OUT + "solvent_saved_test_predictions.parquet")
res["solvent"] = dict(n=int(len(yb)), acc=accuracy_score(yb, hat), prec=precision_score(yb, hat), rec=recall_score(yb, hat), f1=f1_score(yb, hat), auc=roc_auc_score(yb, pr), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
# water RF (the saved object's own RF applied to the reproduced test set)
Xtr, Xte, ytr, yte = paper_random_split(dw.assign(water=dw["water"] - 1), "water"); rf = rf_obj.model
pw = rf.predict_proba(Xte); hat = rf.classes_[pw.argmax(1)]
acc = accuracy_score(yte, hat); auc = roc_auc_score(yte, pw, multi_class="ovr")
from sklearn.metrics import classification_report
print(f"  water RF (n={len(yte)}): accuracy {acc:.4f}, macro-OVR AUC {auc:.4f}   | repo docstring 0.6621 / 0.8400 ; SI 0.671")
print(classification_report(yte, hat, digits=3, zero_division=0))
print("  confusion matrix (rows=true 0..3, cols=pred 0..3):\n", confusion_matrix(yte, hat))
pd.DataFrame({"name": Xte.index, "truth": yte.values, **{f"p{c}": pw[:, i] for i, c in enumerate(rf.classes_)}}).to_parquet(OUT + "water_saved_test_predictions.parquet")
res["water"] = dict(n=int(len(yte)), acc=acc, auc=auc)

print("\n== 3b. the saved_models/ copies and the TRAIN-set performance of the released networks")
for task, key in (("thermal", "thermal_saved"), ("solvent", "solvent_saved")):
    Xtr, Xte, ytr, yte = paper_random_split(split[task], task)
    for tag, net in ((task, nets[task]), (key, nets[key])):
        o, ot = predict(net, scal[task].transform(Xte)), predict(net, scal[task].transform(Xtr))
        if task == "thermal":
            r = dict(test_mae=mean_absolute_error(yte, o), test_rmse=mean_squared_error(yte, o) ** 0.5, test_r2=r2_score(yte, o), train_mae=mean_absolute_error(ytr, ot), train_r2=r2_score(ytr, ot))
        else:
            yb, ybt = (yte.values > 0).astype(int), (ytr.values > 0).astype(int); p_, pt_ = 1 / (1 + np.exp(-o)), 1 / (1 + np.exp(-ot)); tn_, fp_, fn_, tp_ = confusion_matrix(yb, p_ >= .5).ravel()
            r = dict(test_acc=accuracy_score(yb, p_ >= .5), test_auc=roc_auc_score(yb, p_), tn=int(tn_), fp=int(fp_), fn=int(fn_), tp=int(tp_), train_acc=accuracy_score(ybt, pt_ >= .5))
        res[f"{tag}_full"] = r
        print(f"  {tag}:", {k: round(float(v), 4) for k, v in r.items()})

rfp = rf.get_params()
print("\n== 4. hyperparameters of the saved random forest:", {k: rfp[k] for k in ("n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "bootstrap", "max_features", "random_state")}, "| n features", rf.n_features_in_)
print("   SI Table S5: 250 trees, max depth 20, split threshold 8")
res["rf_params"] = {k: (None if rfp[k] is None else (rfp[k] if isinstance(rfp[k], (int, float, str, bool)) else str(rfp[k]))) for k in ("n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "bootstrap", "max_features")}
json.dump(res, open(OUT + "saved_models_eval.json", "w"), indent=1, default=float)
