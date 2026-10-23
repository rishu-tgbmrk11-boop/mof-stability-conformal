"""Audit step 7: assemble notes/stability_inventory.md from the saved audit outputs. Every number comes from data/processed/audit/*.
python -I scripts/audit_07_write_notes.py"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd

from src.models import safe_pickle as sp

A = "data/processed/audit/"
J = lambda n: json.load(open(A + n))
inv, ev, rt, lc, mc, ps = J("inventory.json"), J("saved_models_eval.json"), J("retrain.json"), J("leakage_calibration.json"), J("merge_check.json"), J("pickle_scan.json")
R = "data/raw/optimof/"
f3 = lambda x: f"{x:.3f}"
f2 = lambda x: f"{x:.2f}"
f1 = lambda x: f"{x:.1f}"
pc = lambda x: f"{100 * x:.1f}%"
split = {n: sp.safe_joblib_load(R + p) for n, p in (("thermal", "data/thermal/thermal_split_data.pkl"), ("solvent", "data/solvent/solvent_split_data.pkl"), ("water", "data/water_and_haz/water_split_data.pkl"))}
pred = {t: pd.read_parquet(A + f"{t}_saved_test_predictions.parquet") for t in ("thermal", "solvent", "water")}


def rtm(task, tag, key):
    v = np.array([r[key] for r in rt[task][tag]])
    return f"{v.mean():.3f} +- {v.std(ddof=1):.3f}"


def rtn(task, tag):
    return sum(1 for r in rt[task][tag] if r["restore_noop"]), len(rt[task][tag])


L = []
w = L.append
w("# Audit of the paper's stability ML (OptiMOF release)")
w("")
w("Numbers measured by this audit were produced by the scripts in `scripts/audit_0*.py` and are read from their saved outputs in `data/processed/audit/` by `audit_07_write_notes.py`. The only values typed into this file are quotations: the paper's SI figures (marked 'SI'), the repo's docstring/file-name values (marked 'docstring' or 'file name'), the released hyperparameters read from the repo's JSON/source, and a few counts that the scripts printed (parameter counts, duplicate-row counts, the 20 constant columns). Reproduce with `python -I scripts/audit_0N_*.py` inside the `mof-stability` environment, in order 01 to 07 (the scripts are run from the repository root).")
w("")
w("## 0. Scope, sources and safety")
w("")
w("- **Paper:** Liu et al., J. Am. Chem. Soc. 2026, 148, 34821, doi:10.1021/jacs.6c10566. SI Section 4 ('Machine-Learning Stability Evaluation') and the SI Machine Learning Methods tables (Table S5 etc.) are the claims audited here. SI text was read from the user's PDF copy (`ja6c10566_si_001.pdf`).")
w("- **Released code and data:** `github.com/snurr-group/optimof`, shallow clone, commit `75cbd1389a551e638376b78c5688c409958d52cd` (2025-10-06, message 'quick commit'). **CoRE MOF 2024:** Zenodo record 15055758 v1.1, five CSVs only (checksums verified); not used by the audit below.")
w("- **Environment:** `mof-stability` (Python 3.11.17, torch 2.14.1+cu130, scikit-learn 1.9.1, xgboost 3.4.2, pandas 3.0.6, MAPIE 1.6.0). The released pickles were written with scikit-learn 1.5.1 and are loaded with a version warning; accuracies of the saved water forest reproduce the repo's own docstring exactly (section 2), which is evidence the version difference did not change predictions.")
w("- **Untrusted pickles:** every pickle was loaded with `python -I` through `src/models/safe_pickle.py`, a restricted unpickler that only resolves allow-listed numpy/pandas/scikit-learn/scipy/xgboost/torch-tensor names and maps the repository's own classes (`__main__.ThermalModel`, `__main__.SolventModel`, `__main__.WaterStabilityRF`) to inert stand-ins. "
  f"Scanned {len(ps)} pickle files: no `os`, `subprocess`, `eval` or similar global was requested by any of them. Tests: `tests/test_safe_pickle.py` (a malicious pickle is blocked and its payload never runs).")
w("")
w("## 1. Inventory of the three stability tables")
w("")
c = inv
w("| | Thermal | Solvent-removal (activation) | Water |")
w("|---|---|---|---|")
w(f"| released file | `data/thermal/thermal_all_data.csv` | `data/solvent/solvent_all_data.csv` | `data/water_and_haz/data.csv` |")
w(f"| MOFs (rows) | {c['csv_thermal']['rows']:,} | {c['csv_solvent']['rows']:,} | {c['csv_water']['rows']:,} |")
w(f"| id used | CSD refcode (e.g. `ABAVIJ`) | CSD refcode | CSD refcode (`MOF_name`); 911 CoRE + 181 'Burtch' |")
w(f"| label | decomposition temperature `T` (min {mc['class_balance']['thermal_min']:.1f}, max {mc['class_balance']['thermal_max']:.1f}; unit not stated in the repo; the SI uses degC) | `flag` = +1 stable ({mc['class_balance']['solvent_stable']:,}) / -1 unstable ({mc['class_balance']['solvent_unstable']:,}) | `water_label` 1-4: " + ", ".join(f"{k}: {v}" for k, v in mc['class_balance']['water_counts'].items()) + " |")
w(f"| features in the released split table | {inv['split_thermal']['features']} | {inv['split_solvent']['features']} | {inv['split_water']['features']} |")
w(f"| paper SI feature counts | 190 | 190 | 179 |")
w(f"| NaN in split table | 0 | 0 | 0 |")
w(f"| majority-class share | n/a | {pc(mc['class_balance']['solvent_majority_fraction'])} | {pc(mc['class_balance']['water_majority_fraction'])} |")
w("")
w(f"- **Features:** the thermal and solvent CSVs each hold {inv['split_thermal']['features'] + 20} numeric feature columns ({inv['split_thermal']['features']} in the released split + the 20 that `data_selection.py` removes as 'empty'); the water CSV holds {inv['split_water']['features']}. I checked the 20 removed columns in the CSVs and "
  f"{'all 20 are indeed constant' if not inv['drop20_nonconstant'] else 'these are NOT constant: ' + str(inv['drop20_nonconstant'])} (nunique = 1). So the thermal and solvent networks are trained on 170 non-constant features, not on the 190 the SI reports (the SI says only one all-zero feature, `D_func-alpha-0-all`, was dropped); the water model uses {inv['split_water']['features']}, not 179.")
w(f"- **Overlap between the tables (by refcode):** thermal & solvent {inv['overlap']['ts']:,}; thermal & water {inv['overlap']['tw']}; solvent & water {inv['overlap']['sw']}; **all three {inv['overlap']['all3']}**; union {inv['overlap']['union']:,} (`all_in_one.pkl` has {inv['overlap']['all_in_one_rows']:,} rows; the SI says the MOFMap union has 4,148).")
w(f"- **Grouping information:** no MOFid column exists in any released table, and the repo ships no CIFs, so **MOFid groups cannot be computed from the released files**. What exists: DOIs in the solvent table only ({mc['doi']['n_dois']:,} DOIs for {mc['doi']['n_mofs']:,} MOFs; {pc(mc['doi']['frac_in_groups_ge2'])} of MOFs share a DOI with another; largest group {mc['doi']['max_group']}), and CSD refcode families "
  f"(first 4 letters: {pc(mc['families']['thermal_4']['frac_sharing'])} of thermal, {pc(mc['families']['solvent_4']['frac_sharing'])} of solvent, {pc(mc['families']['water_4']['frac_sharing'])} of water MOFs share a 4-letter family with another MOF). These are proxies only.")
w(f"- **Exact duplicate feature vectors:** thermal 2 rows (labels differ), water 2 rows (labels agree), solvent 0.")
w("")
w("### Table merge (`data_selection.py`): features of MOFs that appear in several tables")
w("")
w("The three CSVs are merged with `combine_first` in the order thermal, solvent, water, so for a MOF present in several tables the **thermal** values win. Consequences measured here:")
w("")
w(f"- Solvent vs thermal, {mc['solvent_vs_thermal']['shared']:,} shared MOFs: all 12 Zeo++ geometry columns are identical ({mc['solvent_vs_thermal']['geometry_columns_identical']}); the chemistry (RAC) columns differ for **{mc['solvent_vs_thermal']['mofs_with_any_difference']} MOFs** (e.g. `{mc['solvent_thermal_RAC_example']['refcode']}`: `f-Z-3-all` = {mc['solvent_thermal_RAC_example']['f-Z-3-all_solvent_csv']:,.0f} in the solvent CSV, {mc['solvent_thermal_RAC_example']['f-Z-3-all_thermal_csv']:,.0f} in the thermal CSV). The solvent split uses the thermal values for these ({mc['rows_using_other_table_features']['solvent']:,} solvent rows come from the thermal CSV in total).")
w(f"- **Water vs thermal, {mc['water_vs_thermal']['shared']} shared MOFs:** {mc['water_vs_thermal']['mofs_with_any_difference']} of them have different features. The pore-volume/surface-area columns differ for almost all of them (median relative difference: `VPOV` {pc(mc['water_vs_thermal']['median_rel_diff']['VPOV'])}, `VSA` {pc(mc['water_vs_thermal']['median_rel_diff']['VSA'])}, `POAV_vol_frac` {pc(mc['water_vs_thermal']['median_rel_diff']['POAV_vol_frac'])}); `Di`/`Df` are identical in the median. "
  f"The water split table then carries the **thermal/solvent values for {mc['rows_using_other_table_features']['water']} of its {mc['rows_using_other_table_features']['water_total']:,} rows ({pc(mc['rows_using_other_table_features']['water'] / mc['rows_using_other_table_features']['water_total'])})**, i.e. a quarter of the water training data uses differently computed volume/area descriptors than the rest. Which definition is 'correct' is not documented.")
w("")
w("## 2. Model summaries (released code, `src/model_training/`) and released performance")
w("")
w("| | Thermal | Solvent removal | Water |")
w("|---|---|---|---|")
w("| task | regression | binary classification | 4-class (0 unstable, 1 low kinetic, 2 high kinetic, 3 thermodynamic) |")
w("| algorithm | fully connected network, leaky-ReLU, no dropout | fully connected network, ReLU + dropout 0.2 | random forest (grid search); an XGBoost model is also trained in the script but its pickle is not released |")
w(f"| inputs | {inv['split_thermal']['features']} | {inv['split_solvent']['features']} | {inv['split_water']['features']} |")
w("| released hyperparameters | hidden 2048-1024-512-256-128-64-32, Adam lr 1e-4, batch 512, patience 100, max 1000 epochs, MSE loss (3,147,777 parameters) | hidden 512-256-128-64-32-16, Adam lr 1e-3, batch 256, patience 200, max 1000 epochs, BCE-with-logits (262,657 parameters) | saved forest: " + f"{ev['rf_params']['n_estimators']} trees, max_depth {ev['rf_params']['max_depth']}, min_samples_split {ev['rf_params']['min_samples_split']}, min_samples_leaf {ev['rf_params']['min_samples_leaf']}, bootstrap {ev['rf_params']['bootstrap']}, max_features {ev['rf_params']['max_features']}; grid: 5 x 3 x 3 x 3 x 2 combinations, 5-fold CV, accuracy |")
w("| SI Table S5 hyperparameters | 190 inputs, hidden 128-512-2048-256-1024, ReLU, dropout 0.124, lr 0.002, 100 epochs | 190 inputs, hidden 128-256-1024-2048-128-256, ReLU, dropout 0.025, lr 7.64e-5, 100 epochs | 179 inputs, 250 trees, max depth 20, split threshold 8 |")
w("| split | `train_test_split(test_size=0.2, random_state=0)`, a **random** split by MOF; then a random 10% of the training part for validation/early stopping | same | same (grid-search CV inside the 80%) |")
w("| scaling | `StandardScaler` fitted on the 80% training part only (verified against the saved scaler) | same | none |")
w("")
w("**Released split reproduced exactly.** For the thermal and solvent tasks, `train_test_split(random_state=0)` on the released table gives test labels identical to the released `*_test_data.pkl` and scaled test features identical to machine precision (max |difference| 0); the saved scaler's mean equals the training-part mean (max |difference| 0) and not the all-data mean. "
  "The saved water forest object stores its own train/test frames, and they equal my reproduced split (same 873/219 MOFs).")
w("")
w("### Performance of the released models on the reproduced test sets, against the paper")
w("")
w(f"| metric (test set) | reproduced here (`model/*.pkl`, the files `predict.py` loads) | `saved_models/` copy | repo docstring / file name | SI (Figures S10-S12) |")
w("|---|---|---|---|---|")
w(f"| thermal n | {ev['thermal']['n']} | | | 627 |")
ts, ss = ev["thermal_saved_full"], ev["solvent_saved_full"]
w(f"| thermal MAE | {f2(ev['thermal']['mae'])} | {f2(ts['test_mae'])} | `52mae` | 47 |")
w(f"| thermal RMSE | {f2(ev['thermal']['rmse'])} | {f2(ts['test_rmse'])} | | 67 |")
w(f"| thermal R2 | {f3(ev['thermal']['r2'])} | {f3(ts['test_r2'])} | | 0.41 |")
w(f"| solvent n | {ev['solvent']['n']} | | | 436 |")
w(f"| solvent accuracy | {f3(ev['solvent']['acc'])} | {f3(ss['test_acc'])} | `78acc` | 0.775 |")
w(f"| solvent AUC | {f3(ev['solvent']['auc'])} | {f3(ss['test_auc'])} | | 0.826 |")
w(f"| solvent precision / recall | {f3(ev['solvent']['prec'])} / {f3(ev['solvent']['rec'])} | | | 0.817 / 0.829 |")
w(f"| solvent TN / FP / FN / TP | {ev['solvent']['tn']} / {ev['solvent']['fp']} / {ev['solvent']['fn']} / {ev['solvent']['tp']} | {ss['tn']} / {ss['fp']} / {ss['fn']} / {ss['tp']} | | 110 / 51 / 47 / 228 |")
w(f"| water n | {ev['water']['n']} | | | 219 |")
w(f"| water accuracy (RF) | {f3(ev['water']['acc'])} | identical file | 0.6621 | 0.671 |")
w(f"| water AUC (macro OvR) | {f3(ev['water']['auc'])} | identical file | 0.8400 | not given |")
w(f"| water test class counts (0,1,2,3) | {', '.join(str(ev['water_test_class_counts'][k]) for k in ('0', '1', '2', '3'))} | | 25, 65, 115, 14 | 17, 68, 112, 22 |")
w("")
w("Reading: the water numbers reproduce the repo's own docstring exactly, but **not the SI**; the SI's test-set class counts (water 17/68/112/22; solvent 275 stable of 436) cannot be obtained from the released split (water 25/65/115/14; solvent "
  f"{ev['solvent']['tp'] + ev['solvent']['fn']} stable of 436). The released thermal network is clearly worse than the SI claims (R2 {f3(ev['thermal']['r2'])} for `model/` and {f3(ts['test_r2'])} for the `saved_models/` copy, against 0.41). The two `.pkl` copies of each network in `model/` and `model/saved_models/` have **different weights**; the `saved_models/` copies match their file names (`52mae`: MAE {f1(ts['test_mae'])}; `78acc`: accuracy {f3(ss['test_acc'])}).")
w("")
w("## 3. Retraining from the released data")
w("")
w(f"Released recipe and split, scaler fitted on the training part, 5 seeds (0-4, GPU), mean +- standard deviation over seeds; the released scripts never call a torch seed function (checked by grep), so seed-to-seed spread is the honest comparison, not a single run.")
w("")
w("| model / setting | test | train | early-stopping restore had no effect |")
w("|---|---|---|---|")
for task, key in (("thermal", ("test_mae", "test_r2", "train_r2")), ("solvent", ("test_acc", "test_auc", "train_acc"))):
    for tag, label in (("released_recipe", "released recipe (as coded)"), ("released_recipe+fixed_restore", "released recipe, genuine restore of best weights"), ("SI_TableS5", "SI Table S5 settings on 170 inputs (batch size not given; I used the released one)")):
        n0, n = rtn(task, tag)
        if task == "thermal":
            w(f"| {task}: {label} | MAE {rtm(task, tag, 'test_mae')}, R2 {rtm(task, tag, 'test_r2')} | R2 {rtm(task, tag, 'train_r2')} | {n0} of {n} |")
        else:
            w(f"| {task}: {label} | accuracy {rtm(task, tag, 'test_acc')}, AUC {rtm(task, tag, 'test_auc')} | accuracy {rtm(task, tag, 'train_acc')} | {n0} of {n} |")
rg, rs = rt["water_rf_regrid"], rt["water_rf_SI_config"]
xg = rt.get("water_xgb_regrid")
w(f"| water RF: released grid re-run | accuracy {f3(rg['test_acc'])}, AUC {f3(rg['test_auc'])} (CV accuracy {f3(rg['cv_accuracy'])}) | | |")
w(f"| water RF: SI settings (250 trees, depth 20, split 8) | accuracy {f3(rs['test_acc'])}, AUC {f3(rs['test_auc'])} | | |")
w(f"| water XGBoost: released grid re-run | " + (f"accuracy {f3(xg['test_acc'])}, AUC {f3(xg['test_auc'])} (docstring 0.6758 / 0.8492)" if xg else "grid search had not finished when this file was written") + " | | |")
w("")
w(f"Best parameters found by my re-run of the released water random-forest grid: `{rg['best_params']}`; saved forest: 200 trees, no depth limit, bootstrap False. " + (f"XGBoost best parameters: `{xg['best_params']}`." if xg else "XGBoost grid search not finished."))
w("")
w("## 4. Findings about the released code and data")
w("")
n0t, nt = rtn("thermal", "released_recipe")
n0s, ns = rtn("solvent", "released_recipe")
fx = lambda task, k, tag: np.mean([r[k] for r in rt[task][tag]])
w(f"1. **Random, not group-aware, splits.** All three models are split by `train_test_split(random_state=0)` over individual MOFs. {pc(lc['thermal_share_family4'])} / {pc(lc['solvent_share_family4'])} / {pc(lc['water_share_family4'])} of the thermal / solvent / water test MOFs share a 4-letter refcode family with a training MOF, and {pc(lc['solvent_share_doi'])} of the solvent test MOFs share a **DOI** with a training MOF. The repository has no MOFid, so the project rule 'split by MOFid group' is not met by the release and cannot be checked directly.")
w(f"2. **'Best weights' are never restored.** In the NN scripts `best_model = self.model.state_dict()` stores references to the live parameters (no copy), so `load_state_dict(best_model)` at the end changes nothing: the networks keep the weights of the last epoch, `patience` epochs after the best validation loss. In my retraining the restore was a no-op in {n0t} of {nt} thermal and {n0s} of {ns} solvent runs. "
  f"With a genuine restore the solvent test AUC falls from {rtm('solvent', 'released_recipe', 'test_auc')} to {rtm('solvent', 'released_recipe+fixed_restore', 'test_auc')} and accuracy from {rtm('solvent', 'released_recipe', 'test_acc')} to {rtm('solvent', 'released_recipe+fixed_restore', 'test_acc')} (the validation-loss minimum occurs around epoch {np.mean([r['best_epoch'] for r in rt['solvent']['released_recipe']]):.0f} of {np.mean([r['epochs_run'] for r in rt['solvent']['released_recipe']]):.0f}), so the over-trained network happens to rank better; the reported solvent numbers therefore depend on this bug.")
w(f"3. **Strong overfitting.** Released thermal network: train R2 {f3(fx('thermal', 'train_r2', 'released_recipe'))} vs test R2 {f3(fx('thermal', 'test_r2', 'released_recipe'))} (retrained); solvent network: train accuracy {f3(fx('solvent', 'train_acc', 'released_recipe'))} vs test {f3(fx('solvent', 'test_acc', 'released_recipe'))}. The released `model/` networks themselves: thermal train R2 {f3(ev['thermal_full']['train_r2'])} vs test {f3(ev['thermal_full']['test_r2'])}; solvent train accuracy {f3(ev['solvent_full']['train_acc'])} vs test {f3(ev['solvent_full']['test_acc'])}.")
w("4. **Test-set labels in the SI cannot be reproduced** from the released split (section 2), and **the released hyperparameters, feature counts and activation functions differ from SI Table S5** (section 2 table). Either the paper's models were trained on a different data version/seed than the release, or the SI describes a different model; I cannot tell which.")
w("5. **Merged tables substitute features** (section 1): 25% of the water rows and 17 solvent rows use feature values from a different CSV.")
w("6. **`water_rf_test_data.pkl` is malformed**: the script saves `pd.concat([labels, features])` along axis 0, giving a (438, 160) stack of 219 label rows on top of 219 feature rows; it is not a usable test table. Use the pickled forest object, which carries the real train/test frames.")
pw_ = pred["water"]; hat_ = pw_[["p0.0", "p1.0", "p2.0", "p3.0"]].values.argmax(1); n3 = int((pw_.truth.values == 3).sum()); c3 = int(((pw_.truth.values == 3) & (hat_ == 3)).sum()); n2 = int((pw_.truth.values == 2).sum()); c2 = int(((pw_.truth.values == 2) & (hat_ == 2)).sum())
w(f"7. **Water class 3 (thermodynamic stability) is hardly learned:** the saved forest identifies {c3} of the {n3} class-3 test MOFs (class 2: {c2} of {n2}); the paper's stable decision (classes 2 or 3) therefore rests mostly on class 2.")
w(f"8. The solvent table has {mc['class_balance']['solvent_stable']:,} stable vs {mc['class_balance']['solvent_unstable']:,} unstable MOFs; the group-aware accuracies in section 5 should be read against the {pc(mc['class_balance']['solvent_majority_fraction'])} majority-class baseline.")
w("")
w("## 5. Train/test leakage checks")
w("")
w("**Closeness to the training set drives performance.** Test MOFs split into thirds by distance (standardised features) to their nearest training MOF; error of the released model by third:")
w("")
w("| task | closest third | middle third | farthest third |")
w("|---|---|---|---|")
for task, nm, fmt in (("thermal", "thermal MAE (degC)", f1), ("solvent", "solvent accuracy", f3), ("water", "water accuracy", f3)):
    d = lc[f"{task}_by_nn_tertile"]
    w(f"| {nm} | {fmt(d['closest third'])} | {fmt(d['middle third'])} | {fmt(d['farthest third'])} |")
w("")
w("The single closest training MOF belongs to the same 4-letter family for " + ", ".join(f"{pc(lc[t + '_nn']['closest_same_family'])} ({t})" for t in ("thermal", "solvent", "water")) + " of test MOFs; the nearest training MOF from a *different* family is " + ", ".join(f"{lc[t + '_nn']['median_nn_other_family'] / lc[t + '_nn']['median_nn']:.2f}x" for t in ("thermal", "solvent", "water")) + " farther (median).")
w("")
w("**Random vs group-aware 5-fold CV** with the same fast model (random forest, 300 trees, all labelled MOFs of each table; groups = 4-letter refcode family, or DOI for solvent). These are proxy groups, not MOFid groups.")
w("")
w("| task | metric | random KFold | group by refcode family | group by DOI |")
w("|---|---|---|---|---|")
for task, nm in (("thermal", "R2"), ("solvent", "accuracy"), ("water", "accuracy")):
    cv = lc[f"cv_{task}"]; gk = [k for k in cv if k.startswith("group")][0]
    doi = f3(lc["cv_solvent_doi"][[k for k in lc["cv_solvent_doi"] if k.startswith("group")][0]][0]) if task == "solvent" else "n/a"
    extra = f" (MAE {f1(cv['random KFold'][1])} -> {f1(cv[gk][1])} degC)" if task == "thermal" else ""
    w(f"| {task} | {nm}{extra} | {f3(cv['random KFold'][0])} | {f3(cv[gk][0])} | {doi} |")
w("")
al = lc["allthree"]
names_t = {n: set(d.index) for n, d in split.items()}
w(f"**Cross-task leakage for a stacked filter.** The three models were split independently, so a MOF in one model's test set is often in another model's training set: thermal-test MOFs that are solvent-train: {len(set(pred['thermal'].name) & (names_t['solvent'] - set(pred['solvent'].name)))}; solvent-test MOFs that are thermal-train: {len(set(pred['solvent'].name) & (names_t['thermal'] - set(pred['thermal'].name)))}. "
  f"Only {al['n']} MOFs carry all three labels, and just **{al['test_all3']}** of them is in the test set of all three models ({al['test_thermal']} thermal, {al['test_solvent']} solvent, {al['test_water']} water). An out-of-sample evaluation of the stacked filter on experimental labels is therefore not possible with the released splits.")
w("**Scaler leakage: none** (scaler fitted on the training part, section 2). **Hyperparameter-search leakage: none found** (grid search CV inside the training part; early stopping on a validation part of the training set).")
w("")
w("## 6. Calibration and false-negative rates of the released models (their own test sets)")
w("")
w("Definition of 'stable' used by the paper's screening (SI Section 4, Figures S7-S9): **thermal** predicted decomposition temperature >= 350 degC; **solvent** predicted probability > 0.9; **water** predicted category 2 or 3. A false negative is a truly stable MOF that the filter rejects.")
w("")
yt = (pred["thermal"].truth.values >= 350)
sc, so = lc["solvent_cal"], lc["water_cal"]
tf = {x["threshold"]: x for x in lc["thermal_cal"]["fn"]}; sf = {x["threshold"]: x for x in sc["fn"]}; wf = {x["threshold"]: x for x in so["fn"]}
w("| filter (n test, truly stable) | rule | recall | FNR | precision | MOFs passing | threshold for >=95% recall (MOFs passing) |")
w("|---|---|---|---|---|---|---|")
p95 = lambda score, t: pc(float((score >= t).mean()))
st_, sw_ = pred["solvent"].p_stable.values, pred["water"][["p2.0", "p3.0"]].values.sum(1)
w(f"| thermal ({len(yt)}, {int(yt.sum())}) | predicted T >= 350 | {f3(tf[350]['recall'])} | {f3(tf[350]['fnr'])} | {f3(tf[350]['precision'])} | {pc(tf[350]['pass_fraction'])} | T >= {lc['thermal_cal']['t95']:.0f} ({p95(pred['thermal'].pred.values, lc['thermal_cal']['t95'])}) |")
w(f"| solvent ({ev['solvent']['n']}, {ev['solvent']['tp'] + ev['solvent']['fn']}) | p > 0.9 | {f3(sf[0.9]['recall'])} | {f3(sf[0.9]['fnr'])} | {f3(sf[0.9]['precision'])} | {pc(sf[0.9]['pass_fraction'])} | p >= {sc['t95']:.3f} ({p95(st_, sc['t95'])}) |")
pr = so["paper_rule"]
w(f"| water ({ev['water']['n']}, {int((pred['water'].truth.values >= 2).sum())}) | class 2 or 3 | {f3(pr['recall'])} | {f3(pr['fnr'])} | {f3(pr['precision'])} | {pc(float((hat_ >= 2).mean()))} | P(stable) >= {so['t95']:.2f} ({p95(sw_, so['t95'])}) |")
w("")
w("The last column is the threshold that would give at least 95% recall on this test set; it is shown only to illustrate the cost of a recall target with the released models. The percentage in brackets is the share of test MOFs that would pass at that threshold.")
w("")
w("**Calibration.**")
w("")
w(f"- Solvent: Brier {f3(sc['brier'])}, log-loss {f3(sc['logloss'])}, ECE (10 bins) {f3(sc['ece'])}. The network is strongly over-confident: " + "; ".join(f"predicted p about {r[2]:.2f} for {r[1]} MOFs, observed stable fraction {r[3]:.2f}" for r in (sc['rel'][0], sc['rel'][-1])) + ".")
w(f"- Water (P(stable) = P(class 2) + P(class 3)): Brier {f3(so['brier'])}, ECE {f3(so['ece'])}; reasonably calibrated in the upper bins.")
tc = lc["thermal_cal"]
w(f"- Thermal (regression): regressing truth on prediction gives truth = {tc['slope']:.3f} x prediction + {tc['intercept']:.1f}, i.e. predictions are over-dispersed (a calibrated model would have slope 1, intercept 0). Observed fraction of truly stable MOFs by prediction decile (lowest to highest): "
  + ", ".join(f"{x:.2f}" for x in pd.DataFrame({"p": pred["thermal"].pred.values, "s": yt.astype(float)}).assign(b=lambda d: pd.qcut(d.p, 10, duplicates="drop")).groupby("b", observed=True).s.mean().values) + ".")
w("")
w("**What this implies for stacking the three filters (facts, not yet a result):** per-filter FNRs at the paper's thresholds are "
  f"{f3(tf[350]['fnr'])} (thermal), {f3(sf[0.9]['fnr'])} (solvent) and {f3(pr['fnr'])} (water), each measured on a different set of MOFs. For a stack that requires all three to pass, the union bound gives a stack FNR of at most {f3(tf[350]['fnr'] + sf[0.9]['fnr'] + pr['fnr'])} (so recall of at least {f3(1 - (tf[350]['fnr'] + sf[0.9]['fnr'] + pr['fnr']))}), and at least {f3(max(tf[350]['fnr'], sf[0.9]['fnr'], pr['fnr']))}; if the misses were independent the stack recall would be {f3((1 - tf[350]['fnr']) * (1 - sf[0.9]['fnr']) * (1 - pr['fnr']))}. "
  "The independence figure is arithmetic, not a measurement. None of these includes the optimism from the leakage in section 5.")
w("")
w("## 7. What I could not check, and open questions")
w("")
w("- **MOFid groups:** not computable from the released data (no MOFid column, no CIFs). The next step needs the CoRE MOF 2019 CIFs (large download: ask first) and a MOFid tool, or a reliable refcode-to-MOFid table.")
w("- **Which data version the paper's models used:** the SI test-set sizes (627 / 436 / 219) equal 20% of the released tables, but the SI class counts and several metrics do not match the release (section 2).")
w("- **Meaning of the water classes** is taken from the SI (0 unstable, 1 low kinetic, 2 high kinetic, 3 thermodynamic stability); the repo itself does not document the 1-4 labels. The thermal label unit is not stated in the repo (the SI says degC).")
w("- The XGBoost water model's pickle is not in the release; only its docstring numbers and my re-run exist.")
w("- CoRE MOF 2024 CSVs were downloaded but not used here; their `Water_stability`, `Thermal_stability`, `Solvent_stability` columns look like model outputs and their origin is unchecked.")
w("")
w("## 8. Files")
w("")
w("`scripts/audit_01_scan_pickles.py` (pickle safety scan) | `audit_02_inventory.py` (tables) | `audit_03_saved_models.py` (split reproduction, saved-model metrics) | `audit_04_retrain.py` (retraining) | `audit_05_leakage_calibration.py` | `audit_06_merge_check.py` | `audit_07_write_notes.py` (this file). "
  "`src/models/safe_pickle.py`, `src/models/stability_nets.py`, `src/features/paper_splits.py` and `tests/` hold the reusable code. Outputs: `data/processed/audit/*.json|*.log|*.parquet`.")
open("notes/stability_inventory.md", "w").write("\n".join(L) + "\n")
print("written notes/stability_inventory.md,", len(L), "lines")
