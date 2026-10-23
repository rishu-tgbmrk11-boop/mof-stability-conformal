# Audit of the paper's stability ML (OptiMOF release)

Numbers measured by this audit were produced by the scripts in `scripts/audit_0*.py` and are read from their saved outputs in `data/processed/audit/` by `audit_07_write_notes.py`. The only values typed into this file are quotations: the paper's SI figures (marked 'SI'), the repo's docstring/file-name values (marked 'docstring' or 'file name'), the released hyperparameters read from the repo's JSON/source, and a few counts that the scripts printed (parameter counts, duplicate-row counts, the 20 constant columns). Reproduce with `python -I scripts/audit_0N_*.py` inside the `mof-stability` environment, in order 01 to 07 (the scripts are run from the repository root).

## 0. Scope, sources and safety

- **Paper:** Liu et al., J. Am. Chem. Soc. 2026, 148, 34821, doi:10.1021/jacs.6c10566. SI Section 4 ('Machine-Learning Stability Evaluation') and the SI Machine Learning Methods tables (Table S5 etc.) are the claims audited here. SI text was read from the user's PDF copy (`ja6c10566_si_001.pdf`).
- **Released code and data:** `github.com/snurr-group/optimof`, shallow clone, commit `75cbd1389a551e638376b78c5688c409958d52cd` (2025-10-06, message 'quick commit'). **CoRE MOF 2024:** Zenodo record 15055758 v1.1, five CSVs only (checksums verified); not used by the audit below.
- **Environment:** `mof-stability` (Python 3.11.17, torch 2.14.1+cu130, scikit-learn 1.9.1, xgboost 3.4.2, pandas 3.0.6, MAPIE 1.6.0). The released pickles were written with scikit-learn 1.5.1 and are loaded with a version warning; accuracies of the saved water forest reproduce the repo's own docstring exactly (section 2), which is evidence the version difference did not change predictions.
- **Untrusted pickles:** every pickle was loaded with `python -I` through `src/models/safe_pickle.py`, a restricted unpickler that only resolves allow-listed numpy/pandas/scikit-learn/scipy/xgboost/torch-tensor names and maps the repository's own classes (`__main__.ThermalModel`, `__main__.SolventModel`, `__main__.WaterStabilityRF`) to inert stand-ins. Scanned 17 pickle files: no `os`, `subprocess`, `eval` or similar global was requested by any of them. Tests: `tests/test_safe_pickle.py` (a malicious pickle is blocked and its payload never runs).

## 1. Inventory of the three stability tables

| | Thermal | Solvent-removal (activation) | Water |
|---|---|---|---|
| released file | `data/thermal/thermal_all_data.csv` | `data/solvent/solvent_all_data.csv` | `data/water_and_haz/data.csv` |
| MOFs (rows) | 3,132 | 2,179 | 1,092 |
| id used | CSD refcode (e.g. `ABAVIJ`) | CSD refcode | CSD refcode (`MOF_name`); 911 CoRE + 181 'Burtch' |
| label | decomposition temperature `T` (min 35.0, max 654.3; unit not stated in the repo; the SI uses degC) | `flag` = +1 stable (1,296) / -1 unstable (883) | `water_label` 1-4: 1: 89, 2: 350, 3: 555, 4: 98 |
| features in the released split table | 170 | 170 | 159 |
| paper SI feature counts | 190 | 190 | 179 |
| NaN in split table | 0 | 0 | 0 |
| majority-class share | n/a | 59.5% | 50.8% |

- **Features:** the thermal and solvent CSVs each hold 190 numeric feature columns (170 in the released split + the 20 that `data_selection.py` removes as 'empty'); the water CSV holds 159. I checked the 20 removed columns in the CSVs and all 20 are indeed constant (nunique = 1). So the thermal and solvent networks are trained on 170 non-constant features, not on the 190 the SI reports (the SI says only one all-zero feature, `D_func-alpha-0-all`, was dropped); the water model uses 159, not 179.
- **Overlap between the tables (by refcode):** thermal & solvent 1,163; thermal & water 244; solvent & water 125; **all three 92**; union 4,963 (`all_in_one.pkl` has 4,963 rows; the SI says the MOFMap union has 4,148).
- **Grouping information:** no MOFid column exists in any released table, and the repo ships no CIFs, so **MOFid groups cannot be computed from the released files**. What exists: DOIs in the solvent table only (1,174 DOIs for 2,179 MOFs; 65.4% of MOFs share a DOI with another; largest group 35), and CSD refcode families (first 4 letters: 50.4% of thermal, 52.5% of solvent, 67.6% of water MOFs share a 4-letter family with another MOF). These are proxies only.
- **Exact duplicate feature vectors:** thermal 2 rows (labels differ), water 2 rows (labels agree), solvent 0.

### Table merge (`data_selection.py`): features of MOFs that appear in several tables

The three CSVs are merged with `combine_first` in the order thermal, solvent, water, so for a MOF present in several tables the **thermal** values win. Consequences measured here:

- Solvent vs thermal, 1,163 shared MOFs: all 12 Zeo++ geometry columns are identical (True); the chemistry (RAC) columns differ for **17 MOFs** (e.g. `YIZGEV`: `f-Z-3-all` = 2,752 in the solvent CSV, 125,952 in the thermal CSV). The solvent split uses the thermal values for these (1,163 solvent rows come from the thermal CSV in total).
- **Water vs thermal, 244 shared MOFs:** 243 of them have different features. The pore-volume/surface-area columns differ for almost all of them (median relative difference: `VPOV` 16.1%, `VSA` 54.3%, `POAV_vol_frac` 12.5%); `Di`/`Df` are identical in the median. The water split table then carries the **thermal/solvent values for 277 of its 1,092 rows (25.4%)**, i.e. a quarter of the water training data uses differently computed volume/area descriptors than the rest. Which definition is 'correct' is not documented.

## 2. Model summaries (released code, `src/model_training/`) and released performance

| | Thermal | Solvent removal | Water |
|---|---|---|---|
| task | regression | binary classification | 4-class (0 unstable, 1 low kinetic, 2 high kinetic, 3 thermodynamic) |
| algorithm | fully connected network, leaky-ReLU, no dropout | fully connected network, ReLU + dropout 0.2 | random forest (grid search); an XGBoost model is also trained in the script but its pickle is not released |
| inputs | 170 | 170 | 159 |
| released hyperparameters | hidden 2048-1024-512-256-128-64-32, Adam lr 1e-4, batch 512, patience 100, max 1000 epochs, MSE loss (3,147,777 parameters) | hidden 512-256-128-64-32-16, Adam lr 1e-3, batch 256, patience 200, max 1000 epochs, BCE-with-logits (262,657 parameters) | saved forest: 200 trees, max_depth None, min_samples_split 2, min_samples_leaf 1, bootstrap False, max_features sqrt; grid: 5 x 3 x 3 x 3 x 2 combinations, 5-fold CV, accuracy |
| SI Table S5 hyperparameters | 190 inputs, hidden 128-512-2048-256-1024, ReLU, dropout 0.124, lr 0.002, 100 epochs | 190 inputs, hidden 128-256-1024-2048-128-256, ReLU, dropout 0.025, lr 7.64e-5, 100 epochs | 179 inputs, 250 trees, max depth 20, split threshold 8 |
| split | `train_test_split(test_size=0.2, random_state=0)`, a **random** split by MOF; then a random 10% of the training part for validation/early stopping | same | same (grid-search CV inside the 80%) |
| scaling | `StandardScaler` fitted on the 80% training part only (verified against the saved scaler) | same | none |

**Released split reproduced exactly.** For the thermal and solvent tasks, `train_test_split(random_state=0)` on the released table gives test labels identical to the released `*_test_data.pkl` and scaled test features identical to machine precision (max |difference| 0); the saved scaler's mean equals the training-part mean (max |difference| 0) and not the all-data mean. The saved water forest object stores its own train/test frames, and they equal my reproduced split (same 873/219 MOFs).

### Performance of the released models on the reproduced test sets, against the paper

| metric (test set) | reproduced here (`model/*.pkl`, the files `predict.py` loads) | `saved_models/` copy | repo docstring / file name | SI (Figures S10-S12) |
|---|---|---|---|---|
| thermal n | 627 | | | 627 |
| thermal MAE | 53.57 | 52.70 | `52mae` | 47 |
| thermal RMSE | 81.30 | 74.42 | | 67 |
| thermal R2 | 0.138 | 0.278 | | 0.41 |
| solvent n | 436 | | | 436 |
| solvent accuracy | 0.778 | 0.775 | `78acc` | 0.775 |
| solvent AUC | 0.830 | 0.831 | | 0.826 |
| solvent precision / recall | 0.805 / 0.833 | | | 0.817 / 0.829 |
| solvent TN / FP / FN / TP | 120 / 53 / 44 / 219 | 123 / 50 / 48 / 215 | | 110 / 51 / 47 / 228 |
| water n | 219 | | | 219 |
| water accuracy (RF) | 0.662 | identical file | 0.6621 | 0.671 |
| water AUC (macro OvR) | 0.840 | identical file | 0.8400 | not given |
| water test class counts (0,1,2,3) | 25, 65, 115, 14 | | 25, 65, 115, 14 | 17, 68, 112, 22 |

Reading: the water numbers reproduce the repo's own docstring exactly, but **not the SI**; the SI's test-set class counts (water 17/68/112/22; solvent 275 stable of 436) cannot be obtained from the released split (water 25/65/115/14; solvent 263 stable of 436). The released thermal network is clearly worse than the SI claims (R2 0.138 for `model/` and 0.278 for the `saved_models/` copy, against 0.41). The two `.pkl` copies of each network in `model/` and `model/saved_models/` have **different weights**; the `saved_models/` copies match their file names (`52mae`: MAE 52.7; `78acc`: accuracy 0.775).

## 3. Retraining from the released data

Released recipe and split, scaler fitted on the training part, 5 seeds (0-4, GPU), mean +- standard deviation over seeds; the released scripts never call a torch seed function (checked by grep), so seed-to-seed spread is the honest comparison, not a single run.

| model / setting | test | train | early-stopping restore had no effect |
|---|---|---|---|
| thermal: released recipe (as coded) | MAE 54.440 +- 0.490, R2 0.178 +- 0.037 | R2 0.839 +- 0.024 | 5 of 5 |
| thermal: released recipe, genuine restore of best weights | MAE 54.561 +- 0.808, R2 0.227 +- 0.010 | R2 0.739 +- 0.054 | 0 of 5 |
| thermal: SI Table S5 settings on 170 inputs (batch size not given; I used the released one) | MAE 58.106 +- 1.376, R2 0.253 +- 0.024 | R2 0.608 +- 0.048 | 5 of 5 |
| solvent: released recipe (as coded) | accuracy 0.763 +- 0.009, AUC 0.820 +- 0.008 | accuracy 0.964 +- 0.002 | 5 of 5 |
| solvent: released recipe, genuine restore of best weights | accuracy 0.739 +- 0.008, AUC 0.762 +- 0.021 | accuracy 0.809 +- 0.026 | 0 of 5 |
| solvent: SI Table S5 settings on 170 inputs (batch size not given; I used the released one) | accuracy 0.755 +- 0.016, AUC 0.809 +- 0.008 | accuracy 0.954 +- 0.007 | 5 of 5 |
| water RF: released grid re-run | accuracy 0.662, AUC 0.840 (CV accuracy 0.670) | | |
| water RF: SI settings (250 trees, depth 20, split 8) | accuracy 0.639, AUC 0.830 | | |
| water XGBoost: released grid re-run | accuracy 0.689, AUC 0.845 (docstring 0.6758 / 0.8492) | | |

Best parameters found by my re-run of the released water random-forest grid: `{'bootstrap': False, 'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}`; saved forest: 200 trees, no depth limit, bootstrap False. XGBoost best parameters: `{'colsample_bytree': 1.0, 'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 200, 'subsample': 0.8}`.

## 4. Findings about the released code and data

1. **Random, not group-aware, splits.** All three models are split by `train_test_split(random_state=0)` over individual MOFs. 43.7% / 50.2% / 58.9% of the thermal / solvent / water test MOFs share a 4-letter refcode family with a training MOF, and 60.6% of the solvent test MOFs share a **DOI** with a training MOF. The repository has no MOFid, so the project rule 'split by MOFid group' is not met by the release and cannot be checked directly.
2. **'Best weights' are never restored.** In the NN scripts `best_model = self.model.state_dict()` stores references to the live parameters (no copy), so `load_state_dict(best_model)` at the end changes nothing: the networks keep the weights of the last epoch, `patience` epochs after the best validation loss. In my retraining the restore was a no-op in 5 of 5 thermal and 5 of 5 solvent runs. With a genuine restore the solvent test AUC falls from 0.820 +- 0.008 to 0.762 +- 0.021 and accuracy from 0.763 +- 0.009 to 0.739 +- 0.008 (the validation-loss minimum occurs around epoch 25 of 226), so the over-trained network happens to rank better; the reported solvent numbers therefore depend on this bug.
3. **Strong overfitting.** Released thermal network: train R2 0.839 vs test R2 0.178 (retrained); solvent network: train accuracy 0.964 vs test 0.763. The released `model/` networks themselves: thermal train R2 0.842 vs test 0.138; solvent train accuracy 0.966 vs test 0.778.
4. **Test-set labels in the SI cannot be reproduced** from the released split (section 2), and **the released hyperparameters, feature counts and activation functions differ from SI Table S5** (section 2 table). Either the paper's models were trained on a different data version/seed than the release, or the SI describes a different model; I cannot tell which.
5. **Merged tables substitute features** (section 1): 25% of the water rows and 17 solvent rows use feature values from a different CSV.
6. **`water_rf_test_data.pkl` is malformed**: the script saves `pd.concat([labels, features])` along axis 0, giving a (438, 160) stack of 219 label rows on top of 219 feature rows; it is not a usable test table. Use the pickled forest object, which carries the real train/test frames.
7. **Water class 3 (thermodynamic stability) is hardly learned:** the saved forest identifies 0 of the 14 class-3 test MOFs (class 2: 89 of 115); the paper's stable decision (classes 2 or 3) therefore rests mostly on class 2.
8. The solvent table has 1,296 stable vs 883 unstable MOFs; the group-aware accuracies in section 5 should be read against the 59.5% majority-class baseline.

## 5. Train/test leakage checks

**Closeness to the training set drives performance.** Test MOFs split into thirds by distance (standardised features) to their nearest training MOF; error of the released model by third:

| task | closest third | middle third | farthest third |
|---|---|---|---|
| thermal MAE (degC) | 30.3 | 51.3 | 79.1 |
| solvent accuracy | 0.973 | 0.703 | 0.655 |
| water accuracy | 0.795 | 0.616 | 0.575 |

The single closest training MOF belongs to the same 4-letter family for 22.2% (thermal), 29.8% (solvent), 23.7% (water) of test MOFs; the nearest training MOF from a *different* family is 1.19x, 1.34x, 1.21x farther (median).

**Random vs group-aware 5-fold CV** with the same fast model (random forest, 300 trees, all labelled MOFs of each table; groups = 4-letter refcode family, or DOI for solvent). These are proxy groups, not MOFid groups.

| task | metric | random KFold | group by refcode family | group by DOI |
|---|---|---|---|---|
| thermal | R2 (MAE 44.1 -> 50.1 degC) | 0.482 | 0.374 | n/a |
| solvent | accuracy | 0.793 | 0.716 | 0.620 |
| water | accuracy | 0.662 | 0.558 | n/a |

**Cross-task leakage for a stacked filter.** The three models were split independently, so a MOF in one model's test set is often in another model's training set: thermal-test MOFs that are solvent-train: 189; solvent-test MOFs that are thermal-train: 189. Only 92 MOFs carry all three labels, and just **1** of them is in the test set of all three models (27 thermal, 13 solvent, 22 water). An out-of-sample evaluation of the stacked filter on experimental labels is therefore not possible with the released splits.
**Scaler leakage: none** (scaler fitted on the training part, section 2). **Hyperparameter-search leakage: none found** (grid search CV inside the training part; early stopping on a validation part of the training set).

## 6. Calibration and false-negative rates of the released models (their own test sets)

Definition of 'stable' used by the paper's screening (SI Section 4, Figures S7-S9): **thermal** predicted decomposition temperature >= 350 degC; **solvent** predicted probability > 0.9; **water** predicted category 2 or 3. A false negative is a truly stable MOF that the filter rejects.

| filter (n test, truly stable) | rule | recall | FNR | precision | MOFs passing | threshold for >=95% recall (MOFs passing) |
|---|---|---|---|---|---|---|
| thermal (627, 344) | predicted T >= 350 | 0.791 | 0.209 | 0.800 | 54.2% | T >= 299 (78.8%) |
| solvent (436, 263) | p > 0.9 | 0.772 | 0.228 | 0.822 | 56.7% | p >= 0.006 (75.2%) |
| water (219, 129) | class 2 or 3 | 0.829 | 0.171 | 0.799 | 61.2% | P(stable) >= 0.31 (79.0%) |

The last column is the threshold that would give at least 95% recall on this test set; it is shown only to illustrate the cost of a recall target with the released models. The percentage in brackets is the share of test MOFs that would pass at that threshold.

**Calibration.**

- Solvent: Brier 0.190, log-loss 1.410, ECE (10 bins) 0.188. The network is strongly over-confident: predicted p about 0.01 for 134 MOFs, observed stable fraction 0.18; predicted p about 0.99 for 247 MOFs, observed stable fraction 0.82.
- Water (P(stable) = P(class 2) + P(class 3)): Brier 0.156, ECE 0.079; reasonably calibrated in the upper bins.
- Thermal (regression): regressing truth on prediction gives truth = 0.575 x prediction + 145.4, i.e. predictions are over-dispersed (a calibrated model would have slope 1, intercept 0). Observed fraction of truly stable MOFs by prediction decile (lowest to highest): 0.06, 0.19, 0.27, 0.37, 0.52, 0.71, 0.75, 0.92, 0.86, 0.84.

**What this implies for stacking the three filters (facts, not yet a result):** per-filter FNRs at the paper's thresholds are 0.209 (thermal), 0.228 (solvent) and 0.171 (water), each measured on a different set of MOFs. For a stack that requires all three to pass, the union bound gives a stack FNR of at most 0.608 (so recall of at least 0.392), and at least 0.228; if the misses were independent the stack recall would be 0.506. The independence figure is arithmetic, not a measurement. None of these includes the optimism from the leakage in section 5.

## 7. What I could not check, and open questions

- **MOFid groups:** not computable from the released data (no MOFid column, no CIFs). The next step needs the CoRE MOF 2019 CIFs (large download: ask first) and a MOFid tool, or a reliable refcode-to-MOFid table.
- **Which data version the paper's models used:** the SI test-set sizes (627 / 436 / 219) equal 20% of the released tables, but the SI class counts and several metrics do not match the release (section 2).
- **Meaning of the water classes** is taken from the SI (0 unstable, 1 low kinetic, 2 high kinetic, 3 thermodynamic stability); the repo itself does not document the 1-4 labels. The thermal label unit is not stated in the repo (the SI says degC).
- The XGBoost water model's pickle is not in the release; only its docstring numbers and my re-run exist.
- CoRE MOF 2024 CSVs were downloaded but not used here; their `Water_stability`, `Thermal_stability`, `Solvent_stability` columns look like model outputs and their origin is unchecked.

## 8. Files

`scripts/audit_01_scan_pickles.py` (pickle safety scan) | `audit_02_inventory.py` (tables) | `audit_03_saved_models.py` (split reproduction, saved-model metrics) | `audit_04_retrain.py` (retraining) | `audit_05_leakage_calibration.py` | `audit_06_merge_check.py` | `audit_07_write_notes.py` (this file). `src/models/safe_pickle.py`, `src/models/stability_nets.py`, `src/features/paper_splits.py` and `tests/` hold the reusable code. Outputs: `data/processed/audit/*.json|*.log|*.parquet`.
