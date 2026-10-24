# Recall-guaranteed stacked stability filters for MOFs

A reproducibility audit of the machine-learning stability filters used in a recent MOF screening study, plus an attempt to give the stacked filters (thermal, solvent-removal and water stability) a statistical recall guarantee using conformal calibration. This is an independent analysis of the paper's released data and models; it is not the authors' repository.

## Background: the paper

**Identification of Metal-Organic Frameworks for CO2 Capture from Humid Flue Gas: Integrating Molecular Simulation, Machine Learning, and Experimental Synthesis**
Jiayang Liu, Xiaoliang Wang, Xiyang Liu, Zi-Ming Ye, Thang D. Pham, Filip Formalik, Manuel Tsotsalas, Omar K. Farha and Randall Q. Snurr.
*J. Am. Chem. Soc.* **2026**, *148*, 34821-34829 (published 4 August 2026).
Paper: https://pubs.acs.org/jacsat/article/148/32/34821/5243531/Identification-of-Metal-Organic-Frameworks-for-CO2 - DOI: https://doi.org/10.1021/jacs.6c10566

What the paper does (as stated in the article):
- It presents an integrated computational high-throughput screening workflow for MOFs that capture CO2 from humid flue gas. The workflow starts from a library of over 110,000 experimental MOFs or MOF-like structures (the CSD MOF subset), explicitly includes water in the adsorption simulations, and adds machine-learning stability analysis.
- Structures are curated (disordered frameworks and frameworks with charge-balancing counterions are excluded, solvent is removed, and quality filters are applied) into a database of 9,922 structures called OptiMOF. Single-molecule CO2 and H2O energy minimisations act as an early filter, followed by grand canonical Monte Carlo mixture simulations that include water vapour.
- MOFs with CO2 uptake above 1.5 mol/kg and CO2/N2 selectivity near or above 50 at 80% relative humidity (81 candidates, 50 unique frameworks by MOFid analysis) were then evaluated with machine-learning models predicting **thermal, activation (solvent-removal) and water stability**, trained on stability data extracted from thousands of experimentally reported MOFs compiled by Kulik and coworkers. Fifteen unique MOFs passed the stability evaluation.
- One top candidate, the known MOF Zn(imPim), was synthesised and showed strong CO2 capture at elevated relative humidity. The study also reports structure-property relationships and structural motifs for CO2 capture under humid conditions.
- The OptiMOF data set and the machine-learning models are released in a public repository (https://github.com/snurr-group/optimof).

The stability filters are the part of the paper this repository examines. According to the paper's Supporting Information, the thermal model is a regression network, the activation model a binary classifier and the water model a random forest with four ordered classes (0 unstable, 1 low kinetic stability, 2 high kinetic stability, 3 thermodynamic stability); they are trained on 3,132, 2,179 and 1,092 MOFs and evaluated on held-out 20% test sets. Candidates are kept if the predicted thermal decomposition temperature is at least 350 degC, the activation metric exceeds 0.9, and the water class is 2 or 3.

## What this repository does

1. **Reproduces and audits** the paper's stability machine learning using the released OptiMOF data and models: inventory of the data, retraining, comparison with the Supporting Information, train/test leakage checks, and calibration of the released models.
2. **Adds a statistical guarantee**: conformal (risk-controlling) calibration so that a filter keeps at least a chosen fraction of the truly stable MOFs, first per stage and then for the stacked filters.

## Main findings so far

Numbers, intervals and caveats are in the notes linked below; this is only a summary.
- The released training code for the thermal and solvent networks stores the "best" weights by reference, so the early-stopping restore does not restore the best epoch.
- The test-set sizes and class counts in the Supporting Information (thermal 627, solvent 436 with 275 stable, water 219 with classes 17/68/112/22) are reproduced exactly by `train_test_split(test_size=0.2, random_state=0)` on the released CSV rows.
- Retraining on that split with the released recipe approximately reproduces the Supporting Information results for solvent and water. The thermal R2 of 0.41 is not reproduced by the released code or checkpoints; the Supporting Information's Table S5 network trained on standardised temperatures reaches it in about a quarter of 50 runs. This was found after the fact and is a diagnosis, not a confirmed account of what the authors did (`notes/phase6.md`).
- Splitting by group (refcode family and DOI, and MOFid where it could be computed) instead of at random lowers test performance clearly; the size of the drop varies a lot from split to split (`notes/phase5.md`, `notes/phase6.md`).
- A per-stage split-conformal recall rule meets its target on average when calibration and test MOFs are exchangeable, but fell short when calibrated on different feature-space clusters (`notes/phase5.md`).
- For the stacked filters, a joint score calibrated on the 17 MOFs (in 14 groups) that are stable at all three stages gives a recall guarantee of 1 - 2 alpha under stated assumptions; with so few positives the highest certifiable target is 0.867. The union-bound version of the stacked guarantee is an assumption, not proved (`notes/phase6.md`).

## Getting started

```
mamba env create -f environment.yml
conda activate mof-stability
pytest tests          # not bare `pytest`: data/raw/ holds vendored test files that fail at collection
bash figures/make_all.sh
```

`data/raw/` is not part of the repository. Put a clone of https://github.com/snurr-group/optimof in `data/raw/optimof/` and the CoRE MOF 2024 CSV files (Zenodo record 15055758) in `data/raw/coremof/csv/`. Most files under `data/processed/` are generated by the scripts in `scripts/` and are not all tracked; each phase's notes end with the commands that regenerate it. The optimof data and models and the CoRE MOF data are not redistributed here; see their own licences. This repository has no licence file yet.

## Notes

| file | content |
|---|---|
| `notes/stability_inventory.md` | audit of the released tables, models and metrics |
| `notes/phase2.md` | group-aware splits, honest retraining, per-stage and cascade conformal control |
| `notes/phase3.md` | tree-model candidates, group cross-validation selection, conformal rerun |
| `notes/phase4.md` | price of recall, limitations, figures |
| `notes/review.md` | independent review of phases 1-4 |
| `notes/phase5.md` | retrain on the paper's split, claim corrections, leave-clusters-out validation, ranges over many splits |
| `notes/phase6.md` | thermal gap diagnosis, joint cascade guarantee, MOFid and fingerprint groups |
| `notes/related_work.md` | related literature (verification status stated in the file) |
| `figures/captions.md` | draft captions for the figures |

## Project notes: goals, rules and plan

The project notes below are kept as originally written.

### Goal
1. **Reproduce** the stability-model machine learning of Liu et al., "Identification of Metal-Organic Frameworks for CO2 Capture from
   Humid Flue Gas: Integrating Molecular Simulation, Machine Learning, and Experimental Synthesis", J. Am. Chem. Soc. 2026, 148, 34821-34829
   (doi:10.1021/jacs.6c10566), using the released data and models of the OptiMOF repository (github.com/snurr-group/optimof) and the
   CoRE MOF 2024 dataset (Zenodo record 15055758). The paper's ML screens candidate MOFs for **water stability, thermal stability and
   solvent-removal (activation) stability**.
2. **Add a statistical guarantee:** apply conformal (risk-controlling) calibration across the **stacked** water, thermal and solvent filters, so
   that the combined filter provably keeps at least a chosen fraction of truly stable MOFs (a recall guarantee), instead of the guarantee
   holding only for each filter in isolation.

### Where things come from (verify before relying on any of it)
- OptiMOF repo (`data/raw/optimof/`): stability tables under `data/` (thermal, solvent, water_and_haz), saved models under `model/`, plots under `performance/`.
  The repo has **no** CO2/H2O adsorption (GCMC) data.
- CoRE MOF 2024 (`data/raw/coremof/`): the CSVs from Zenodo 15055758 only. **Do not download TSA.zip or mofid-v2.zip** (large). The CSVs have MOF ids,
  MOFid strings, geometry descriptors and `Water_stability`, `Thermal_stability`, `Solvent_stability` columns whose origin must be checked, not assumed.
- Earlier work (an abandoned, larger-scope attempt) is archived in `notes/archive_week1.md`. Treat its statements as history, not as verified facts of this project.

### Rules (IMPORTANT)
- **Never fabricate data or results.** If something cannot be computed, say so. Every number written to notes must come from a command that was run, and the command's output must be shown.
- **Ask before deleting** anything, and ask before any large download (> ~50 MB).
- **Small verified steps:** run the code, show the output, check it, then move on.
- **Split by MOFid group** for every train/validation/test split (no MOF, and no duplicate or near-duplicate of one, on both sides of a split). Where the released split does not do this, report it as leakage; do not silently change the paper's split.
- **Units in column names**, e.g. `thermal_stability_c`, `pld_ang`, `density_gcm3`. Probabilities and scores are unit-free but named (`p_water_stable`).
- Random seeds are fixed and recorded with every result.
- **Pickles and other model files from the internet are untrusted data.** Never load them with a plain `pickle.load`. Inspect them first (opcode listing for imported globals),
  load them only with a restricted unpickler that allows scikit-learn / numpy / pandas / xgboost / torch names, and run Python as `python -I` so the working directory is not on the import path.
- Only create or change the conda environment `mof-stability` under `~/miniforge3/envs/`. Do not modify other environments, `~/src` or any other project.
- Write code in `src/` (importable, tested with pytest); notebooks only call it. Every script takes a config and writes outputs and a log to a run folder.
- Run the tests with `pytest tests`, not bare `pytest`: `data/raw/` holds vendored optimof/Zeo++ test files that fail at collection.

### Tech stack
- Environment: mamba from `~/miniforge3`, env `mof-stability`, Python 3.11, pinned in `environment.yml`.
- ML: PyTorch (CUDA), scikit-learn, XGBoost, MAPIE (conformal prediction); pandas + pyarrow (parquet tables); pymatgen and ASE for structures.
- Hardware: laptop with an NVIDIA RTX 4050 (6 GB VRAM) under WSL2. Never use more than 5.5 GB VRAM. The ML here is small (tabular, ~1-3k rows) and fits on the CPU.

### Repo layout
```
data/raw/          # optimof clone + CoRE MOF 2024 CSVs (gitignored)
data/processed/    # parquet feature/label tables
src/features/      # feature and label loading, MOFid grouping
src/models/        # stability models (water, thermal, solvent)
src/conformal/     # conformal recall control across the stacked filters
configs/           # run configs
notebooks/         # exploration only
scripts/           # CLI entry points
tests/             # pytest
notes/             # notes, audit results (stability_inventory.md), archive_week1.md
```

### Plan
- [x] Phase 0: project reset, environment `mof-stability` (imports and GPU verified), optimof clone and CoRE MOF 2024 CSVs (checksums verified).
- [x] Phase 1 (done; see `notes/stability_inventory.md`): **audit of the paper's ML** (`notes/stability_inventory.md`): inventory of the three stability tables, model summaries (algorithm, features, hyperparameters, split, metrics),
      retraining from the released data and comparison with the saved models, train/test leakage check, calibration and false-negative rates. Stop and report after this phase.
- [x] Phase 2 (done; see `notes/phase2.md`): group-aware splits (proxy groups: refcode family + DOI), honest retraining, per-stage and cascade conformal recall control.
- [x] Phase 3: conformal recall guarantee for the stacked filters; empirical coverage check on held-out groups.
- [x] Phase 4 (done; see `notes/phase4.md`): price of recall, limitations table, paper figures.
- [x] Independent review (`notes/review.md`) and Phase 5 (done; see `notes/phase5.md`): CSV-split retrain against the SI, claim corrections, leave-clusters-out validation, sensitivity of the 17, group-vs-random ranges over many splits.
- [x] Phase 6 (done; see `notes/phase6.md`): thermal-gap diagnosis against the SI, joint cascade score with CV+/jackknife+ calibration, MOFid (built from source) and fingerprint groups.
- **Status of the guarantee (read before quoting it):** the per-stage split-conformal recall rule gives mean recall >= 1 - alpha only if calibration and deployment MOFs are exchangeable; under leave-clusters-out shift it fell short (`notes/phase5.md` section 3), and the distance-aware variant tried did not fix that. The stacked (union-bound) guarantee for MOFs stable at all three stages is an assumption, **not proved**.
