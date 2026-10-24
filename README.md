# Project: Recall-guaranteed stacked stability filters for MOFs

## Goal
1. **Reproduce** the stability-model machine learning of Liu et al., "Identification of Metal-Organic Frameworks for CO2 Capture from
   Humid Flue Gas: Integrating Molecular Simulation, Machine Learning, and Experimental Synthesis", J. Am. Chem. Soc. 2026, 148, 34821-34829
   (doi:10.1021/jacs.6c10566), using the released data and models of the OptiMOF repository (github.com/snurr-group/optimof) and the
   CoRE MOF 2024 dataset (Zenodo record 15055758). The paper's ML screens candidate MOFs for **water stability, thermal stability and
   solvent-removal (activation) stability**.
2. **Add a statistical guarantee:** apply conformal (risk-controlling) calibration across the **stacked** water, thermal and solvent filters, so
   that the combined filter provably keeps at least a chosen fraction of truly stable MOFs (a recall guarantee), instead of the guarantee
   holding only for each filter in isolation.

## Where things come from (verify before relying on any of it)
- OptiMOF repo (`data/raw/optimof/`): stability tables under `data/` (thermal, solvent, water_and_haz), saved models under `model/`, plots under `performance/`.
  The repo has **no** CO2/H2O adsorption (GCMC) data.
- CoRE MOF 2024 (`data/raw/coremof/`): the CSVs from Zenodo 15055758 only. **Do not download TSA.zip or mofid-v2.zip** (large). The CSVs have MOF ids,
  MOFid strings, geometry descriptors and `Water_stability`, `Thermal_stability`, `Solvent_stability` columns whose origin must be checked, not assumed.
- Earlier work (an abandoned, larger-scope attempt) is archived in `notes/archive_week1.md`. Treat its statements as history, not as verified facts of this project.

## Rules (IMPORTANT)
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

## Tech stack
- Environment: mamba from `~/miniforge3`, env `mof-stability`, Python 3.11, pinned in `environment.yml`.
- ML: PyTorch (CUDA), scikit-learn, XGBoost, MAPIE (conformal prediction); pandas + pyarrow (parquet tables); pymatgen and ASE for structures.
- Hardware: laptop with an NVIDIA RTX 4050 (6 GB VRAM) under WSL2. Never use more than 5.5 GB VRAM. The ML here is small (tabular, ~1-3k rows) and fits on the CPU.

## Repo layout
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

## Plan
- [x] Phase 0: project reset, environment `mof-stability` (imports and GPU verified), optimof clone and CoRE MOF 2024 CSVs (checksums verified).
- [x] Phase 1 (done; see `notes/stability_inventory.md`): **audit of the paper's ML** (`notes/stability_inventory.md`): inventory of the three stability tables, model summaries (algorithm, features, hyperparameters, split, metrics),
      retraining from the released data and comparison with the saved models, train/test leakage check, calibration and false-negative rates. Stop and report after this phase.
- [x] Phase 2 (done; see `notes/phase2.md`): group-aware splits (proxy groups: refcode family + DOI), honest retraining, per-stage and cascade conformal recall control.
- [x] Phase 3: conformal recall guarantee for the stacked filters; empirical coverage check on held-out groups.
- [x] Phase 4 (done; see `notes/phase4.md`): price of recall, limitations table, paper figures.
- [x] Independent review (`notes/review.md`) and Phase 5 (done; see `notes/phase5.md`): CSV-split retrain against the SI, claim corrections, leave-clusters-out validation, sensitivity of the 17, group-vs-random ranges over many splits.
- **Status of the guarantee (read before quoting it):** the per-stage split-conformal recall rule gives mean recall >= 1 - alpha only if calibration and deployment MOFs are exchangeable; under leave-clusters-out shift it fell short (`notes/phase5.md` section 3), and the distance-aware variant tried did not fix that. The stacked (union-bound) guarantee for MOFs stable at all three stages is an assumption, **not proved**.
