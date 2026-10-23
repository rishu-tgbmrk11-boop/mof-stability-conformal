# Week 1 notes

## Environment checks (2026-10-21)
- `nvidia-smi` in WSL2: NVIDIA GeForce RTX 4050, 6141 MiB total (759 MiB used at idle), driver 592.82 (Windows) / 590.74, max supported CUDA 13.1.
- No `nvcc` / CUDA toolkit in WSL (not needed for PyTorch conda/pip builds).
- No mamba/conda/micromamba installed yet -> `environment.yml` written but NOT built.

## optimof inventory (clone of snurr-group/optimof, branch main, HEAD 75cbd13)
Location: `data/raw/optimof/` (gitignored).

**Bottom line: the ternary CO2/N2/H2O GCMC results for ~2,772 MOFs are NOT in this repo.**
OptiMOF is a MOF *stability* predictor (thermal breakdown T, solvent-removal, water stability). No column, file or
notebook mentions GCMC, CO2 uptake, selectivity or humidity. (Searched with grep, checked CSV headers.) The paper's
GCMC data must come from the paper's SI/Zenodo or be regenerated with gRASPA.

### data/ (25 MB, 14 files)
- `data/README.md`: features from `extract_all`; CSVs are processed from MOFSimplify.
- `thermal/thermal_all_data.csv` (3131 rows, 196 cols; Zeo++/RAC features), + split/test pkl
- `solvent/solvent_all_data.csv` (2179 rows, 197 cols; CoRE_name, refcode, doi, flag, ...), + split/test pkl
- `water_and_haz/data.csv` (1092 rows, 165 cols; MOF_name, water/acid/base/boiling labels), + split/test pkl
- `all_in_one.pkl`, `all_in_one_cols.pkl`, `mof_map.pkl`, `check_features.py`

### performance/ (448 KB, 9 PNGs)
Evaluation plots only: thermal (residuals, qq, actual_vs_predicted, residual_distribution), solvent and water_boost
(ROC, confusion matrix), water_rf (confusion matrix).

### model/ (48 MB, 10 files)
Pickled models: `thermal_model.pkl`, `solvent_model.pkl`, `water_rf_model.pkl`, hyperparameter JSONs, scalers in
`preprocess/`, and `saved_models/` (solvent 78% acc, thermal 52 MAE, water RF 66% acc).

### Other top-level
`app/` (Flask), `src/`, `zeo++-0.3/` (bundled Zeo++ source, useful for the Week 1 Zeo++ build), `taskfile.yml`, `environment.yml` (py<3.8 for feature extraction).

## Status
- [x] GPU check
- [ ] mamba env (blocked: no mamba installed)
- [x] optimof cloned + inventoried
- [x] repo layout

## Environment (built 2026-10-21)
- Miniforge in `~/miniforge3` (mamba 2.9.0); env `mof-humid` from `environment.yml` (versions pinned after the build).
- PyTorch 2.14.1 came out as the **cu130** wheel, not 12.x (`--extra-index-url` does not override a newer PyPI wheel). `torch.cuda.is_available()` = True,
  RTX 4050 Laptop GPU, 6140.5 MiB total, 5072 MiB free at the time (Windows desktop uses ~760 MiB).

## Zeo++ (zeo++-0.3 from optimof, `make`, g++ 13.3)
Binary: `data/raw/optimof/zeo++-0.3/network`. On `app/temp/ABAVIJ_clean.cif` (3.7 s wall):
Di 4.455, Df 2.497, Dif 4.407 A; density 1.525 g/cm3; ASA 0 (probe 1.65 A); 0 channels, 2 pockets (non-porous to this probe).

## gRASPA (commit f3d176c) on WSL2
- No nvc++ (HPC SDK) available -> built with nvcc 12.6 from a user-local conda env (`cuda-build`), `-arch=sm_89 -rdc=true --expt-relaxed-constexpr` (the flag is needed because
  `std::sqrt` is called in a kernel in VDW_Coulomb.cu). Build script: `scripts/graspa_wsl/build.sh`.
- The stock source **segfaults** on WSL2 (also on the shipped Henrys_coefficient example): `cudaDevAttrConcurrentManagedAccess = 0`, so the CPU may not touch managed
  memory (`Vars.Sims`) while a kernel runs. Fix applied to a copy (`~/src/gRASPA_wsl`): `cudaMallocManaged` -> `cudaMallocHost` for 3 allocations plus one `cudaDeviceSynchronize()`
  (`scripts/graspa_wsl/wsl2_managed_memory.patch`).
- Validation done: the shipped CO2_MgMOF74_UFF example's initialization energies (host-host Fourier 44714.15321, exclusion energies) match its committed `output.txt`. No full adsorption result was compared to a reference.

## Test system (NOT CALF-20)
CALF-20 CIF could not be obtained verified (and would lack partial charges). Used **Mg-MOF-74** (`Examples/Henrys_coefficient/MgMOF74_charged.cif`, charges from PACMOF), 0x5x3x3 unit cells,
313 K, 14 A cutoffs, Ewald 1e-6. Force field: UFF framework, TraPPE CO2 (27 K/2.80 A, 79 K/3.05 A, +0.70/-0.35), TIP4P water (78 K/3.154 A, +-0.52, M-site -1.04).
Inputs: `runs/week1/*/simulation.input` (gitignored). Wrapper: `scripts/run_graspa.sh`.

## Timings (RTX 4050, one run each, wall-clock incl. startup; GPU memory = nvidia-smi total used minus 759 MiB idle baseline, sampled at 1 Hz)
| run | length | wall (s) | GPU mem peak / delta (MiB) | result |
|---|---|---|---|---|
| Widom CO2 | 100,000 single-step cycles | 28.5 | 864 / +105 | Henry coeff 7.2e-5 +- 2.1e-5 mol/kg/Pa (short run) |
| Widom H2O (TIP4P) | 100,000 | 34.5 | 864 / +105 | Henry coeff 1.3e-3 +- 1.0e-3 mol/kg/Pa (short run, large error) |
| GCMC H2O, 5907 Pa | 100,000 init + 100,000 prod | 72.3 | 864 / +105 | **not converged**: loading still rising (block averages 6.9 -> 9.5 mol/kg) |

- 5907 Pa = 0.80 x 7384 Pa, where 7384 Pa is the approximate experimental saturation pressure of water at 313.15 K (from memory, not looked up; TIP4P's own
  saturation pressure differs). Fugacity coefficient set to 1.
- Runs are far too short for production numbers; they are timing/smoke tests. Per-MC-step cost is the usable number: roughly 0.7 ms/step for H2O GCMC (72.3 s / 200k steps), including startup.
- Not done: Mg-MOF-74 is not the target system, and no CALF-20 timing exists.

---
# Week 1, part 2 (CoRE MOF 2024, MOFX-DB, water saturation pressure)

## 1. CoRE MOF 2024 download (Zenodo 15055758, v1.1, 2025-03-20) -> `data/raw/coremof/` (393 MB extracted, gitignored)
Downloaded: 5 CSVs, `CoREMOF2024DB_SI_20250204.zip` (42 MiB), `water.zip` (12 MiB). md5 of both zips matched Zenodo. **Not** downloaded: `TSA.zip` (259 MiB), `mofid-v2.zip`, xlsx SI files.
- **This is the SI subset, not the full database**: ASR 1,372 rows, FSR 1,192, ION 100 (~2.7k structures with CIFs under `cifs_db/SI/CR/{ASR,FSR}` and `SI/NCR/*`). `12089-recommended-screening-list.csv` is only 12,089 ids (no properties); `ASR_FSR_check.csv` maps ASR_id <-> FSR_id (5,112 rows).
- ASR/FSR/ION CSV columns (57/56/57): number, coreid, refcode, name, mofid-v1, (mofid-v2: ASR+ION only), LCD, PLD, LFPD (A), Density, ASA/NASA (A2, m2/cm3, m2/g), PV/NAV (A3, VF, cm3/g), structure_dimension, topology (single/all nodes, + zeo variants), catenation, dimension_by_topo, hall, number_spacegroup, Metal Types, Has OMS, OMS Types, **Charge**, average_atomic_mass, heat capacity at 300/350/400 K (+std, k_cp, cp0), natoms, Source, DOI, Year, Time, Publication, Extension, unmodified, **Thermal_stability (C)**, **Solvent_stability**, **Water_stability**, **KH_Classes**, memo.
- Water / hydrophobicity / charges / stability:
  - `Charge` is just the method label (`PACMAN-DDEC6` for all 1,372 ASR rows); the actual **per-atom DDEC6 charges are in the CIFs** (`_atom_site_charge`; all 1,372 ASR CIFs have it). Directly usable for gRASPA.
  - `Water_stability`: fractional score 0.07-0.99 (numeric for 1,363/1,372 ASR, 1,166/1,192 FSR; rest `unknown`). `Thermal_stability (C)` and `Solvent_stability` likewise (`unknown` 9 ASR / 26 FSR). The values look like model outputs (non-round floats); I did **not** verify how they were produced - check the paper before relying on them as labels.
  - `KH_Classes` (water Henry class), ASR: weak 663, none 265, strong 256, superstrong_high_loading 106, unknown 68, superweak 7, strong_high_loading 4.
  - `water.zip`: `water_data.xlsx` (GEMC water isotherms per structure, sheets ASR 8,448 / FSR 6,641 / Ion 658 rows; columns include Pressure (Pa), Loading (molecules/supercell), Free_Energy_Transfer, K_H in several units, Sim_Status, KH_Classes) and `widom.xlsx` (H2O Henry constants, sheets ASR 8,857 / FSR 7,638 / Ion 710), plus 15.7k `.aif` isotherm files. Method (from the .aif headers): MCCCS-MN GEMC, **298 K**, UFF4MOF + TIP4P, cutoff 9 A, 5e3+1.5e4+2.5e4 cycles, pressures 0.1-1000 Pa. So water data cover only up to 1 kPa at 298 K, not 313 K / 80% RH, and no CO2 data are in the release.
  - Note the water xlsx covers more structures (~8.4k ASR) than the 1,372-row SI CSV.
- Env additions: openpyxl 3.1.5 (conda), iapws 1.5.5 (pip), pinned in `environment.yml`.

## 2. Dry-CO2 comparison against MOFX-DB: NOT DONE - premise does not hold
- MOFX-DB (mof.tech.northwestern.edu, API `/mofs.json`) databases: CoREMOF 2014, CoREMOF 2019, CSD, hMOF, IZA, PCOD-syn, Tobacco. **CoRE MOF 2024 is not among them.**
- CoREMOF 2019 entries in MOFX-DB carry **only N2 (77 K) and Ar (87 K)** isotherms (databases page lists only Argon/Nitrogen; page 1 of 61 checked: 200/200 records). A name lookup of ABAVIJ returns only N2/Ar for 2014 and 2019.
- The CO2-filtered query (`gases[]=CarbonDioxide`) returns 689 pages (~137.7k MOFs); pages 1, 200, 400, 689 are 100% hMOF (hypothetical MOFs, not CoRE). I did not scan every page, so "no CoRE MOF has CO2 data" is strongly suggested, not proven. (Note the `database` filter appears to be ignored when `gases[]` is present.)
- hMOF CO2 sims (600 sampled, all identical setup): RASPA, 298 K, `Forcefield CrystalGenerator`, TraPPE CO2, Ewald with **charges from a MOL file**, 12 A cutoff, P up to 2.5 bar, units mol/kg. The MOFX-DB CIFs have **no charges**, so these runs cannot be reproduced from MOFX-DB alone, and the framework FF is not plain UFF.
- Therefore no honest like-for-like comparison was run, and no numbers are reported. No GCMC was started.

## 3. Water saturation pressure at 313.15 K
- Experiment: **7384.9 Pa** (NIST Webbook, saturation properties of water, T=313.15 K). Cross-checks: IAPWS-IF97 7384.43 Pa, IAPWS-95 7384.94 Pa (iapws 1.5.5), Wagner-Pruss 2002 equation 7385.11 Pa. 80% RH = **5908 Pa**.
- My earlier remembered value (7384 Pa -> 5907 Pa used in `runs/week1/gcmc_h2o`) was correct to <0.02%; no rerun needed for that.
- TIP4P's own saturation pressure (original TIP4P, Horn, Swope & Pitera, J. Chem. Phys. 123, 194504 (2005), Table I; read via a summary, **not verified against the PDF**): 298 K simulated 0.047 +- 0.005 atm vs experiment 0.0310 atm (+52%); 373 K simulated 1.414 +- 0.101 atm vs 0.9956 atm (+42%). Boiling point of TIP4P 363.7 +- 5.1 K.
  **No 313 K value was found.** Rough extrapolation only (NOT a source value): the model overshoots experiment by ~40-50% over 298-373 K, which would put TIP4P's Psat(313 K) near 10-11 kPa.
- Consequence if that holds: running at 5908 Pa is about 55% of TIP4P's saturation pressure, not 80%. "80% RH" in the model needs P = 0.8 x Psat_TIP4P(313 K), which must be measured (e.g. gRASPA Gibbs-ensemble run, the NVT-Gibbs example) or taken from a real 313 K source. Decide which convention (same absolute vapour pressure vs same relative saturation) the project uses.

---
# Week 1, part 3 (paper methods, RASPA vs gRASPA, TIP4P saturation pressure)

## A. What the paper did (Liu et al., JACS 2026, 148, 34821; main text + SI read from the PDFs in Windows `Downloads`)
The path `/mnt/user-data/uploads/jacs_6c10566_2.pdf` does not exist in WSL. I used `/mnt/c/Users/HP/Downloads/jacs.6c10566.pdf` (3 copies, identical extracted text) and `ja6c10566_si_001.pdf` (3 identical copies). Page/section numbers refer to the SI.
- **Water model:** TIP4P (SI ref 13), CO2/N2 TraPPE, rigid, framework fixed. Ewald summation, LJ cutoff 12.8 A, Lorentz-Berthelot, tail corrections on adsorbate-adsorbate pairs only (minimizations); "all adsorbate-adsorbate" tail corrections in the mixture GCMC. RASPA 2.0.
- **How 80% RH was defined (SI Sec. 3):** "relative humidity is defined as the ratio of the water **fugacity** to the **saturation fugacity of the TIP4P model** at the simulation temperature, and the fugacity coefficient for water was fixed at unity." Saturation pressure at 313 K = **10,600 Pa**, "estimated by fitting the Antoine equation to TIP4P saturation pressure data reported by NIST" (NIST SAT-TMMC, TIP4P water (LRC), SI ref 15), treated as the saturation fugacity. At 298 K they used **4,596 Pa**, "taken directly from the literature" (SI ref 16).
  -> 80% RH at 313 K = 0.80 x 10,600 = **8,480 Pa** water; gas CO2:N2:H2O = 15 : 76.5 : 8.5 mol% at 1 bar total (8.5% of 1 bar = 8,500 Pa, consistent). CO2/N2 fugacity coefficients from Peng-Robinson.
- **Charges:** the 2,772-MOF humid GCMC screen used **PACMOF** charges, not DDEC (SI Sec. 5: "for the large-scale ... GCMC simulations at 80% relative humidity, partial charges were instead assigned using PACMOF"). DDEC (VASP-derived) was used for the CALF-20/Zn(imPim) force-field benchmarks and, in the main text, DDEC charges were computed for all 116 top MOFs and the mixture simulations repeated with DDEC + UFF: 80 retained CO2 > 1.5 mol/kg and selectivity > 50, plus one more at selectivity 48.4 (81 total, SI Table S6).
- **Convergence:** 10,000 init + 10,000 production cycles, then Gelman-Rubin factor on energy trajectories (J=5 blocks), extend by 10,000-cycle chunks until R <= 2.0; range 20,000-80,000 cycles. A cycle = max(20, N molecules) steps. Moves for the single-component/mixture isotherms: swap:translation:rotation:reinsertion = 2:1:1:1.
- **Candidate pressures to use for "80% RH" (decision still open):**
  1. **Paper's definition** (fugacity relative to TIP4P's own saturation fugacity): P_H2O = 0.80 x P_sat,TIP4P(313 K) = **8,480 Pa** (using their 10,600 Pa).
  2. **Experimental water:** 0.80 x 7,384.9 Pa = **5,908 Pa** (NIST). This is what my earlier Mg-MOF-74 test run used; it corresponds to only ~56% of TIP4P's saturation (5,908 / 10,600), so it is not what the paper means by 80% RH.
  A gRASPA-derived TIP4P P_sat for our own setup is reported in section C, for use in place of 10,600 Pa if you prefer self-consistency.
- Paper's data availability: the released OptiMOF repo has no GCMC data (see earlier notes); the SI has Table S6 (top 81 candidates under the mixture conditions).

## B. Dry-CO2 validation: RASPA2 (2.0.50, conda-forge env `raspa`) vs gRASPA (patched WSL build)
**System:** CoRE MOF 2024 ASR `2010[Cr][bpq]3[ASR]1` (MIL-53(Cr)-type, 36 atoms/cell, PLD 7.09 A, density 1.02 g/cm3), DDEC6 charges from the CIF (sum 6e-10, neutral),
3x5x5 cells (2,700 framework atoms, 34,816.65 g/mol per supercell). Note: the mu-OH hydrogens are absent in this CoRE structure (H8 per Cr2).
**Identical settings in both codes:** UFF (same 110-element table file), TraPPE CO2 (27 K/2.80 A, 79 K/3.05 A, +0.70/-0.35, 1.16 A), Lorentz-Berthelot, LJ truncated (not shifted) at 12.8 A,
**no tail corrections**, Ewald (precision 1e-6, real-space cutoff 12.8 A), 313 K, 15,000 Pa, fugacity coefficient fixed at 1, rigid framework, moves translation/rotation/reinsertion/swap 1:1:1:1,
cycle = max(20, N) steps in both. Not identical: RNGs, and CBMC details (gRASPA 10 trial positions/orientations; RASPA defaults). Inputs: `runs/week1/validation/*`, parser `scripts/parse_loading.py`.
**Protocol:** 5,000 init cycles + 20,000 production cycles, 5 blocks, several seeds per code (printed error bars are defined differently in the two codes, so SE below is recomputed from block values / seeds).

| code | runs (20k-cycle) | absolute loading, mol/kg | spread |
|---|---|---|---|
| RASPA2 | seeds 12, 13, 14: 0.4796, 0.4774, 0.4752 | **0.4774** | SE over seeds 0.0013; block SE per run 0.005-0.009 |
| gRASPA | seeds 12-18: 0.4760, 0.4683, 0.4669, 0.4718, 0.4746, 0.4781, 0.4714 | **0.4724** | SE over seeds 0.0015; block SE per run 0.002-0.006 |

**Difference gRASPA - RASPA = -0.0050 +- 0.0020 mol/kg (-1.0%)**, i.e. about 2.5 standard errors: small but probably real at this precision (not explained by noise alone).
(The first 5,000-cycle runs gave RASPA 0.4851 and gRASPA 0.4794; including them, 4-run RASPA mean 0.4793 and 8-run gRASPA mean 0.4733.) Mean loading is ~16.5 CO2 molecules per supercell, so statistics are
modest; the per-block scatter is consistent between codes and run to run.
Possible causes of the ~1% gap (not tested): Ewald parameter/kmax selection, CBMC trial settings, and gRASPA's treatment of overlap/charge grids. The gap is much smaller than the ~10% differences that would matter for the screening thresholds.
**Caveat on gRASPA seed 12:** its block averages were unusually tight (SE 0.0007 vs 0.002-0.006 for other seeds); its mean is within the range of the others. A repeat with the same seed (`graspa_s12_rerun`) reproduced all five block values exactly (0.4783, 0.4763, 0.4739, 0.4756, 0.4760), so gRASPA is deterministic per seed and the tight spread is a property of that random stream, not run-to-run noise. Cause of the unusually low block-to-block scatter not identified; it is reported alongside the other six seeds, not removed or singled out. (The rerun's 12,424 s wall time is the laptop sleeping, not compute time.)
**Timing (same system, 25,000 cycles):** RASPA2, one CPU core each, 3 runs in parallel: ~39 min wall (2343-2349 s). gRASPA on the RTX 4050: ~180-190 s uncontended (GPU delta +107 MiB); ~5,000+5,000 cycles 79 s. About 12x faster per run than one RASPA core.
(Caveat: this system has only ~17 molecules; the GPU advantage can differ for bigger loadings, e.g. water.)

## C. TIP4P saturation pressure by gRASPA Gibbs ensemble (RESULTS PENDING - runs in progress)
**Setup:** original TIP4P (eps 78 K, sigma 3.154 A, qH +0.52, M-site -1.04 at 0.15 A), NVT Gibbs, two cubic boxes built from a 1 A empty cell, Ewald (1e-6), LJ truncated at 12.8 A **with analytic tail corrections** (`tailcorrections yes`),
CBMC 10 trial positions/orientations, moves translation/rotation/reinsertion/Gibbs transfer (1:1:1:1) + Gibbs volume change (0.1), 5 blocks, 4,000 init + 12,000 production cycles (cycle = max(20,N) steps; ~1.3 s/cycle on the 4050).
- 313 K: liquid box 29 A start (700 molecules), vapor box 100 A (3 molecules at start), seed 31, `runs/week1/gibbs_tip4p/t313_s31`.
- 298 K (validation): liquid 29 A (700), vapor box 130 A (3), seed 32, `t298_s32`; started alongside the 313 K job on the same GPU (queue script stopped; 313 K job untouched).
- P_sat will be taken as <N_vapor> kT / <V_vapor> (ideal-gas estimate; the second-virial correction at ~4-10 kPa is of order -0.4%, i.e. smaller than the expected statistical error and not applied). Uncertainty from the 5 block averages.
- Hardware log (temperature, utilisation, memory, power, clock, throttle reasons, every 30 s): `runs/week1/gibbs_tip4p/gpu_monitor.log`, script `scripts/gpu_monitor.sh`.

**Reference values for comparison (what exactly is verified):**
- **NIST SAT-TMMC "TIP4P Water (LRC)"** (open page; LJ 10 A + analytic LRC, Ewald; 3 independent runs each): p_sat = 5.137e-2 bar +- 1.6e-4 at 300 K, 0.1960 bar +- 2.9e-4 at 325 K, 0.6004 bar at 350 K. No 298 K / 313 K rows, so interpolating ln p vs 1/T:
  **298 K -> 4,571-4,577 Pa** (extrapolating 2 K below the first NIST point), **313.15 K -> 10,582-10,670 Pa** (range from using the 300/325 vs 300/350 pairs). This agrees with the JACS SI values (4,596 Pa at 298 K; 10,600 Pa at 313 K from an Antoine fit to the same NIST data).
- **Horn, Swope, Pitera, J. Chem. Phys. 123, 194504 (2005): NOT verified against the PDF.** The paper is closed access (OpenAlex: not OA, no repository full text; AIP returns 403 to downloads) and I could not obtain it. From the PubMed abstract (verified text): TIP4P boiling point 363.7 +- 5.1 K "from simulations that employ finite range treatment of electrostatic and Lennard-Jones interactions" and TIP4P-Ew 370.3 +- 1.9 K; the abstract says free-energy results for TIP4P-Ew were analytically corrected before comparing to experiment. The abstract contains **no vapor-pressure numbers**.
  The figure "TIP4P 0.047 atm (4,762 Pa) at 298 K, 1.414 atm at 373 K" came from a model-written summary of the paper's Table I and is **unverified**; it is also unclear whether those are the paper's own TIP4P calculations or quoted literature values with another truncation scheme. It is ~4% above the NIST/JACS value (4,571-4,596 Pa), which is plausible for a different cutoff treatment but is not a confirmed comparison. Needs the PDF (Table I) from you.

### C.1 Incident: WSL restarted at 2026-10-22 06:45 UTC (laptop shutdown/restart) - both Gibbs runs killed
- The first 313 K run (started Oct 21 23:52) reached production cycle ~6,900/12,000 (last output flush 05:21 UTC); the 298 K run (started 05:02) was still at initialization cycle 0 in its log. GPU logger and queue were also lost. Earlier "stalls" (wall time >> CPU time) were laptop sleep; this one was a real restart (`uptime` 1 min, `last -x` boot 06:45).
- gRASPA does write `Restart/System_*/restartfile`, but `RestartFileParser` reads a hard-coded `RestartInitial/System_0/restartfile`, so a two-box Gibbs run cannot be resumed without patching the code; I did not do that. Both runs were relaunched from scratch with the same inputs and seeds (31, 32) at 06:50 UTC; old outputs kept in `t313_s31_killed/` and `t298_s32_killed/`. New runs use `scripts/graspa_linebuf.sh` (line-buffered output).
- GPU log up to the crash: max temperature 80 C; clock 2445-2475 MHz vs 2460 MHz nominal at ~76-80 C; throttle-reason bitmask occasionally `0x24` (software thermal slowdown + software power cap flags set), otherwise `0x1` (idle flag). Slowdown was a few MHz, i.e. negligible, but the GPU does run at its thermal limit with two jobs.

### C.2 Interrupted 313 K run: raw molecule counts only (no saturation pressure quoted, per agreed rule)
From the 69 production prints (every 100 cycles, cycles 0-6,800) of `t313_s31_killed/output.txt`: total molecules constant at 703; liquid box 701.6 on average;
vapor-box counts: 0 x20, 1 x23, 2 x14, 3 x3, 4 x6, 5 x1, 6 x2; <N_vapor> = 1.46, 5-block SE 0.20 (first half 1.59, second half 1.34); vapor box 99.9989 A cube (9.9997e5 A^3).
For comparison, the literature reference pressure (NIST-derived, section C) would correspond to <N_vapor> ~ 2.45 in this box, so this partial run sits ~5 SE below it. Cause not established: a real bias in the Gibbs setup
(tail-correction handling, 12.8 A truncation vs NIST's 10 A + LRC, finite-size effects with ~1.5 vapor molecules) or too few effectively independent samples (69 strongly correlated prints). Resolved only by the full reruns and the Widom cross-check (C.3).
No saturation pressure from our own simulations is stated in these notes until the Widom check and the Gibbs reruns agree within error.

### C.3 Liquid-Widom cross-check of the TIP4P saturation pressure (started 2026-10-22; results pending)
**Method.** In the liquid at the same temperature, mu_l = mu_v. With an ideal-gas vapor, rho_v = rho_l / <W>, where <W> = <exp(-beta dU)> is the Widom (test-particle) Rosenbluth weight of inserting one TIP4P molecule
into the liquid (mu_ex = -kT ln<W>), so **P_sat ~ rho_l kT / <W>** (second-virial correction of order -0.4% at these pressures neglected; liquid at 10.6 kPa instead of exactly P_sat is a negligible difference).
gRASPA reports <W> and mu_ex directly ("Averaged Rosenbluth Weight", "Averaged Excess Mu" in K), and the NPT average volume (needed for rho_l = N/<V>) in "BLOCK AVERAGES (VOLUME)".
**Code check (by reading the source):** `WidomMove` runs the same `Insertion_Body` as real insertions, which applies the Ewald correction and the tail correction (`TailCorrectionDifference`) to the Rosenbluth weight, so <W> includes both. The same routine is used in the Gibbs transfers.
**Setup:** 700 TIP4P in a cubic box (29 A start, NPT at 10,600 Pa and 313 K, volume-move probability 0.25), UFF-style tables as above, 12.8 A cutoff, Ewald 1e-6, tail corrections on, CBMC Widom_Trials = Widom_Orientation = 10, Widom probability 10 relative to 1:1:1 for translation/rotation/reinsertion
(~75% of steps are insertions, ~530 per cycle), 6,000 NPT initialization cycles, 12,000 production cycles, 5 blocks, **three independent seeds (51, 52, 53)**: `runs/week1/widom_tip4p/w313_s{51,52,53}`. Guard: `scripts/gpu_guard.sh` pauses only these three processes if GPU > 85 C (events: `widom_tip4p/guard_events.log`); GPU log: `gibbs_tip4p/gpu_monitor.log`.
**Known issue seen in a 600-cycle test:** the liquid started at 29 A (0.86 g/cm3) and the box first expanded to ~27,400 A^3 (0.77 g/cm3) and was still drifting down, i.e. the initial configuration needs thousands of cycles to relax; production blocks will be checked for volume drift before use, and drifting blocks will be reported, not silently included.
**Plan for later Gibbs runs (not the two currently running):** use a larger vapor box (current ~1.0e6 A^3 holds only ~1.5-2.5 molecules), and split production into several independent runs with different seeds so the error bars can be checked against between-run scatter.

### C.4 Live monitoring tools (read-only; added 2026-10-22)
- `scripts/watch_runs.sh [once]`: terminal status of GPU, processes and last printed cycle of the five runs.
- `scripts/dashboard/server.py` + `index.html`: local web dashboard (stdlib only, bound to 127.0.0.1:8765, opened at http://localhost:8765 from Windows). Shows job progress/ETAs, GPU temperature/utilisation/memory/throttle flags with the 85 C guard line,
  Widom total-energy curves, Gibbs running-mean vapor-box occupancy, and a values table. It only reads the run logs, `ps` and the GPU log; it writes nothing to run folders and signals no process. Refreshes every 15 s.
  Start: `~/miniforge3/envs/mof-humid/bin/python -I scripts/dashboard/server.py 8765`. Stop: `pkill -f dashboard/server.py`. The page computes no saturation pressure.
- `scripts/dashboard/molecules.html` (served at http://localhost:8765/molecules): 3D viewer (own canvas renderer, drag to rotate, scroll to zoom) of the LAMMPS-format snapshots gRASPA writes to `Movies/System_*/result_<cycle>.data`.
  Production runs write a frame only every 5,000 cycles (default `MoviesEvery`), so those are separate configurations, not an animation. For motion there is a separate **visual-only demo run** (`runs/week1/viz_demo`: 100 TIP4P, 15 A box, 7 A cutoff, 313 K,
  `MoviesEvery 2`); it is Monte Carlo (trial moves), not molecular dynamics, has no time axis, is too small for any science, and feeds no results. Sanity check of the parsing: O-H distance in a 313 K snapshot molecule = 0.957 A (TIP4P: 0.9572 A).
- `scripts/dashboard/mof.html` (http://localhost:8765/mof, optional `?run=graspa_s12`): viewer for the MIL-53(Cr)-type MOF + CO2 frames saved by the **finished** gRASPA validation runs (`validation/graspa*/Movies/System_0/result_<cycle>.data`, frames at production cycles 0/5000/10000/15000).
  Handles the triclinic 3x5x5 cell (33.6 x 55.5 x 30.4 A with tilt), wraps atoms into the cell, finds bonds by covalent radii (x1.15, minimum image), draws bonds crossing the cell as two half-bonds. The last saved frame is production cycle 15,000, not the very last cycle (frames every 5,000 cycles).
  No MOF job is running in the background: the long jobs are pure-water boxes (Gibbs + Widom).

### C.5 Crash 2026-10-22 17:16 UTC and resume (laptop shut down hard with the lid closed)
- All five production runs (9,100-9,500 of 12,000 production cycles) were killed when the laptop went down at ~17:16 UTC (Windows: Kernel-Power 41 / EventLog 6008 "unexpected shutdown"; boot 17:39 UTC). Circumstantial cause: GPU temperature rose 62 -> 80 C in the 7 minutes before the log ends, after the lid was closed.
  Nothing was lost on disk (backup: `runs/week1/backup_interrupted_runs_2026-10-22.tar.gz`), but the Widom block statistics (weights, volumes) are written only at the end of a run, so the three Widom runs' results were lost.
- gRASPA saves restart files only every ~5,000 cycles, so resumed runs start from production cycle ~5,000 of each original run, not from the crash point. The stock code cannot resume a two-box (Gibbs) run (the restart reader is only called in single-simulation mode and has `System_0` hard-coded);
  a small patch (`~/src/gRASPA_wsl_restart`, binary `nvcc_restart.x`) reads `RestartInitial/System_<k>/restartfile` for every box. The restart reader does not restore the box size, so `scripts/make_resume.py` writes a cell file per box with the edge from the restart file.
- Validation before launching: restored molecule counts and box volumes; initial energy recomputed by the code (-2,683,798.9) vs the original run's printed value at cycle 5,000 (-2,683,797.9), relative difference 4e-7; energy drift between running and recomputed energy = 0 in every component over 100 cycles for the Gibbs (both boxes) and Widom resumes.
  The cycle-0 printed line of a resumed run shows a different "Total Energy" number than the recomputed one (cosmetic print offset, not understood), so dashboard energy curves jump at the restart.
- Resumed runs (launched ~18:03 UTC): Gibbs 313 K and 298 K: 7,000 further production cycles from the saved state (new seeds 131, 132); Widom: 4,000 further NPT initialization cycles (to recover the equilibration lost since the restart point, the liquid was still densifying) + 8,000 production cycles, seeds 151-153.
  `analyze_final.py` stitches the original Gibbs samples before cycle 5,000 to the resumed samples. Guard: `scripts/gpu_guard2.sh` pauses ALL five runs above 75 C and resumes them below 68 C (events: `runs/week1/guard2_events.log`).

### C.6 Crashes, clean-up and lessons for later runs (2026-10-23)
**Status of the saturation-pressure question: still open.** No saturation pressure from our own simulations is reported anywhere in these notes; the Gibbs-ensemble and Widom runs never finished, and the Widom statistics (weights, volumes) are only written at the end of a run, so none were saved.
Reference values for TIP4P remain the literature ones in section C (paper: 10,600 Pa at 313 K and 4,596 Pa at 298 K; NIST SAT-TMMC interpolation 10,582-10,670 Pa and 4,571-4,577 Pa).

**Cause of the crashes: loss of battery power, not software (this replaces the "circumstantial cause: GPU temperature" remark in C.5).** Both times the laptop powered off hard and every run died with it.
- **No software error:** the stderr of the crashed runs contains only the harmless "Cannot Open Restart/System_k/restartfile" start-up lines, and the outputs simply end mid-run.
- **Windows System log, crash 1:** "Power source change" (Kernel-Power 105) at 2026-10-22 10:46:09 IST; Windows' last-alive stamp 10:47:04 IST; at the next boot, 12:12:54 IST, Kernel-Power 41: "The system has rebooted without cleanly shutting down first. This error could be caused if the system stopped responding, crashed, or lost power unexpectedly",
  and EventLog 6008: "The previous system shutdown at 10:47:04 AM on 10/22/2026 was unexpected." WSL boot record: 06:45 UTC (12:15 IST).
- **Windows System log, crash 2:** last-alive stamp 22:38:07 IST (17:08 UTC); "Power source change" at 22:39:14 IST (17:09:14 UTC); at the next boot, 23:09:01 IST (17:39 UTC), Kernel-Power 41 (same text) and EventLog 6008: "The previous system shutdown at 10:38:07 PM on 10/22/2026 was unexpected."
  WSL boot record: 17:47 UTC. (The Windows event log does not state the direction of a power-source change; the "battery loss" reading is the user's report, supported by these events.)
- **GPU heating before the last crash (`gibbs_tip4p/gpu_monitor.log`, kept):** 62-63 C for the previous 10 hours with the lid open; after the lid was closed it read 62 C at 17:09:34 UTC, 72 C at 17:12:34, 78 C at 17:15:24 and 80 C at 17:16:23 UTC, the last sample, with the software power-cap/thermal-slowdown flags set (0x24) and the clock at 2,445 MHz (nominal 2,460).
  The log ends there. Whether the heating contributed is not determined; closed-lid heating and the power-source change coincide in time.

**Clean-up (approved by the user):** deleted 17 simulation folders under `runs/week1/`: `gibbs_tip4p/{t313_s31, t313_s31_killed, t313_s31_resume, t298_s32, t298_s32_killed, t298_s32_resume, t313_test}` and `widom_tip4p/{w313_s51, w313_s52, w313_s53, w313_s5{1,2,3}_resume, w313_s5{1,2,3}_resume_aborted, test}`.
**Kept:** `runs/week1/backup_interrupted_runs_2026-10-22.tar.gz` (3.5 MB; now the only copy of the interrupted Gibbs/Widom outputs, their Restart files and Movies frames; the `_resume` folders were not in it), the small `*.log` files and `gibbs_tip4p/gpu_monitor.log`,
`~/src/gRASPA_wsl` (+ patch and build script in `scripts/graspa_wsl/`), `~/src/gRASPA_wsl_restart`, `data/raw/`, the Mg-MOF-74 timing outputs, `validation/`, `viz_demo/`, `environment.yml`. The dashboard server was stopped. `make_resume.py` needs the original run folders, so resuming those exact runs now requires first restoring them from the tarball.

**Lessons for later runs**
1. **Short resumable segments.** gRASPA writes restart files only about every 5,000 cycles and the Widom block statistics only at the end, so a crash cost up to ~4 hours per run and all of the Widom statistics. Run in segments of ~1,000-2,000 cycles, each ending with a restart file and its own saved statistics,
   and chain them with `RestartFile yes` and the patched reader in `~/src/gRASPA_wsl_restart` (stock gRASPA cannot resume a two-box run; the reader also does not restore box size, see `scripts/make_resume.py`). Analyse segments separately so partial work is never lost.
2. **A larger vapor box.** The Gibbs vapor boxes (100 A at 313 K, 130 A at 298 K) held only ~1-2.5 molecules at the target pressure, giving very noisy counts (0-6 molecules; strongly autocorrelated). Size the vapor box for several to ~10 molecules and run several independent seeds so the error bars can be checked between runs.
3. Run on mains power with the charger connected, lid open (or on a stand); stop the runs before the laptop leaves power. Give the Widom liquid a longer NPT equilibration (its energy and volume were still drifting after 11,000-15,000 cycles) and treat early production blocks as unsettled.
