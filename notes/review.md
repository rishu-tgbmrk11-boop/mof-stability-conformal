# Independent review of Phases 1-4

Reviewer stance: I read `README.md` and `notes/stability_inventory.md`, `phase2.md`, `phase3.md`, `phase4.md` (and the project code only where a check needed it), then tried to break the results. **Nothing was fixed.** All checks are scripts in `notes/review_scripts/` (run from the repo root as `python -I notes/review_scripts/<file>` in the `mof-stability` env); the output quoted below was produced by those scripts in this session. Seeds are in each script.

Independence: checks 1, 3, 4, 6 and the re-derivations use only the raw released CSVs (`data/raw/optimof/data/`) and my own code. Where I read a saved project file (`groups.parquet`, `pred_*.parquet`, `cascade_oof_rep0.parquet`) it was used as **data** to compare against, never through the project's functions. The only project import is `src.conformal.recall` in check 2, where the point was to test that code (and I first confirmed it equals my own 6-line implementation on 36 cases).

## Verdict in one paragraph

The numbers I could re-derive reproduce (group-vs-random metrics, groups, 92/17 counts, the early-stopping bug, the Phase 4 arithmetic), and the plain split-conformal rule is correct. The weaknesses are not arithmetic errors but **what the guarantee covers**: (1) the audit contains a factual error about the paper's test sets that points to a likelier explanation of the paper-vs-release gap; (2) the "empirical coverage check" cannot detect the failure mode that matters (shift to new families/sources); (3) the cascade guarantee is stated for the all-three-stable population but is only justified for each stage's own positives, and the little data there is points the wrong way.

## Findings, ranked

### HIGH

**H1. The notes say the SI's test-set counts cannot be obtained from the release. They can, from the CSVs.**
`stability_inventory.md` section 2 and finding 4 say the SI's test class counts (water 17/68/112/22; solvent 275 stable of 436) "cannot be obtained from the released split" and conclude the paper's models used "a different data version/seed than the release, or the SI describes a different model". A plain `train_test_split(df, test_size=0.2, random_state=0)` on the **released CSV rows** gives exactly the SI counts:

```
solvent test n 436 stable 275 unstable 161 | SI: 275 stable of 436
thermal test n 627 stable(>=350) 343
water (split of data.csv rows) n 219 {1: 17, 2: 68, 3: 112, 4: 22} | SI 17/68/112/22
solvent stable count in a random 436-MOF test set over 200 seeds: mean 257.9 sd 9.4; P(count>=275) = 0.030
```
(`r5_si_counts.py`; the SI counts are as recorded in `stability_inventory.md`, I could not re-open the SI, see check 5.) The chance of hitting 275 by luck is about 3%, and the water counts match all four classes, so this is not a coincidence. The counts in the notes (263 stable; 25/65/115/14) come from the split of the **released `*_split_data.pkl` / saved forest frames**, which differ from a split of the CSV rows (the audit found that the merge in `data_selection.py` substitutes some features; I did not examine row order, so why the splits differ is not established). So the likely story is: the SI's metrics come from a seed-0 split of the CSV rows, while the released `*_split_data.pkl` and saved models use another split. Consequences: (i) the claim "cannot be reproduced from the release" is wrong as stated; (ii) the 'paper vs release' performance gap (thermal R2 0.41 vs 0.138/0.278; solvent AUC 0.826 vs 0.830) was never tested against a CSV-split retrain; (iii) note that models trained on the pkl-split train part would have seen many CSV-split test MOFs, so evaluating the released models on the CSV-split test set is not a valid test either. I did not retrain, so I do **not** claim the SI metrics are reproducible, only that the stated impossibility is false.

**H2. The "empirical coverage check" cannot detect the failure that matters.**
`validity_check` (and `phase2.md`/`phase3.md` 'Do the curves track the target?') repeatedly re-partitions the pooled held-out groups into calibration/test at random with scores fixed. Random re-partition makes calibration and test exchangeable **by construction**, so mean recall ~ 1-alpha is a restatement of the k/(n+1) arithmetic (confirmed in my simulation, S1: mean recall within 0.0005 of theory). It cannot detect shift between calibration and deployment (new families, new data sources, later years). Simulation S6: if positives in the deployment set score 0.3 sd lower than calibration positives, mean recall is 0.838 at target 0.90 and 0.707 at target 0.80. The only real held-out evidence is one calibration-to-test draw per stage (Fig. 2); there the water stage sits below the diagonal for both model sets at most targets (Phase 3 water, target 0.80: realised 0.705), and the calibration and test positive scores differ detectably for Phase 2 water (KS p = 0.033, medians 0.702 vs 0.655) and Phase 3 thermal (p = 0.023); the other four of six comparisons are not significant (p = 0.48, 0.48, 0.94, 0.10; `r7_assumptions.py`). Six tests, so this is suggestive, not proof. Net: the Phase 3 plan item "empirical coverage check on held-out groups" is met only in the weak sense of one draw per stage; the guarantee is **untested against a real shift** (the CoRE MOF 2024 tables would be a natural source of such a test, but they share no MOFs with the optimof tables: overlap of refcodes is 0, I checked).

**H3. The cascade guarantee is not proven for the population it is claimed for.**
`phase2.md` section 4 flags the assumption, but the headline (README.md goal; Fig. 4; "union bound gives recall >= 1 - alpha for MOFs that are stable at all three stages") needs more than the flag. Each stage's conformal guarantee is for **that stage's positives**; it is not a statement about the subset that is also stable at the other two stages, and nothing implies it. If all-three-stable MOFs are 'harder' for a stage, per-stage coverage on that subset drops. Evidence from the data (`r7_assumptions.py`, out-of-fold scores, rep 0, n = 17):

```
[phase2] thermal median score: all-stable positives 379.5 vs other stage positives 429.6
[phase2] water   median score: all-stable positives 0.590 vs other stage positives 0.698
[phase2] solvent alpha_k=0.1: stage recall among the 17 all-stable = 0.824 | among the other 49 stage-stable = 0.837
```
The 17 score lower than the other stage positives in thermal and water (Phase 3: 360 vs 394, 0.604 vs 0.687). With n = 17 this is not significant, but the direction is the dangerous one. Simulation S7(c): if the all-stable positives are 0.5 sd 'harder' in every stage, the union-bound cascade's recall is 0.779 at target 0.90 and 0.488 at target 0.70 (independent stages and perfect dependence give 0.914/0.956 at target 0.90, so the bound is conservative only when exchangeability carries over). The observed recall 0.94-0.98 on the 17 is consistent with the guarantee but cannot confirm it (the notes say so correctly); my point is that the **claim of a proof for this population is the stronger one**, and the data cannot rescue it.

### MEDIUM

**M1. Proxy groups leave measurable residual leakage, and the notes say "not measured".**
Check 3 passed: my own connected components over refcode[:4] and DOI give an identical partition (2,777 groups, bijection with `groups.parquet`); no group spans train/val/cal/test in `groups.parquet` or in any `pred_group_*_s0.parquet`; no DOI whitespace/case variants. But (`r3_groups.py`, `r3b_nearest.py`):
- 181 of 4,963 names are not CSD refcodes (`UiO-66-Br`, `Zn-MOF-74`, `HKUST-1`, ...) and are grouped by their first four characters; the same material under a refcode in one table and a common name in another is not linked.
- 960 of the 2,777 groups merge several distinct 6-letter refcode families (more conservative than needed); with 6-letter families + DOI there would be 3,757 groups. So the 4-letter key over-merges for most refcodes but cannot link renamed duplicates.
- Near-duplicates across groups remain. Test MOFs with a train MOF in a different group closer than 0.5 (z-scored features): thermal 42/589, solvent 10/425, water 7/232. Their nearest-neighbour label agrees 0.81 / 1.00 / 0.71 against 0.67 / 0.64 / 0.66 for all test MOFs, i.e. these pairs carry label information. Small share, but `phase4.md` limitations say the effect "is not measured". Under the random split 24-39% of test MOFs have their nearest train neighbour in the same group (0 under the group split).
README.md's rule is 'no MOF, and no duplicate or near-duplicate of one, on both sides of a split'; the proxy meets the letter for exact groups and not for near-duplicates.

**M2. The group-vs-random gap is robust in sign, much less in size; the notes quote one split.**
Check 1 reproduces the notes (water, RF, saved splits, 5 seeds: group AUC_stable 0.624 +- 0.013 vs notes 0.628 / seed-0 0.641; accuracy 0.592 vs 0.600; random 0.830 vs 0.830; acc 0.726 vs 0.718). Over 30 independent splits of my own (RF classifier, AUC for stable, 60/20/20):

```
thermal  AUC random 0.847±0.012  group 0.773±0.028  gap +0.074 (gap>0 in 30/30)
solvent  AUC random 0.835±0.019  group 0.610±0.047  gap +0.226 (gap>0 in 30/30)
water    AUC random 0.801±0.025  group 0.695±0.036  gap +0.106 (gap>0 in 30/30)
random-split test MOFs whose proxy group also appears in train: 0.527
```
Gaps differ from the notes' single split (water 0.832 to 0.641 = 0.19; thermal 0.777 to 0.754 = 0.02; solvent 0.764 to 0.634 = 0.13) by up to a factor of 3 either way, because the split sd is 0.03-0.05. The qualitative conclusion stands; the specific figures in `phase2.md` 'Reading' and Fig. 1 are single-split draws (the notes say 'one split' in the limitations; the 'Reading' paragraph does not). My model is a random forest, not the released networks, so this tests robustness of the gap, not the networks' exact numbers.

**M3. Within-group dependence makes the point-level rule slightly under-cover; the notes call the guarantee a group-level one.**
`recall.py` and `phase2.md` say exchangeability 'is at the group level'. The rule is the point-level k-th order statistic, which is exact only for exchangeable **points**. With clustered positives the mean recall falls below nominal (`r2b_cluster.py`, 5000 reps, exact population recall; marginal of each positive N(1,1)):

```
40x4 s=0.0 alpha=0.1: 0.9004 ±0.0003  target 0.9
40x4 s=0.7 alpha=0.1: 0.8975 ±0.0004  target 0.9
15x4 s=0.9 alpha=0.1: 0.8890 ±0.0009  target 0.9   (theory for iid: 0.9016)
15x8 s=0.9 alpha=0.2: 0.7894 ±0.0011  target 0.8
```
(s = sd of the shared group effect; G x m = groups x positives per group.) The shortfall is up to ~1.4 points in extreme clustering and negligible at s = 0, and the real groups are mostly singletons or pairs (1,753 of 2,777 groups have one MOF), so I expect it to be small here, but I did not measure that. It is the opposite sign to 'ties only make it conservative'. The notes' 'mean realised recall within 0.003-0.005 of the target' is consistent with this.

**M4. The PAC 'failure' is mostly test-set noise; the notes' diagnosis lists the wrong main cause.**
In the exactly-iid case with the real stage sizes, `conformal_threshold_pac` reaches the target in the population in 89.4% / 90.4% / 91.7% of calibration draws (thermal/solvent/water, nominal 90%), so the Beta rule is right. Measured on a finite test set (as the notes' 'partitions >= target' does) it reaches the target in only 82.7% / 83.9% / 82.9%, from test noise alone (`r2c_pac_finite.py`; plain split conformal: 55.1% / 57.7% / 52.3%). The notes' 77.6% / 71.0% / 78.8% (alpha 0.10, group-aware) are therefore only a little below the iid expectation for thermal and water (~5 points) and further for solvent (~13 points), and the 500 partitions overlap, so their effective n is small. The notes correctly say 'unverified'; the stated causes (finite test and group dependence) are right but the **check as designed cannot give 90% even for a correct rule**, which is not said. A correct check measures population-style recall (or uses a much larger test set).

**M5. The count '17 stable at all three stages' depends on definitions the review could not check against the paper.**
Counts reproduce (92 MOFs labelled in all three tables; 17 stable under thermal >= 350 C, solvent flag > 0, water raw label >= 3; `r3_groups.py`). But the number is sensitive to the stability definition: water raw label >= 4 only gives 4; thermal >= 300 C gives 23, >= 400 C gives 9. The water definition (classes 2 and 3) and the 350 C cutoff come from the SI via the earlier audit; I could not re-open the SI (check 5). Every cascade statement rests on those 17.

**M6. Phase 4 numbers reproduce, but not to the third decimal.**
Analytic claims reproduce (kept at recall 0.95, prevalence 0.5247: 0.887 / 0.819 / 0.704 / 0.616 at AUC 0.70/0.80/0.90/0.95 against notes 0.89/0.82/0.70/0.62). Real pooled values from my own re-derivation (`r8_price.py`, pooled cal+test, seed-0, kept at recall 0.80/0.90/0.95): thermal Phase 3 0.585/0.728/0.842 (notes 0.582/0.726/0.840), water Phase 3 0.732/0.834/0.919 (notes 0.725/0.832/0.911), solvent Phase 3 0.740/0.866/0.923 (0.738/0.865/0.922). Differences are 0.001-0.008, always in one direction (mine higher), which suggests a small off-by-one in the threshold index in `p4_01` (e.g. `floor` vs `ceil`), not a bug that changes any conclusion. AUCs reproduce to 3 decimals (0.776, 0.658, 0.665; Phase 2: 0.722, 0.639, 0.685).

### LOW / confirmations

- **L1 (check 4, reproduced).** The released early-stopping restore is a no-op: `best_model = self.model.state_dict()` stores live references (`thermal_model.py:253`, `solvent_model.py:273`). 20-line toy (`r4_earlystop.py`):
  ```
  deepcopy=False: best val MSE 0.3314 at epoch 10, stopped at 40; val MSE after 'load best' = 0.4145  -> restore was a NO-OP
  deepcopy=True : best val MSE 0.3314 at epoch 10, stopped at 40; val MSE after 'load best' = 0.3314  -> restore WORKED
  ```
  This confirms the mechanism, not the size of its effect on the paper's models. Note the Phase 2 'baseline' networks use the **fixed** restore, so they are not the paper's models; the notes disclose this (the audit's 0.820 vs 0.762 solvent AUC).
- **L2.** Bare `pytest` fails at collection because of vendored test files under `data/raw/` (now noted in README.md). `pytest tests` passes (28).
- **L3.** The notes record the synthetic-score seed only in code (`RandomState(2026)` in `p4_01`), not in `phase4.md`; README.md asks for seeds recorded with every result.
- **L4.** No evidence of test-set tuning in Phase 3: `p3_01_select.py` asserts that only train rows are used. I could not check the order in which things were run, and note that the decision to try tree models was itself motivated by the Phase 2 test results (a mild, standard garden-of-forking-paths concern).
- **L5.** CoRE MOF 2024 CSVs carry `mofid-v1` strings and DOIs, but none of the 4,963 optimof MOFs appear in them (refcode and name overlap are both 0; they are keyed by different ids), so they offer no MOFid for these MOFs. The notes are right that MOFid groups are not available.

## Check 5: SI Table S5 and the thermal R2 0.41 claim

**Not done.** `/mnt/user-data/uploads/jacs_6c10566_2.pdf` does not exist in this environment (`/mnt/user-data` is absent). I found PDFs with similar names in `/mnt/c/Users/HP/Downloads/` but did not open them, since you pointed me at a different file; say if I should use them. Please check by hand:

| item | what the notes recorded for the SI | released-code / my value | note |
|---|---|---|---|
| thermal R2 | 0.41 | 0.138 (`model/`), 0.278 (`saved_models/`), 0.227 (retrained, released recipe, genuine restore), 0.178 (as coded) | the SI figure is for a test set of 627 MOFs; the CSV seed-0 split also gives 627 test MOFs (343 stable at >= 350 C), see H1 |
| thermal MAE / RMSE | 47 / 67 | 53.57 / 81.30 (`model/`), 52.70 / 74.42 (`saved_models/`) | |
| solvent accuracy / AUC | 0.775 / 0.826 | 0.778 / 0.830 (`model/`), 0.775 / 0.831 (`saved_models/`) | |
| solvent TN/FP/FN/TP | 110/51/47/228 | 120/53/44/219 (`model/`), 123/50/48/215 (`saved_models/`) | SI: 275 stable of 436, matches a CSV seed-0 split (H1) |
| water test class counts | 17/68/112/22 | 25/65/115/14 on the pkl-split frames; **17/68/112/22 on a CSV seed-0 split** | H1 |
| water accuracy | 0.671 | 0.662 | |
| Table S5 hyperparameters, thermal | 190 inputs, hidden 128-512-2048-256-1024, ReLU, dropout 0.124, lr 0.002, 100 epochs | released: 170 inputs, 2048-1024-512-256-128-64-32, leaky-ReLU, no dropout, lr 1e-4 | check that the SI table is for the model in the paper, not the repo |
| Table S5, solvent | 190 inputs, 128-256-1024-2048-128-256, ReLU, dropout 0.025, lr 7.64e-5, 100 epochs | released: 170 inputs, 512-256-128-64-32-16, dropout 0.2, lr 1e-3 | |
| Table S5, water | 179 inputs, 250 trees, depth 20, split 8 | released: 159 inputs, 200 trees, no depth limit, bootstrap off | |
| stability definitions | thermal >= 350 C, solvent +1, water classes 2-3 (0-based) | used as such in `stability_data.py` | drives the 17 (M5) |

All SI values in this table are copied from `stability_inventory.md`, not re-read from the SI.

## Check 7: claims stated more strongly than the evidence supports

1. `stability_inventory.md` finding 4 / section 2: SI test-set counts 'cannot be reproduced' from the release. False for the CSV rows (H1). Also the inference 'different data version/seed' is not supported.
2. `phase2.md` section 3 and `phase3.md` section 3: 'Plain split conformal behaves as the theory says' from the re-partition check. True arithmetic, but the check is exchangeable by construction (H2); it is not evidence about held-out-group deployment.
3. `recall.py` / `phase2.md` section 3: 'exchangeability is therefore at the group level', 'ties only make it conservative'. The point-level rule is not exactly valid with clustered positives (M3); ties are conservative, but clustering is not.
4. `phase2.md` section 4 and README.md goal: 'union bound gives recall >= 1 - alpha for MOFs that are stable at all three stages'. Justified only for each stage's own positives (H3); the notes' caveat is there but the sentence and the figure title-level claim read as a proof, and the Phase 3 checkbox is ticked.
5. `phase2.md` section 4: 'The union bound is conservative: it keeps more MOFs than needed.' Conservative only if coverage carries over to the all-stable subset; with 'harder' positives it is not (S7c).
6. `phase3.md` section 3 / `phase2.md`: the PAC 'validity' table treated as evidence against the PAC property; the check cannot reach the nominal level even for a valid rule (M4).
7. `phase2.md` section 2 'Reading' (water AUC 0.832 to 0.641, solvent accuracy 0.694 to 0.600, thermal MAE 52.4 to 61.7): quoted as the cost of group-aware splitting from one split; split-to-split sd is 0.03-0.05 AUC, so sizes are not established (M2).
8. `phase4.md` limitations: 'Duplicates or near-duplicates in different groups would leak ... not measured'. A rough measurement exists (M1: 7% of thermal test MOFs have a close cross-group twin with 0.81 label agreement); the limitation is understated rather than overstated.
9. `phase3.md` section 5: 'the limit is the features/labels rather than the model class'. Supported only by three tree families plus the released networks on one split; hedged with 'suggests', but not tested (no GNN, no richer features, no external set).
10. `phase4.md` 1a/'Reading': 'the real stage models lie close to the binormal curve ... a reasonable guide'. The notes say this was not tested formally; the gaps in the table (up to 0.03 in kept fraction, e.g. thermal Phase 2 at recall 0.80: 0.629 vs 0.659) are of the size of the intervals, so it is neither supported nor contradicted.
11. The headline '17 all-stable MOFs' (all cascade figures): presented as a fixed fact; it depends on three definitions I could not check against the paper (M5).

## What I could not check

- SI Table S5 and the R2 0.41 claim (file unreachable; see above).
- Whether the SI metrics are reproducible from a CSV-split retrain (not run; would follow H1).
- Whether `p4_01`'s threshold index is off by one (inferred from a consistent 0.002-0.008 difference; I did not read the function).
- The model-training results of `p2_02` / `p3_02` beyond what the saved prediction files give: I recomputed AUC/kept from the saved scores, but I did not retrain the neural networks.
