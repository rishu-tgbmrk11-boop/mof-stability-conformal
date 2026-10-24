"""Phase 5, step 4: sensitivity of the '17 MOFs stable at all three stages' to the stability cut-offs. Raw labels of the 92 MOFs present in all three released CSVs
(thermal T in degC, solvent flag +-1, water raw label 1-4 = class 0-3 + 1). Part B applies the union-bound cascade pass flags SAVED from Phase 2/3 (repetition 0;
stage thresholds were calibrated for the ORIGINAL positives (T >= 350, flag +1, class >= 2), so for other definitions the pass rate is only indicative, not a
re-calibrated cascade). python -I scripts/p5_06_sensitivity_17.py"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
from scipy.stats import beta

R = "data/raw/optimof/data/"
T = pd.read_csv(R + "thermal/thermal_all_data.csv").set_index("refcode")["T"]; S = pd.read_csv(R + "solvent/solvent_all_data.csv").set_index("refcode")["flag"]
W = pd.read_csv(R + "water_and_haz/data.csv").set_index("MOF_name")["water_label"]
names = sorted(set(T.index) & set(S.index) & set(W.index)); lab = pd.DataFrame({"T": T.loc[names], "solvent": S.loc[names], "water": W.loc[names]})
print(f"{len(lab)} MOFs labelled in all three tables")
variants = []
for tc in (300, 325, 350, 375, 400):
    for wc, wn in ((2, "raw>=2 (class>=1)"), (3, "raw>=3 (class>=2, SI)"), (4, "raw>=4 (class 3 only)")):
        variants.append((tc, wc, wn))
rows = []
cp = lambda k, n: ((0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)), (1.0 if k == n else beta.ppf(0.975, k + 1, n - k)))
for tc, wc, wn in variants:
    st = (lab["T"] >= tc) & (lab["solvent"] > 0) & (lab["water"] >= wc)
    rows.append(dict(thermal_cut_degC=tc, water_def=wn, n_all_stable=int(st.sum()), share_of_92=float(st.mean())))
df = pd.DataFrame(rows)
print("\nA. number of MOFs stable at all three stages (of 92); solvent: flag +1 in all rows")
print(df.pivot(index="thermal_cut_degC", columns="water_def", values="n_all_stable").to_string())
# single-stage counts for context
print("\nstage-positive counts among the 92:", {f"thermal>={c}": int((lab['T'] >= c).sum()) for c in (300, 350, 400)}, "solvent+1:", int((lab.solvent > 0).sum()), {f"water raw>={c}": int((lab.water >= c).sum()) for c in (2, 3, 4)})
out = {"counts": rows}
print("\nB. share of the redefined positives that pass the SAVED union-bound cascade (rep 0; indicative only; alpha_k = alpha/3)")
for ph in ("phase2", "phase3"):
    o = pd.read_parquet(f"data/processed/{ph}/cascade_oof_rep0.parquet"); o = o.loc[o.index.intersection(lab.index)]
    for a, ak in ((0.05, "0.016667"), (0.1, "0.033333"), (0.2, "0.066667"), (0.3, "0.1")):
        keep = o[f"pass_thermal_{ak}"].astype(bool) & o[f"pass_solvent_{ak}"].astype(bool) & o[f"pass_water_{ak}"].astype(bool)
        res = []
        for tc, wc, wn in variants:
            if wn.startswith("raw>=3") and tc in (300, 350, 400):
                st = (lab.loc[o.index, "T"] >= tc) & (lab.loc[o.index, "solvent"] > 0) & (lab.loc[o.index, "water"] >= wc)
                k, n = int((keep & st).sum()), int(st.sum()); lo, hi = cp(k, n)
                res.append(dict(thermal_cut_degC=tc, water_def=wn, passes=k, n=n, recall=k / n, cp_lo=lo, cp_hi=hi))
        out[f"{ph}_alpha{a}"] = res
        print(f"  {ph} target {1 - a:.2f}: " + "; ".join(f"T>={r['thermal_cut_degC']}: {r['passes']}/{r['n']} = {r['recall']:.2f} [{r['cp_lo']:.2f}, {r['cp_hi']:.2f}]" for r in res))
st = (lab.loc[o.index, "T"] >= 350) & (lab.loc[o.index, "solvent"] > 0) & (lab.loc[o.index, "water"] >= 3)
print(f"\n(check) original definition gives {int(st.sum())} positives")
json.dump(out, open("data/processed/phase5/sensitivity_17.json", "w"), indent=1, default=float)
