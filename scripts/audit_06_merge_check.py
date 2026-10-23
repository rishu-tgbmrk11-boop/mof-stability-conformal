"""Audit step 6: numbers on the table merge in data_selection.py (thermal has priority over solvent over water for MOFs in several tables).
python -I scripts/audit_06_merge_check.py  -> data/processed/audit/merge_check.json"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd

R = "data/raw/optimof/data/"
T = pd.read_csv(R + "thermal/thermal_all_data.csv").set_index("refcode"); S = pd.read_csv(R + "solvent/solvent_all_data.csv").set_index("refcode"); W = pd.read_csv(R + "water_and_haz/data.csv").set_index("MOF_name")
out = {}
num = lambda A, B: [c for c in A.columns if c in B.columns and c != "0" and pd.api.types.is_numeric_dtype(A[c]) and pd.api.types.is_numeric_dtype(B[c])]
# solvent vs thermal
sh = sorted(set(S.index) & set(T.index)); cols = num(S, T)
a, b = S.loc[sh, cols].astype(float).values, T.loc[sh, cols].astype(float).values
diff_rows = ~np.all(np.isclose(a, b, rtol=1e-9, atol=1e-12), axis=1)
out["solvent_vs_thermal"] = dict(shared=len(sh), shared_cols=len(cols), mofs_with_any_difference=int(diff_rows.sum()), cols_that_differ=int((~np.isclose(a, b, rtol=1e-9, atol=1e-12)).any(axis=0).sum()))
geo = ["Di", "Df", "Dif", "GPOAV", "GPONAV", "GPOV", "GSA", "POAV", "PONAV", "VPOV", "VSA", "cell_v"]
out["solvent_vs_thermal"]["geometry_columns_identical"] = bool(np.allclose(S.loc[sh, geo].astype(float).values, T.loc[sh, geo].astype(float).values, rtol=1e-9))
# water vs thermal / solvent
for name, O in (("thermal", T), ("solvent", S)):
    sh = sorted(set(W.index) & set(O.index)); cols = num(W, O)
    a, b = W.loc[sh, cols].astype(float).values, O.loc[sh, cols].astype(float).values
    anyd = ~np.all(np.isclose(a, b, rtol=1e-9, atol=1e-12), axis=1)
    rel = lambda c: float(np.nanmedian(np.abs(W.loc[sh, c].astype(float).values - O.loc[sh, c].astype(float).values) / (np.abs(O.loc[sh, c].astype(float).values) + 1e-12)))
    out[f"water_vs_{name}"] = dict(shared=len(sh), shared_cols=len(cols), mofs_with_any_difference=int(anyd.sum()), median_rel_diff={c: rel(c) for c in ("VPOV", "GPOV", "VSA", "GSA", "POAV_vol_frac", "Di", "Df")},
                                   frac_differ={c: float((~np.isclose(W.loc[sh, c].astype(float).values, O.loc[sh, c].astype(float).values, rtol=1e-9, atol=1e-12)).mean()) for c in ("VPOV", "GPOV", "VSA", "GSA", "POAV_vol_frac")})
tn, sn, wn = set(T.index), set(S.index), set(W.index)
out["rows_using_other_table_features"] = dict(solvent=len(sn & tn), water=len(wn & (tn | sn)), water_total=len(wn), solvent_total=len(sn))
# RAC differences among the solvent/thermal shared rows
sh = sorted(sn & tn); rac = [c for c in num(S, T) if c.startswith(("f-", "mc-", "lc-", "func-", "D_"))]
d = (S.loc[sh, rac].astype(float).values - T.loc[sh, rac].astype(float).values); bad = (np.abs(d) > 1e-9).any(axis=1)
out["solvent_thermal_RAC_rows_differ"] = int(bad.sum()); out["solvent_thermal_RAC_example"] = {"refcode": sh[int(np.argmax(np.abs(d[:, rac.index('f-Z-3-all')])))], "f-Z-3-all_solvent_csv": float(S.loc[sh[int(np.argmax(np.abs(d[:, rac.index('f-Z-3-all')])))], "f-Z-3-all"]), "f-Z-3-all_thermal_csv": float(T.loc[sh[int(np.argmax(np.abs(d[:, rac.index('f-Z-3-all')])))], "f-Z-3-all"])}
# DOI / families / duplicates
doi = S["doi"].value_counts()
out["doi"] = dict(n_dois=int(doi.size), n_mofs=len(S), median_group=int(doi.median()), max_group=int(doi.max()), frac_in_groups_ge2=float((S["doi"].map(doi) >= 2).mean()))
out["families"] = {f"{n}_{k}": dict(n_families=int(pd.Series([str(i)[:k] for i in ids]).nunique()), frac_sharing=float(pd.Series([str(i)[:k] for i in ids]).map(pd.Series([str(i)[:k] for i in ids]).value_counts()).ge(2).mean())) for n, ids in (("thermal", T.index), ("solvent", S.index), ("water", W.index)) for k in (4, 5)}
out["class_balance"] = dict(solvent_stable=int((S["flag"] == 1).sum()), solvent_unstable=int((S["flag"] == -1).sum()), solvent_majority_fraction=float((S["flag"] == 1).mean()),
                           water_counts={int(k): int(v) for k, v in W["water_label"].value_counts().sort_index().items()}, water_majority_fraction=float(W["water_label"].value_counts().max() / len(W)),
                           thermal_frac_ge_350=float((T["T"] >= 350).mean()), thermal_n_ge_350=int((T["T"] >= 350).sum()), thermal_min=float(T["T"].min()), thermal_max=float(T["T"].max()))
json.dump(out, open("data/processed/audit/merge_check.json", "w"), indent=1)
print(json.dumps(out, indent=1)[:3500])
