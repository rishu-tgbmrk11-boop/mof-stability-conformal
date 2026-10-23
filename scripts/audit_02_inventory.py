"""Audit step 2: inventory of the three stability tables (raw CSV + released split tables).  python -I scripts/audit_02_inventory.py"""
import json
import os
import re
import sys

sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd

from src.models import safe_pickle as sp

R = "data/raw/optimof/"
out = {}
T = pd.read_csv(R + "data/thermal/thermal_all_data.csv")
S = pd.read_csv(R + "data/solvent/solvent_all_data.csv")
W = pd.read_csv(R + "data/water_and_haz/data.csv")
split = {n: sp.safe_joblib_load(R + p) for n, p in (("thermal", "data/thermal/thermal_split_data.pkl"), ("solvent", "data/solvent/solvent_split_data.pkl"), ("water", "data/water_and_haz/water_split_data.pkl"))}
A = sp.safe_joblib_load(R + "data/all_in_one.pkl")

print("== raw CSV tables")
for n, d, idc in (("thermal", T, "refcode"), ("solvent", S, "refcode"), ("water", W, "MOF_name")):
    nan_cols = d.columns[d.isna().any()].tolist()
    print(f"{n}: rows {len(d)}, columns {d.shape[1]}, id column '{idc}' unique {d[idc].nunique()}, NaN cells {int(d.isna().sum().sum())} in columns {nan_cols[:5]}")
    out[f"csv_{n}"] = dict(rows=len(d), cols=d.shape[1], nan_cells=int(d.isna().sum().sum()), nan_cols=nan_cols)

print("\n== released split tables (what the models were trained on)")
for n, d in split.items():
    print(f"{n}: rows {d.shape[0]}, features {d.shape[1]-1} (+1 label column), index={d.index.name}, unique index {d.index.is_unique}, NaN {int(d.isna().sum().sum())}")
    out[f"split_{n}"] = dict(rows=d.shape[0], features=d.shape[1] - 1)
print("\npaper SI feature counts: thermal 190, solvent 190, water 179  (released: %d, %d, %d)" % tuple(split[n].shape[1] - 1 for n in ("thermal", "solvent", "water")))

print("\n== the 20 columns data_selection.py drops as 'empty' (merged_drop_cols): are they really empty in the CSVs?")
drop20 = ["D_func-I-0-all", "D_func-I-1-all", "D_func-I-2-all", "D_func-I-3-all", "D_func-S-0-all", "D_func-T-0-all", "D_func-Z-0-all", "D_func-alpha-0-all", "D_func-chi-0-all",
          "D_lc-T-0-all", "D_lc-Z-0-all", "D_lc-alpha-0-all", "D_lc-chi-0-all", "D_mc-I-0-all", "D_mc-I-1-all", "D_mc-I-2-all", "D_mc-I-3-all", "D_mc-S-0-all", "D_mc-T-0-all", "D_mc-chi-0-all"]
rows = []
for c in drop20:
    row = {"column": c}
    for n, d in (("thermal", T), ("solvent", S), ("water", W)):
        if c in d.columns:
            row[n] = f"nunique={d[c].nunique()}, std={d[c].std():.3g}"
        else:
            row[n] = "absent"
    rows.append(row)
dd = pd.DataFrame(rows)
print(dd.to_string(index=False))
nonconst = [c for c in drop20 if any(c in d.columns and d[c].nunique() > 1 for d in (T, S, W))]
print(f"\ncolumns among the 20 that are NOT constant in at least one CSV: {len(nonconst)} -> {nonconst}")
out["drop20_nonconstant"] = nonconst

print("\n== the 11 extra columns dropped for water only (water_additional_drops): variance in the water CSV")
w11 = ["D_lc-I-0-all", "D_lc-I-1-all", "D_lc-I-2-all", "D_lc-I-3-all", "D_lc-S-0-all", "D_mc-Z-0-all", "POAV", "PONAV", "cell_v", "lc-I-0-all", "mc-I-0-all"]
print({c: (int(W[c].nunique()) if c in W.columns else "absent") for c in w11})

print("\n== overlaps between the three tables (by name/refcode)")
tn, sn, wn = set(T["refcode"]), set(S["refcode"]), set(W["MOF_name"])
print(f"thermal {len(tn)}, solvent {len(sn)}, water {len(wn)}")
print(f"thermal&solvent {len(tn & sn)}, thermal&water {len(tn & wn)}, solvent&water {len(sn & wn)}, all three {len(tn & sn & wn)}, union {len(tn | sn | wn)}")
print(f"all_in_one.pkl rows {len(A)} (names unique {A['name'].nunique()})")
out["overlap"] = dict(thermal=len(tn), solvent=len(sn), water=len(wn), ts=len(tn & sn), tw=len(tn & wn), sw=len(sn & wn), all3=len(tn & sn & wn), union=len(tn | sn | wn), all_in_one_rows=len(A))
# are the shared MOFs' features identical across tables?
common = sorted(tn & sn)
feat = [c for c in split["thermal"].columns if c != "thermal" and c in split["solvent"].columns]
a, b = split["thermal"].loc[common, feat].astype(float), split["solvent"].loc[common, feat].astype(float)
print(f"features shared by thermal & solvent split tables: {len(feat)}; max |difference| over {len(common)} shared MOFs: {np.abs(a.values - b.values).max():.3g}")

print("\n== identifiers and grouping information available")
print("MOFid column in any table:", [n for n, d in (("thermal", T), ("solvent", S), ("water", W)) if any('mofid' in c.lower() for c in d.columns)] or "none")
doi = S["doi"].value_counts()
print(f"solvent table DOI groups: {doi.size} DOIs for {len(S)} MOFs; group size: median {int(doi.median())}, max {int(doi.max())}; MOFs in groups of >=2: {int((S['doi'].map(doi) >= 2).sum())} ({(S['doi'].map(doi) >= 2).mean():.1%})")
for k in (4, 5):
    for n, ids in (("thermal", T["refcode"]), ("solvent", S["refcode"]), ("water", W["MOF_name"])):
        pre = ids.str[:k]; vc = pre.value_counts()
        print(f"  refcode first-{k}-letter families, {n}: {vc.size} families for {len(ids)} MOFs; MOFs sharing a family with another: {(pre.map(vc) >= 2).mean():.1%}")
print("water data_set source:", W["data_set"].value_counts().to_dict())
print("refcode formats (thermal), first 12:", T["refcode"].head(12).tolist())
print("name pattern check (thermal): refcodes not matching ^[A-Z]{6}\\d*$ :", int((~T["refcode"].str.match(r"^[A-Z]{6}\d*$")).sum()))

print("\n== exact duplicate feature rows (different MOF, identical feature vector) in each split table")
for n, d in split.items():
    f = d.drop(columns=[n]); dup = f.duplicated(keep=False)
    print(f"{n}: {int(dup.sum())} rows ({dup.mean():.1%}) share an identical feature vector with another MOF; distinct duplicated vectors: {int(f[dup].drop_duplicates().shape[0])}")
    if dup.any():
        g = d[dup].groupby(list(f.columns), sort=False)[n]
        agree = g.agg(lambda s: s.nunique() == 1)
        print(f"    within those duplicate groups, labels agree in {agree.mean():.1%} of groups")
json.dump(out, open("data/processed/audit/inventory.json", "w"), indent=1, default=str)
