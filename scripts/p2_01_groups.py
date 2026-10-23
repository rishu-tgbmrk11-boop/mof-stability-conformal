"""Phase 2, step 1: group-aware splits (connected components over 4-letter refcode family and DOI), 60/20/20 train/cal/test by group.
python -I scripts/p2_01_groups.py   (run from the repo root)"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd

from src.features.groups import assign_groups, build_groups, random_assign, size_summary
from src.features import stability_data as sd
from src.models import safe_pickle as sp

OUT = "data/processed/phase2/"
FRAC = {"train": 0.6, "cal": 0.2, "test": 0.2}
SEED = 0
# feature columns of the released split tables (data only; loaded with the restricted loader)
R = "data/raw/optimof/"
cols = {t: [c for c in sp.safe_joblib_load(R + p).columns if c != t] for t, p in (("thermal", "data/thermal/thermal_split_data.pkl"), ("solvent", "data/solvent/solvent_split_data.pkl"), ("water", "data/water_and_haz/water_split_data.pkl"))}
json.dump(cols, open(sd.FEATURE_COLUMNS_JSON, "w"))
print("feature columns:", {t: len(c) for t, c in cols.items()})

data = {t: sd.load_task(t) for t in sd.TASKS}
names = sorted(set().union(*[set(d.index) for d in data.values()]))
g = build_groups(names, sd.doi_by_name(), k=4)
g["split"] = assign_groups(g, FRAC, seed=SEED).values
g["split_random"] = random_assign(g["name"].values, FRAC, seed=SEED).reindex(g["name"]).values
g.to_parquet(OUT + "groups.parquet")

# integrity checks
chk = g.groupby("group_id")["split"].nunique()
assert (chk == 1).all(), "a group spans two splits"
print(f"\nMOFs in the union: {len(g):,}; groups: {g.group_id.nunique():,}  (4-letter refcode family + DOI links)")
print("every group lies wholly in one split:", bool((chk == 1).all()))
print("\ngroup size distribution (all groups):")
print(size_summary(g).to_string(index=False))
sz = g.drop_duplicates("group_id")["group_size"]
print(f"largest groups: {sorted(sz.tolist(), reverse=True)[:8]}; median group size {int(sz.median())}; mean {sz.mean():.2f}")
print("\nsplit summary (group-aware assignment, seed 0):")
rows = []
for s in FRAC:
    m = g.split == s
    rows.append(dict(split=s, groups=g[m].group_id.nunique(), mofs=int(m.sum()), frac=m.mean(), largest_group=int(g[m].group_size.max())))
print(pd.DataFrame(rows).round(4).to_string(index=False))
print("\nMOFs per task per split (group-aware | MOF-level random):")
summ = {}
for t, d in data.items():
    gi = g.set_index("name")
    a = gi.loc[d.index, "split"].value_counts(); r = gi.loc[d.index, "split_random"].value_counts()
    st = d.assign(split=gi.loc[d.index, "split"].values).groupby("split")["stable"].agg(["sum", "mean"])
    print(f"  {t:8s} total {len(d):5d} | group: train {a['train']:4d} cal {a['cal']:4d} test {a['test']:4d} | random: train {r['train']:4d} cal {r['cal']:4d} test {r['test']:4d} | stable (group split): " + ", ".join(f"{s} {int(st.loc[s, 'sum'])} ({st.loc[s, 'mean']:.1%})" for s in FRAC))
    summ[t] = dict(total=len(d), group={s: int(a[s]) for s in FRAC}, random={s: int(r[s]) for s in FRAC})
three = set(data["thermal"].index) & set(data["solvent"].index) & set(data["water"].index)
gi = g.set_index("name")
stable3 = [n for n in three if data["thermal"].loc[n, "stable"] == 1 and data["solvent"].loc[n, "stable"] == 1 and data["water"].loc[n, "stable"] == 1]
print(f"\nMOFs labelled in all three tables: {len(three)}; truly stable at all three stages: {len(stable3)}; their splits (group-aware): " + str(gi.loc[sorted(three), 'split'].value_counts().to_dict()))
json.dump(dict(n_mofs=len(g), n_groups=int(g.group_id.nunique()), sizes=size_summary(g).to_dict("records"), per_split=rows, per_task=summ, n_three=len(three), n_stable3=len(stable3), seed=SEED, fractions=FRAC), open(OUT + "groups_summary.json", "w"), indent=1, default=float)
