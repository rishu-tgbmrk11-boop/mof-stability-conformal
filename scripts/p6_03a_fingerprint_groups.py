"""Phase 6 (3, fallback): fingerprint groups from the released features. Fingerprint of a MOF = z-scored vector of the feature columns common to the thermal, solvent and water
tables, taken from the first table that contains the MOF (thermal, then solvent, then water); z-scoring over the union of MOFs. Two MOFs are LINKED when their Euclidean distance
divided by sqrt(number of features) (an RMS z-difference per feature) is below eps; groups = connected components. Three eps fixed in advance and all reported (no selection on results):
0.1, 0.25, 0.5 (RMS z-units per feature). 'fp+proxy' groups also keep the refcode-family + DOI links of the proxy groups (a superset merge).
python -I scripts/p6_03a_fingerprint_groups.py   (repo root; writes data/processed/phase6/groups_fp_*.parquet and groups_fp_summary.json)"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors

from src.features import stability_data as sd

OUT = "data/processed/phase6/"
EPS = [0.1, 0.25, 0.5]
data = {t: sd.load_task(t) for t in sd.TASKS}
fc = sd.feature_columns(); common = [c for c in fc["thermal"] if c in set(fc["solvent"]) and c in set(fc["water"])]
print(f"common feature columns across the three tables: {len(common)} (thermal {len(fc['thermal'])}, solvent {len(fc['solvent'])}, water {len(fc['water'])})")
proxy = pd.read_parquet("data/processed/phase2/groups.parquet").set_index("name")
names = sorted(proxy.index); idx = {n: i for i, n in enumerate(names)}
rows = []
for n in names:
    for t in ("thermal", "solvent", "water"):
        if n in data[t].index:
            rows.append(data[t].loc[n, common].values.astype(float)); break
F = np.array(rows); Z = (F - F.mean(0)) / np.where(F.std(0) == 0, 1, F.std(0)); Z /= np.sqrt(Z.shape[1])          # RMS z-difference per feature
nn = NearestNeighbors(radius=max(EPS)).fit(Z); G = nn.radius_neighbors_graph(Z, mode="distance").tocoo()
dist = G.data; ii, jj = G.row, G.col; keep = ii < jj
print(f"MOFs {len(names)}; pairs within eps={max(EPS)}: {int(keep.sum())}; nearest-neighbour distance quantiles (5/25/50/75%): " + str(np.quantile(NearestNeighbors(n_neighbors=2).fit(Z).kneighbors(Z)[0][:, 1], [.05, .25, .5, .75]).round(3).tolist()))
pg = proxy["group_id"].values; summary = {}
for eps in EPS:
    m = keep & (dist <= eps)
    for tag, extra in (("fp", False), ("fp+proxy", True)):
        r, c = list(ii[m]), list(jj[m])
        if extra:
            byg = pd.Series(range(len(names))).groupby(pg).apply(list)
            for mem in byg:
                for j in mem[1:]:
                    r.append(mem[0]); c.append(j)
        A = coo_matrix((np.ones(len(r)), (r, c)), shape=(len(names),) * 2)
        lab = connected_components(A, directed=False)[1]; size = np.bincount(lab)
        df = pd.DataFrame({"name": names, "group_id": lab, "group_size": size[lab]})
        df.to_parquet(OUT + f"groups_{tag.replace('+', '_')}_eps{eps}.parquet")
        summary[f"{tag}|{eps}"] = dict(n_groups=int(len(size)), largest=int(size.max()), share_largest=float(size.max() / len(names)), singletons=int((size == 1).sum()), median_size=float(np.median(size)))
        print(f"  {tag:9s} eps {eps}: {len(size)} groups, largest {size.max()} MOFs ({100 * size.max() / len(names):.1f}%), singletons {int((size == 1).sum())}")
print(f"  proxy (refcode family + DOI): {proxy.group_id.nunique()} groups, largest {int(proxy.group_size.max())}")
summary["proxy"] = dict(n_groups=int(proxy.group_id.nunique()), largest=int(proxy.group_size.max()))
json.dump(summary, open(OUT + "groups_fp_summary.json", "w"), indent=1)

# ---- second fallback family, ADDED after seeing that single-link components chain into a giant group at eps >= 0.25:
# complete-linkage agglomerative clustering (cluster diameter <= d in the same RMS z-units; cannot chain). d fixed at 0.1, 0.25, 0.5 (all reported).
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
Lk = linkage(pdist(Z), method="complete")
byg = pd.Series(range(len(names))).groupby(pg).apply(list)
for d in (0.1, 0.25, 0.5):
    lab = fcluster(Lk, t=d, criterion="distance")
    for tag, merge in (("cl", False), ("cl+proxy", True)):
        if merge:
            r, c = [], []
            for k in np.unique(lab):
                mem = np.where(lab == k)[0]; r += [mem[0]] * (len(mem) - 1); c += list(mem[1:])
            for mem in byg:
                for j in mem[1:]:
                    r.append(mem[0]); c.append(j)
            lab2 = connected_components(coo_matrix((np.ones(len(r)), (r, c)), shape=(len(names),) * 2), directed=False)[1]
        else:
            lab2 = lab
        size = np.bincount(lab2 if lab2.min() == 0 else lab2 - lab2.min()); lab_ = lab2 if lab2.min() == 0 else lab2 - lab2.min()
        pd.DataFrame({"name": names, "group_id": lab_, "group_size": size[lab_]}).to_parquet(OUT + f"groups_{tag.replace('+', '_')}_d{d}.parquet")
        summary[f"{tag}|{d}"] = dict(n_groups=int((size > 0).sum()), largest=int(size.max()), share_largest=float(size.max() / len(names)), singletons=int((size == 1).sum()), median_size=float(np.median(size[size > 0])))
        print(f"  {tag:9s} diameter {d}: {(size > 0).sum()} groups, largest {size.max()} MOFs ({100 * size.max() / len(names):.1f}%), singletons {int((size == 1).sum())}")
json.dump(summary, open(OUT + "groups_fp_summary.json", "w"), indent=1)
