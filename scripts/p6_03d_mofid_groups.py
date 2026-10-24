"""Phase 6 (3): MOFid groups. Reads data/processed/phase6/mofid_results.csv (p6_03b), builds groups, and compares them with the proxy groups (refcode family + DOI).
MOFid key: `mofid_full` = the MOFid string before ';' (building-block SMILES + topology + catenation); `mofid_bb` = building-block SMILES only (topology and catenation dropped).
Uncovered MOFs (no CIF, timeout, error, empty SMILES) are singletons in the '-only' groups and keep their proxy group in the '+proxy' groups (connected components over equal MOFid OR same proxy group).
python -I scripts/p6_03d_mofid_groups.py"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from src.features import stability_data as sd

OUT = "data/processed/phase6/"
res = pd.read_csv(OUT + "mofid_results.csv").drop_duplicates("name", keep="last").set_index("name")
proxy = pd.read_parquet("data/processed/phase2/groups.parquet").set_index("name")
names = sorted(proxy.index); idx = {n: i for i, n in enumerate(names)}
data = {t: sd.load_task(t) for t in sd.TASKS}
status = res.status.str.split(":").str[0].value_counts().to_dict()
ok = res[(res.status == "ok") & res.mofid.notna() & (res.mofid.str.len() > 0)].copy()
ok["full"] = ok.mofid.str.split(";").str[0]
ok["bb"] = ok.full.str.rsplit(" MOFid-v1.", n=1).str[0]
ok["topology"] = ok.full.str.rsplit(" MOFid-v1.", n=1).str[1].str.split(".").str[0]
ok = ok[ok.bb.str.len() > 0]
S = dict(n_names=len(names), n_with_cif=int(len(res)), status=status, n_mofid=int(len(ok)), n_topology_error_or_unknown=int(ok.topology.isin(["ERROR", "UNKNOWN"]).sum()))
S["coverage_by_stage"] = {t: dict(covered=int(sum(n in ok.index for n in data[t].index)), total=int(len(data[t]))) for t in sd.TASKS}
L = sorted(set(data["thermal"].index) & set(data["solvent"].index) & set(data["water"].index))
allst = [n for n in L if all(data[t].loc[n, "stable"] == 1 for t in sd.TASKS)]
S["coverage_92"] = dict(covered=int(sum(n in ok.index for n in L)), total=len(L)); S["coverage_17"] = dict(covered=int(sum(n in ok.index for n in allst)), total=len(allst))
S["distinct_mofid_full"] = int(ok.full.nunique()); S["distinct_mofid_bb"] = int(ok.bb.nunique())
print(f"CIF-backed MOFs {len(res)}; status {status}; usable MOFid {len(ok)} of {len(names)} ({100 * len(ok) / len(names):.1f}%); distinct full {ok.full.nunique()}, distinct building-block keys {ok.bb.nunique()}; topology ERROR/UNKNOWN in {S['n_topology_error_or_unknown']}")
print("coverage by stage:", S["coverage_by_stage"], "| 92 labelled:", S["coverage_92"], "| 17 all-stable:", S["coverage_17"])


def comps(links):
    r, c = [], []
    for mem in links:
        r += [mem[0]] * (len(mem) - 1); c += list(mem[1:])
    A = coo_matrix((np.ones(len(r)), (r, c)), shape=(len(names),) * 2)
    return connected_components(A, directed=False)[1]


def by_key(key):
    d = {}
    for n, k in key.items():
        d.setdefault(k, []).append(idx[n])
    return list(d.values())


proxy_links = list(pd.Series(range(len(names))).groupby(proxy.group_id.values).apply(list))
for tag, col in (("mofid_full", "full"), ("mofid_bb", "bb")):
    for suffix, extra in (("only", []), ("proxy", proxy_links)):
        lab = comps(by_key(ok[col]) + extra); size = np.bincount(lab)
        df = pd.DataFrame({"name": names, "group_id": lab, "group_size": size[lab]}); df.to_parquet(OUT + f"groups_{tag}_{suffix}.parquet")
        S[f"{tag}_{suffix}"] = dict(n_groups=int(len(size)), largest=int(size.max()), singletons=int((size == 1).sum()))
        print(f"  groups {tag}_{suffix}: {len(size)} groups, largest {size.max()} MOFs ({100 * size.max() / len(names):.1f}%), singletons {int((size == 1).sum())}")
# agreement between MOFid and the proxy groups (covered MOFs only)
o = ok.join(proxy[["group_id"]]); o["pg"] = o.group_id
for col in ("full", "bb"):
    g = o.groupby(col).agg(n=("pg", "size"), n_proxy=("pg", "nunique"))
    multi = g[g.n > 1]
    in_dup = o[o[col].isin(multi.index)]
    S[f"agree_{col}"] = dict(n_keys_with_several_mofs=int(len(multi)), mofs_in_such_keys=int(len(in_dup)), keys_spanning_several_proxy_groups=int((multi.n_proxy > 1).sum()),
                            mofs_whose_key_spans_several_proxy_groups=int(o[o[col].isin(multi[multi.n_proxy > 1].index)].shape[0]))
    pg = o.groupby("pg")[col].nunique(); sz = o.groupby("pg").size()
    S[f"agree_{col}"]["proxy_groups_with_several_distinct_keys"] = int(((pg > 1)).sum()); S[f"agree_{col}"]["proxy_groups_with_2plus_covered_mofs"] = int((sz > 1).sum())
    print(f"  [{col}] keys shared by >=2 MOFs: {len(multi)} covering {len(in_dup)} MOFs; of these keys {int((multi.n_proxy > 1).sum())} span several proxy groups (covering {S[f'agree_{col}']['mofs_whose_key_spans_several_proxy_groups']} MOFs); "
          f"proxy groups with >=2 covered MOFs: {int((sz > 1).sum())}, of which {int((pg > 1).sum())} contain several distinct {col} keys")
# cross-split leakage under the Phase 2 proxy-group split (seed 0): test MOFs whose MOFid also occurs in the training part
sp = proxy["split"]
for col in ("full", "bb"):
    tr = set(ok.loc[[n for n in ok.index if sp[n] == "train"], col]); te = [n for n in ok.index if sp[n] == "test"]
    leak = [n for n in te if ok.loc[n, col] in tr]
    S[f"proxy_split_leak_{col}"] = dict(test_covered=len(te), test_with_identical_key_in_train=len(leak))
    print(f"  Phase 2 proxy-group split, seed 0: {len(leak)} of {len(te)} covered test MOFs have an identical {col} key in the training part ({100 * len(leak) / max(1, len(te)):.1f}%)")
    for t in sd.TASKS:
        names_t = [n for n in te if n in data[t].index]; lk = [n for n in names_t if ok.loc[n, col] in tr]
        S[f"proxy_split_leak_{col}"][t] = dict(test_covered=len(names_t), leaking=len(lk))
json.dump(S, open(OUT + "mofid_groups_summary.json", "w"), indent=1, default=float)
print("saved")
