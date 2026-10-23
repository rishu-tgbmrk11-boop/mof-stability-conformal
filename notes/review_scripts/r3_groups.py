"""Review check 3 + 6: rebuild groups independently (scipy connected components, no project imports) and compare with the saved groups.parquet / predictions."""
import numpy as np, pandas as pd, re
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
R="data/raw/optimof/data/"
th=pd.read_csv(R+"thermal/thermal_all_data.csv"); sv=pd.read_csv(R+"solvent/solvent_all_data.csv"); wt=pd.read_csv(R+"water_and_haz/data.csv")
names=sorted(set(th.refcode)|set(sv.refcode)|set(wt.MOF_name)); idx={n:i for i,n in enumerate(names)}
def comps(fam):
    rows,cols=[],[]
    keyed={}
    for n in names: keyed.setdefault(fam(n),[]).append(idx[n])
    for m in keyed.values():
        for j in m[1:]: rows.append(m[0]); cols.append(j)
    dois={}
    for r,d in zip(sv.refcode,sv.doi):
        if isinstance(d,str) and d.strip(): dois.setdefault(d.strip(),[]).append(idx[r])
    for m in dois.values():
        for j in m[1:]: rows.append(m[0]); cols.append(j)
    A=coo_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(names),)*2)
    return connected_components(A,directed=False)[1]
g=pd.read_parquet("data/processed/phase2/groups.parquet").set_index("name")
mine=pd.Series(comps(lambda n:n[:4]),index=names)
saved=g.loc[names,"group_id"]
print("names",len(names),"mine groups",mine.nunique(),"saved groups",saved.nunique())
# partitions identical iff the pair (mine, saved) is a bijection
pair=pd.DataFrame({"m":mine.values,"s":saved.values}); print("identical partition:", pair.groupby("m").s.nunique().max()==1 and pair.groupby("s").m.nunique().max()==1)
# DOI hygiene
d=sv.doi.dropna().astype(str); print("DOI strings with whitespace/case variants:", (d!=d.str.strip()).sum(), "unique raw",d.nunique(),"unique normalised(lower,strip)",d.str.strip().str.lower().nunique())
# splits: no group spans splits; consistency with prediction files
print("groups spanning >1 split (groups.parquet):", int((g.groupby("group_id").split.nunique()>1).sum()))
for t in ("thermal","solvent","water"):
    p=pd.read_parquet(f"data/processed/phase2/pred_group_{t}_s0.parquet")
    spans=(p.groupby("group_id").split.nunique()>1).sum()
    # train/val/cal/test sets of group ids mutually disjoint
    sets={s:set(p[p.split==s].group_id) for s in p.split.unique()}
    inter={(a,b):len(sets[a]&sets[b]) for a in sets for b in sets if a<b}
    # prediction-file group_id equals groups.parquet group_id
    same=(p.set_index("name").group_id==g.loc[p.name,"group_id"].values).all()
    # split label agrees with groups.parquet (val is carved from train)
    gs=g.loc[p.name,"split"].values; ok=((p.split.values==gs)|((p.split.values=="val")&(gs=="train"))).all()
    print(t,"groups spanning splits in pred file:",int(spans),"pairwise overlaps:",inter,"gid matches:",bool(same),"split matches groups.parquet:",bool(ok))
# 4-letter vs 6-letter family sensitivity and non-CSD names
csd=[bool(re.fullmatch(r"[A-Z]{6}(\d{2})?",n)) for n in names]; print("non-CSD-style names:",len(names)-sum(csd))
m6=pd.Series(comps(lambda n:n[:6]),index=names); print("groups with 6-letter family (+DOI):",m6.nunique(),"vs 4-letter:",mine.nunique())
# refcode families where the 6-letter family differs but they share 4 letters: how many 4-letter groups merge several 6-letter families
f6=pd.Series([n[:6] for n in names],index=names); print("4-letter groups containing >1 distinct 6-letter family:",int((pd.DataFrame({'g':mine.values,'f':f6.values}).groupby('g').f.nunique()>1).sum()))
# (6) the 92 and 17
T=th.set_index("refcode"); S=sv.set_index("refcode"); W=wt.set_index("MOF_name")
for n,df in (("thermal",T),("solvent",S),("water",W)): print(n,"rows",len(df),"unique index",df.index.is_unique)
three=sorted(set(T.index)&set(S.index)&set(W.index)); print("labelled in all three:",len(three))
tt=T.loc[three,"T"]; ss=S.loc[three,"flag"]; ww=W.loc[three,"water_label"]
print("thermal label range",tt.min(),tt.max(),"| solvent flag values",sorted(ss.unique()),"| water labels",sorted(ww.unique()))
print("water label counts in the 92:",ww.value_counts().sort_index().to_dict())
for wmin in (3,4):  # raw label >=3 == 0-based class>=2
    st=((tt>=350)&(ss>0)&(ww>=wmin)); print(f"stable at all three with thermal>=350, solvent>0, water raw label>={wmin}:",int(st.sum()))
for thr in (300,350,400): print("thermal thr",thr,int(((tt>=thr)&(ss>0)&(ww>=3)).sum()))
print("solvent flag raw values in full table:",sorted(S.flag.unique()),S.flag.value_counts().to_dict())
print("water raw label counts in full table:",W.water_label.value_counts().sort_index().to_dict())
print("thermal NaN:",int(T["T"].isna().sum()),"solvent NaN:",int(S.flag.isna().sum()),"water NaN:",int(W.water_label.isna().sum()))
