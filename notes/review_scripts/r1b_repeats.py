"""Review check 1b: is the single-split group-vs-random gap representative? 30 independent random group splits and 30 random MOF splits (60/20/20 over the union,
own implementation), RandomForest classifier for 'stable' per stage, AUC on the test part. Own groups = refcode[:4] + DOI components."""
import numpy as np, pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
R="data/raw/optimof/data/"
th=pd.read_csv(R+"thermal/thermal_all_data.csv").set_index("refcode"); sv=pd.read_csv(R+"solvent/solvent_all_data.csv").set_index("refcode"); wt=pd.read_csv(R+"water_and_haz/data.csv").set_index("MOF_name")
names=sorted(set(th.index)|set(sv.index)|set(wt.index)); idx={n:i for i,n in enumerate(names)}
rows,cols,fam,dois=[],[],{},{}
for n in names: fam.setdefault(n[:4],[]).append(idx[n])
for r,d in zip(sv.index,sv.doi):
    if isinstance(d,str): dois.setdefault(d,[]).append(idx[r])
for m in list(fam.values())+list(dois.values()):
    for j in m[1:]: rows.append(m[0]); cols.append(j)
gid=connected_components(coo_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(names),)*2),directed=False)[1]
size=np.bincount(gid)
task={"thermal":(th,[c for c in th.columns if c in set(pd.read_json("data/processed/phase2/feature_columns.json",typ="series")["thermal"])],(th["T"].values>=350).astype(int)),
      "solvent":(sv,list(pd.read_json("data/processed/phase2/feature_columns.json",typ="series")["solvent"]),(sv["flag"].values>0).astype(int)),
      "water":(wt,[c for c in wt.columns if c not in ("data_set","water_label","acid_label","base_label","boiling_label")],(wt["water_label"].values>=3).astype(int))}
def assign(rng,mode):
    if mode=="random": order=rng.permutation(len(names)); lab=np.empty(len(names),object); c=np.cumsum(np.ones(len(names)))/len(names); lab[order]=np.where(c<=.6,"train",np.where(c<=.8,"cal","test")); return lab
    perm=rng.permutation(gid.max()+1); cum=np.cumsum(size[perm])/len(names); gl=np.where(cum<=.6,"train",np.where(cum<=.8,"cal","test")); m=np.empty(gid.max()+1,object); m[perm]=gl; return m[gid]
res={t:{"random":[],"group":[]} for t in task}; leak=[]
for rep in range(30):
    rng=np.random.RandomState(1000+rep)
    for mode in ("random","group"):
        lab=pd.Series(assign(rng,mode),index=names)
        if mode=="random":
            tr=set(gid[[idx[n] for n in names if lab[n]=="train"]]); te=[idx[n] for n in names if lab[n]=="test"]; leak.append(np.mean([gid[i] in tr for i in te]))
        for t,(df,fc,st) in task.items():
            l=lab.loc[df.index].values; trn=np.isin(l,["train"]); tst=l=="test"
            X=df[fc].values.astype(float)
            rf=RandomForestClassifier(n_estimators=150,min_samples_leaf=1,max_features="sqrt",random_state=rep,n_jobs=8).fit(X[trn],st[trn])
            res[t][mode].append(roc_auc_score(st[tst],rf.predict_proba(X[tst])[:,1]))
print("random-split test MOFs whose group (proxy) also has a MOF in train: mean %.3f"%np.mean(leak))
for t in task:
    r,g=np.array(res[t]["random"]),np.array(res[t]["group"]); d=r-g
    print(f"{t:8s} AUC random {r.mean():.3f}±{r.std():.3f}  group {g.mean():.3f}±{g.std():.3f}  gap {d.mean():+.3f} (sd {d.std():.3f}; gap>0 in {int((d>0).sum())}/30 paired reps)")
