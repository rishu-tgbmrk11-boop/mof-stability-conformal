"""Review check 3b: do the proxy groups catch near-duplicates? For each stage, test MOFs of the saved group-aware split: distance to the nearest TRAIN MOF
(z-scored features, train statistics) vs the same for the saved random split; count cross-group (near-)identical feature vectors."""
import json, numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
R="data/raw/optimof/data/"
fc=json.load(open("data/processed/phase2/feature_columns.json"))
D={"thermal":pd.read_csv(R+"thermal/thermal_all_data.csv").set_index("refcode"),"solvent":pd.read_csv(R+"solvent/solvent_all_data.csv").set_index("refcode"),"water":pd.read_csv(R+"water_and_haz/data.csv").set_index("MOF_name")}
g=pd.read_parquet("data/processed/phase2/groups.parquet").set_index("name")
for t,df in D.items():
    X=df[fc[t]].values.astype(float); out=[]
    for col in ("split","split_random"):
        l=g.loc[df.index,col].values; tr=np.isin(l,["train"]); te=l=="test"
        mu,sd=X[tr].mean(0),X[tr].std(0); sd[sd==0]=1; Z=(X-mu)/sd
        d,i=NearestNeighbors(n_neighbors=1).fit(Z[tr]).kneighbors(Z[te]); d=d[:,0]
        gt=g.loc[df.index,"group_id"].values; same=(gt[tr][i[:,0]]==gt[te])
        out.append((col,np.median(d),np.percentile(d,10),(d<1e-6).sum(),(d<0.5).sum(),same.mean(),te.sum()))
    for o in out: print(f"{t:8s} {o[0]:12s} test {o[6]:4d}: median NN dist {o[1]:.2f}, p10 {o[2]:.2f}, identical-vector train twin {o[3]}, NN dist<0.5: {o[4]}, NN in same group {o[5]:.2f}")
    # exact duplicate feature vectors anywhere, and whether they sit in different groups
    dup=df[fc[t]].round(8).duplicated(keep=False); dd=df[dup]; 
    print(f"         exact-duplicate feature rows in table: {int(dup.sum())}; of the duplicate sets, share spanning different proxy groups:",
          (pd.DataFrame({'k':dd[fc[t]].round(8).astype(str).agg('|'.join,axis=1).values,'g':g.loc[dd.index,'group_id'].values}).groupby('k').g.nunique()>1).mean() if dup.any() else None)

print("\nlabel agreement of cross-group close pairs (group-aware split, NN dist<1.0 in z-space) vs all test MOFs' 1-NN agreement")
lab={"thermal":lambda d:(d["T"].values>=350).astype(int),"solvent":lambda d:(d["flag"].values>0).astype(int),"water":lambda d:(d["water_label"].values>=3).astype(int)}
for t,df in D.items():
    X=df[fc[t]].values.astype(float); y=lab[t](df); l=g.loc[df.index,"split"].values; tr=l=="train"; te=l=="test"
    mu,sd=X[tr].mean(0),X[tr].std(0); sd[sd==0]=1; Z=(X-mu)/sd
    d,i=NearestNeighbors(n_neighbors=1).fit(Z[tr]).kneighbors(Z[te]); d=d[:,0]; agree=(y[tr][i[:,0]]==y[te])
    for thr in (0.5,1.0): m=d<thr; print(f"{t:8s} dist<{thr}: n={int(m.sum())}, label agreement {agree[m].mean() if m.any() else float('nan'):.2f}; all test: {agree.mean():.2f}")
