"""Review check 1: independent re-derivation of group-aware vs random metrics for the WATER stage from the raw released CSV
(own feature selection, own RF call, own metrics; saved groups.parquet used only as DATA for split membership)."""
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
R="data/raw/optimof/data/"
wt=pd.read_csv(R+"water_and_haz/data.csv").set_index("MOF_name")
feat=[c for c in wt.columns if c not in ("data_set","water_label","acid_label","base_label","boiling_label")]
print("features:",len(feat),"non-numeric:",[c for c in feat if not np.issubdtype(wt[c].dtype,np.number)],"NaN cells:",int(wt[feat].isna().sum().sum()))
X=wt[feat].values.astype(float); y=(wt.water_label.values-1).astype(int); stable=(y>=2).astype(int)
g=pd.read_parquet("data/processed/phase2/groups.parquet").set_index("name").loc[wt.index]
def metrics(tr,te,seed):
    rf=RandomForestClassifier(n_estimators=200,max_depth=None,bootstrap=False,max_features="sqrt",random_state=seed,n_jobs=8).fit(X[tr],y[tr])
    P=np.zeros((te.sum(),4)); P[:,rf.classes_]=rf.predict_proba(X[te]); ps=P[:,2]+P[:,3]
    return dict(auc_stable=roc_auc_score(stable[te],ps),acc_stable=((P.argmax(1)>=2)==stable[te]).mean(),acc4=(P.argmax(1)==y[te]).mean())
for mode,col in (("group","split"),("random","split_random")):
    tr=g[col].isin(["train","val"]).values if col=="split" else (g[col]=="train").values  # train+val used as training (val carved from train)
    te=(g[col]=="test").values
    res=pd.DataFrame([metrics(tr,te,s) for s in range(5)])
    print(f"{mode:6s} train {tr.sum()} test {te.sum()} ({len(set(g.group_id[te]))} groups) 5-seed mean:",res.mean().round(3).to_dict(),"sd",res.std().round(3).to_dict())
print("\nnotes (phase2.md): group AUC_stable 0.641, acc_stable 0.599, acc4 0.496 | random AUC_stable 0.832, acc_stable 0.721, acc4 0.597 (seed-0 / 5-seed means 0.628,0.600,0.499 | 0.830,0.718,0.599)")
