"""Review: re-derive Phase 4 numbers independently. (a) analytic binormal kept fraction at thermal prevalence 0.52 (notes: 0.89/0.82/0.70/0.62 at AUC .70/.80/.90/.95, recall .95)
(b) real pooled cal+test 'kept at recall' for Phase 3 models (notes 1a: thermal 0.582/0.726/0.840, solvent .738/.865/.922, water .725/.832/.911 at recall .80/.90/.95)."""
import numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
def kept(pi,auc,r): d=np.sqrt(2)*norm.ppf(auc); return pi*r+(1-pi)*(1-norm.cdf(d+norm.ppf(1-r)))
print("(a) analytic, pi=0.5247, recall 0.95:",{a:round(kept(0.5247,a,0.95),3) for a in (0.70,0.80,0.90,0.95)})
for ph in ("phase3","phase2"):
    print(f"(b) {ph}, pooled cal+test, seed 0")
    for t in ("thermal","solvent","water"):
        p=pd.read_parquet(f"data/processed/{ph}/pred_group_{t}_s0.parquet"); p=p[p.split.isin(["cal","test"])]
        pos=np.sort(p.score[p.stable==1].values); n=len(pos); out=[]
        for r in (0.8,0.9,0.95):
            k=int(np.floor((1-r)*n)); thr=pos[k-1] if k>=1 else -np.inf; out.append(round(float((p.score>=thr).mean()),3))
        print(f"   {t:8s} n={len(p)} n_pos={n} AUC {roc_auc_score(p.stable,p.score):.3f} kept@.80/.90/.95 = {out}")
