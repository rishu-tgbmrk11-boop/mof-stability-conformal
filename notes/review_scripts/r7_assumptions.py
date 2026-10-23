"""Review checks for claims: (i) per-stage recall on the 17 all-stable MOFs vs on that stage's other positives among the 92 (cascade assumption);
(ii) two-sample KS: calibration-positive vs test-positive scores per stage (exchangeability across the held-out split)."""
import numpy as np, pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu
for ph in ("phase2","phase3"):
    o=pd.read_parquet(f"data/processed/{ph}/cascade_oof_rep0.parquet"); a17=o.all_stable.astype(bool)
    print(f"\n[{ph}] cascade_oof_rep0: {len(o)} MOFs, {int(a17.sum())} all-stable")
    for t in ("thermal","solvent","water"):
        st=o[f"stable_{t}"].astype(bool); 
        for al in ("0.033333","0.1"):
            p=o[f"pass_{t}_{al}"].astype(bool); 
            print(f"  {t:8s} alpha_k={al:8s}: stage recall among the {int((st&a17).sum())} all-stable = {p[st&a17].mean():.3f} | among the other {int((st&~a17).sum())} stage-stable MOFs of the 92 = {p[st&~a17].mean():.3f}")
        s=o[f"score_{t}"]; print(f"  {t:8s} median score: all-stable positives {s[a17].median():.3f} vs other stage positives {s[st&~a17].median():.3f}")
print("\nKS cal vs test positive scores (seed-0 models)")
for ph in ("phase2","phase3"):
    for t in ("thermal","solvent","water"):
        p=pd.read_parquet(f"data/processed/{ph}/pred_group_{t}_s0.parquet"); pos=p[p.stable==1]
        c=pos[pos.split=="cal"].score; te=pos[pos.split=="test"].score; tr=pos[pos.split.isin(["train"])].score
        print(f"  {ph} {t:8s} n_cal_pos {len(c)} n_test_pos {len(te)}: KS cal-vs-test p={ks_2samp(c,te).pvalue:.3f} | median {c.median():.3f} vs {te.median():.3f} | KS train-vs-test p={ks_2samp(tr,te).pvalue:.1e}")
