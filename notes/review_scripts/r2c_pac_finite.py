"""Review check 2c: in the exactly-iid case, what share of calibration draws reach the target when recall is measured on a FINITE test set (as notes' 'partitions >= target' does)?
n_cal_pos / n_test_pos set to the sizes of the real stages (thermal 336/313, solvent 258/253, water 132/139); alpha 0.10, delta 0.1; 2000 reps; seed 99."""
import sys, numpy as np
from scipy.stats import norm
sys.path.insert(0,".")
from src.conformal.recall import conformal_threshold as plain, conformal_threshold_pac as pac
rng=np.random.RandomState(99); REP=2000; a=0.1
for name,nc,nt in (("thermal",336,313),("solvent",258,253),("water",132,139)):
    P={"plain":[], "pac":[]}; Q={"plain":[], "pac":[]}
    for _ in range(REP):
        cal=rng.randn(nc)+1; te=rng.randn(nt)+1
        for k,f in (("plain",lambda c:plain(c,a)),("pac",lambda c:pac(c,a,0.1))):
            t=f(cal); P[k].append((te>=t).mean()>=1-a); Q[k].append(norm.sf(t-1)>=1-a)
    print(f"{name:8s} (n_cal_pos {nc}, n_test_pos {nt}): P(finite-test recall >= 0.90): plain {np.mean(P['plain']):.3f}, PAC {np.mean(P['pac']):.3f} | P(population recall >= 0.90): plain {np.mean(Q['plain']):.3f}, PAC {np.mean(Q['pac']):.3f}")
