"""Review check 2b: does within-group dependence push mean recall below 1-alpha? Population recall of the threshold under N(1,1)-marginal positives is exact:
group effect sd s, residual sd sqrt(1-s^2) so the marginal of one positive is N(1,1) for every s. Calibration: G groups x m positives. 5000 reps, seed 777."""
import numpy as np
from scipy.stats import norm
rng=np.random.RandomState(777); REP=5000
def thr(pos,a):
    s=np.sort(pos); k=int(np.floor(a*(len(s)+1)+1e-9)); return -np.inf if k<1 else s[k-1]
print("G groups x m positives | rho(sd of group effect) | alpha | mean pop recall | SE | target | theory 1-k/(n+1)")
for G,m in ((40,4),(15,4),(15,8)):
  for s in (0.0,0.7,0.9):
    for a in (0.1,0.2):
        n=G*m; k=int(np.floor(a*(n+1))); r=[]
        for _ in range(REP):
            cal=(rng.randn(G,1)*s+rng.randn(G,m)*np.sqrt(1-s*s)+1).ravel(); t=thr(cal,a); r.append(norm.sf(t-1))
        r=np.array(r); print(f" {G}x{m} s={s:.1f} alpha={a}: {r.mean():.4f} ±{r.std()/np.sqrt(REP):.4f}  target {1-a}  theory {1-k/(n+1):.4f}  P(>=target) {np.mean(r>=1-a):.3f}")
