"""Review check 2: synthetic test of split-conformal recall control. Positives ~ N(1,1) (population recall of 'score >= t' is exactly norm.sf(t-1)), 1000 repetitions."""
import sys, numpy as np
from scipy.stats import norm, beta
sys.path.insert(0, ".")
from src.conformal.recall import conformal_threshold as proj_thr, conformal_threshold_pac as proj_pac
def my_thr(pos, alpha):
    s=np.sort(pos); n=len(s); k=int(np.floor(alpha*(n+1)+1e-9)); return -np.inf if k<1 else s[k-1]
REP=1000; rng=np.random.RandomState(12345)
print("seed 12345; REP =",REP)
# 0. the two implementations agree
ok=all(my_thr(x,a)==proj_thr(x,a) for x in (rng.randn(n) for n in (5,19,20,99,100,500)) for a in (0.01,0.05,0.1,0.2,0.3,0.5)); print("project threshold == independent implementation on 36 cases:",ok)
print("\nS1 iid continuous. pop recall = norm.sf(t-1).  cols: alpha | mean pop recall (target 1-alpha) | min | P(pop recall>=target) | finite test (150 pos): mean realised recall, P(realised>=target)")
for n in (20,50,200):
    print(f" n_cal_pos={n}")
    for a in (0.05,0.1,0.2,0.3,0.5):
        pop=[];fin=[]
        for _ in range(REP):
            cal=rng.randn(n)+1; t=my_thr(cal,a); pop.append(1.0 if t==-np.inf else norm.sf(t-1)); te=rng.randn(150)+1; fin.append((te>=t).mean())
        pop,fin=np.array(pop),np.array(fin); k=int(np.floor(a*(n+1)))
        print(f"  alpha {a:.2f} k={k:2d}: {pop.mean():.4f} (target {1-a:.2f}, theory {1-k/(n+1):.4f}) min {pop.min():.3f}  P>=target {np.mean(pop>=1-a):.3f} | finite: {fin.mean():.4f}, P>=target {np.mean(fin>=1-a):.3f}")
print("\nS2 ties: integer-valued scores (positives ~ round(N(1,1)*2)), n_cal_pos=50: realised recall on 5000 test positives")
for a in (0.1,0.2):
    r=[]
    for _ in range(REP):
        cal=np.round((rng.randn(50)+1)*2); t=my_thr(cal,a); te=np.round((rng.randn(5000)+1)*2); r.append((te>=t).mean())
    print(f"  alpha {a}: mean recall {np.mean(r):.4f} (target {1-a}) -> conservative with ties:", np.mean(r)>=1-a)
print("\nS3 PAC variant (project's Beta rule, delta=0.1) in the iid case, population recall >= 1-alpha in what fraction of calibration draws? (nominal >= 0.90)")
for n in (50,200):
    for a in (0.1,0.2):
        pop=[]
        for _ in range(REP):
            t=proj_pac(rng.randn(n)+1,a,0.1); pop.append(1.0 if t==-np.inf else norm.sf(t-1))
        pop=np.array(pop); print(f"  n={n} alpha={a}: P(pop recall>=target) = {np.mean(pop>=1-a):.3f}; mean pop recall {pop.mean():.3f}")
print("\nS4 k=0 case: alpha*(n+1)<1 (n=8, alpha=0.05) -> threshold:",proj_thr(rng.randn(8),0.05))
print("\nS5 clustered positives: groups of 4 share a random effect (sd .7, residual sd .71); split by GROUP, 40 cal groups (160 pos), 40 test groups. Population-style check via many test groups")
for a in (0.1,0.2):
    r=[]
    for _ in range(REP):
        cal=(rng.randn(40,1)*0.7+rng.randn(40,4)*0.714+1).ravel(); t=my_thr(cal,a)
        te=(rng.randn(2000,1)*0.7+rng.randn(2000,4)*0.714+1).ravel(); r.append((te>=t).mean())
    r=np.array(r); print(f"  alpha {a}: mean recall {r.mean():.4f} (target {1-a}); sd {r.std():.3f} vs iid-160 sd {np.sqrt(a*(1-a)/160):.3f}; P(recall>=target) {np.mean(r>=1-a):.3f}")
print("\nS6 exchangeability broken: test positives shifted by -0.3 (new-family shift), n_cal_pos=200")
for a in (0.1,0.2):
    r=[]
    for _ in range(REP):
        t=my_thr(rng.randn(200)+1,a); r.append(norm.sf(t-0.7))
    print(f"  alpha {a}: mean recall {np.mean(r):.4f} (target {1-a}) -> violated: {np.mean(r)<1-a}")
print("\nS7 cascade, union bound alpha/3 per stage, 3 stages, n_cal_pos=100 each. recall of the pass-all rule on (a) independent stages (b) perfectly dependent stages (c) stable-at-all-three positives are 'harder' in each stage (-0.5 shift)")
for a in (0.1,0.3):
    A=[];B=[];C=[]
    for _ in range(REP):
        ts=[my_thr(rng.randn(100)+1,a/3) for _ in range(3)]
        A.append(np.prod([norm.sf(t-1) for t in ts])); B.append(min(norm.sf(t-1) for t in ts)); C.append(np.prod([norm.sf(t-0.5) for t in ts]))
    print(f"  alpha {a}: (a) {np.mean(A):.4f} (b) {np.mean(B):.4f} (c) {np.mean(C):.4f}   target {1-a}")
