"""Fig 4: cascade recall on the 17 all-stable MOFs (left) and kept fraction of the 92 labelled MOFs (right), out-of-fold, with Clopper-Pearson 95% intervals.
Dots = the 5 repetitions (same MOFs, not independent); marker = mean; interval = Clopper-Pearson for the rounded mean count out of 17 (as in notes/phase3.md)."""
import sys
sys.path.insert(0, "figures")
import numpy as np
from scipy.stats import beta
from common import *

X2, X3 = load(P2 + "cascade.json"), load(P3 + "cascade.json")
A = X3["alphas"]


def cp(k, n):
    return (0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)), (1.0 if k == n else beta.ppf(0.975, k + 1, n - k))


series = [(X3, "union", "Phase 3, union bound (alpha/3 per stage)", ORANGE, "o", 0.0), (X3, "each_alpha", "Phase 3, no union bound (alpha per stage)", GREEN, "^", 0.008),
          (X2, "union", "Phase 2, union bound (alpha/3 per stage)", GREY, "s", -0.008)]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.7))
a1.plot([0.65, 1], [0.65, 1], color=INK, lw=0.8, ls="--", label="target (y = x)")
for X, kind, lab, col, mk, off in series:
    R = X["results"]; n = R[0]["n_all_stable"]; x = np.array([1 - a for a in A]) + off
    cnt = np.array([[r["cascade"][f"{kind}_{a}"]["n_pass_pos"] for r in R] for a in A])
    for j in range(cnt.shape[1]):
        a1.plot(x, cnt[:, j] / n, ".", color=col, alpha=0.35, ms=3)
    m = cnt.mean(1); ints = np.array([cp(int(round(v)), n) for v in m])
    a1.errorbar(x, m / n, yerr=[m / n - ints[:, 0], ints[:, 1] - m / n], fmt=mk + "-", color=col, ms=4.5, capsize=2, lw=1, label=lab)
    a2.plot([1 - a for a in A], [np.mean([r["cascade"][f"{kind}_{a}"]["kept"] for r in R]) for a in A], mk + "-", color=col, ms=4.5, lw=1)
a1.set_xlabel("target recall (1 - alpha)"); a1.set_ylabel(f"recall on the {n} all-stable MOFs"); a1.set_title("Cascade recall"); a1.set_ylim(0, 1.03)
a2.set_xlabel("target recall (1 - alpha)"); a2.set_ylabel(f"fraction of the {X3['results'][0]['n_labelled']} labelled MOFs kept"); a2.set_title("Cascade kept fraction"); a2.set_ylim(0, 1.03)
h, l = a1.get_legend_handles_labels()
fig.legend(h, l, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.1))
fig.tight_layout(); save(fig, "fig4_cascade")
