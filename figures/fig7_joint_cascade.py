"""Fig 7: joint cascade rule calibrated on the all-three-stable MOFs (Phase 6 sections 2 and 2b), nested evaluation, 3 repetitions of the same 92 labelled MOFs.
(a) realised recall of the joint rule with group-level calibration (14 groups as units, orange) and MOF-level calibration (17 MOFs, grey), pooled over the 3 repetitions
(51 positive evaluations; Clopper-Pearson 95% intervals that treat them as independent, i.e. OPTIMISTIC), against the guaranteed 1 - 2 alpha (line); the highest certifiable
target with 14 units (alpha = 1/15, recall 0.867) is marked. (b) fraction of the 92 labelled MOFs kept: group-level joint, MOF-level joint, union bound (alpha / 3 per stage)."""
import sys
sys.path.insert(0, "figures")
from common import *
import numpy as np
import pandas as pd
from scipy.stats import beta

J = pd.read_parquet(P6 + "joint_cascade_group_pvalues.parquet")
ALPHAS = [1 / 15, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30]
pos = J.all_stable.values; npos = int(pos.sum())
RULES = {"group": J.p_group_min.values, "mof": J.p_joint.values}
UNION = (J.p_thermal.values, J.p_solvent.values, J.p_water.values)
cp = lambda k, n: ((0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)), (1.0 if k == n else beta.ppf(0.975, k + 1, n - k)))
A_MAX = 1 / 15


def keep(name, a):
    return RULES[name] > a if name in RULES else (UNION[0] > a / 3) & (UNION[1] > a / 3) & (UNION[2] > a / 3)


def draw(variant):
    fig, (a1, a2) = panels(variant, 2, h2=2.5, h1=2.3)
    xs = np.array(ALPHAS)
    g = np.linspace(0.05, 0.31, 50); a1.plot(g, 1 - 2 * g, color=INK, lw=0.9, ls="--", label="guaranteed recall 1 − 2α")
    for name, col, mk, lab, dx in (("mof", GREY, "s", "joint, MOF-level (17 MOFs)", -0.003), ("group", ORANGE, "o", "joint, group-level (14 groups)", 0.003)):
        rec = np.array([(keep(name, a) & pos).sum() / npos for a in ALPHAS]); ci = np.array([cp(int((keep(name, a) & pos).sum()), npos) for a in ALPHAS])
        a1.errorbar(xs + dx, rec, yerr=[rec - ci[:, 0], ci[:, 1] - rec], fmt=mk, color=col, ms=3.6, capsize=1.6, lw=0.9, label=lab)
    for ax in (a1, a2):
        ax.axvline(A_MAX, color=BLUE, lw=0.8, ls=":")
    a1.annotate("highest certifiable\ntarget 0.867\n(α = 1/15)", xy=(A_MAX, 1 - 2 * A_MAX), xytext=(0.052, 0.58), fontsize=6.2, color=BLUE, va="top", ha="left",
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.6))
    a1.plot([A_MAX], [1 - 2 * A_MAX], marker="D", color=BLUE, ms=4, ls="none", zorder=5)
    a1.set_ylim(0.3, 1.04); a1.set_xlim(0.045, 0.315); a1.set_xlabel("α"); a1.set_ylabel("recall on the 17 all-stable MOFs"); a1.set_title("Recall"); a1.set_xticks([0.0667, 0.1, 0.15, 0.2, 0.25, 0.3]); a1.set_xticklabels(["1/15", "0.10", "0.15", "0.20", "0.25", "0.30"])
    for name, col, mk, ls_, lab in (("union", "#009E73", "^", "-", "union bound (α/3 per stage)"), ("mof", GREY, "s", "-", None), ("group", ORANGE, "o", "-", None)):
        a2.plot(xs, [keep(name, a).mean() for a in ALPHAS], marker=mk, color=col, ms=3.4, lw=1.0, ls=ls_, label=lab)
    a2.set_ylim(0.3, 1.0); a2.set_xlim(0.045, 0.315); a2.set_xlabel("α"); a2.set_ylabel("fraction of the 92 labelled MOFs kept"); a2.set_title("Kept fraction")
    a2.set_xticks([0.0667, 0.1, 0.15, 0.2, 0.25, 0.3]); a2.set_xticklabels(["1/15", "0.10", "0.15", "0.20", "0.25", "0.30"])
    letters([a1, a2])
    h, l = a1.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    fig.legend(h + h2, l + l2, frameon=False, loc="outside lower center", ncol=2)
    save(fig, "fig7_joint_cascade", variant)


for v in VARIANTS:
    draw(v)
