"""Fig 6: family-shift validation (Phase 5 section 3). Realised recall vs target recall for the per-stage split-conformal rule, calibrated on clusters different from the
test clusters (cross-cluster, orange) or on a random half of the test clusters themselves (calibrate-within control, grey). 20 repetitions (k-means seed / cluster order);
median with the 2.5-97.5% range over repetitions (about min-max for 20; repetitions share MOFs and are not independent)."""
import sys
sys.path.insert(0, "figures")
from common import *
import numpy as np
import pandas as pd

F = load(P5 + "family_shift.json"); A = F["alphas"]; X = [1 - a for a in A]


def draw(variant):
    fig, axes = panels(variant, 3, h2=2.4, h1=1.95, sharey=True)
    for ax, t in zip(axes, TASKS):
        df = pd.DataFrame(F["results"][t])
        ax.plot([0.65, 1], [0.65, 1], color=INK, lw=0.7, ls="--", label="target (y = x)")
        for key, col, mk, lab, dx in (("ctrl_recall", GREY, "s", "calibrate within the test clusters", -0.004), ("recall", ORANGE, "o", "calibrate on other clusters", 0.004)):
            v = [df[f"{key}_{a}"] for a in A]
            med = np.array([x.median() for x in v]); lo = np.array([x.quantile(0.025) for x in v]); hi = np.array([x.quantile(0.975) for x in v])
            xs = np.array(X) + dx
            ax.fill_between(xs, lo, hi, color=col, alpha=0.15, lw=0)
            ax.plot(xs, med, marker=mk, color=col, ms=3.4, lw=1.0, label=lab)
        ax.set_title(TITLE[t]); ax.set_ylim(0.1, 1.05); ax.set_xlim(0.67, 0.98); ax.set_xticks(X)
        ax.set_ylabel("realised test recall") if (variant == "1col" or t == "thermal") else None
        ax.set_xlabel("target recall (1 − α)") if (variant == "2col" or t == "water") else None
    letters(axes)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="outside lower center", ncol=3 if variant == "2col" else 1)
    save(fig, "fig6_family_shift", variant)


for v in VARIANTS:
    draw(v)
