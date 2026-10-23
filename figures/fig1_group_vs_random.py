"""Fig 1: group-aware vs random split (Phase 2 baselines, seed 0, 95% bootstrap intervals: over groups for group-aware, over MOFs for random)."""
import sys
sys.path.insert(0, "figures")
from common import *

M = load(P2 + "metrics.json")["metrics"]
PANELS = [("AUC, 'truly stable'", [("thermal", "AUC_stable"), ("solvent", "AUC"), ("water", "AUC_stable")], (0.5, 0.9)),
          ("Accuracy of stable call", [("thermal", "acc_stable_at_350"), ("solvent", "accuracy"), ("water", "acc_stable_argmax_ge2")], (0.5, 0.8)),
          ("MAE (°C)", [("thermal", "MAE_degC")], None)]


def draw(variant):
    if variant == "2col":
        fig, axes = panels(variant, 3, h2=2.4, ratios=[3, 3, 1.3])
        leg_ax = None
    else:                                                       # 2 x 2 grid; the fourth cell holds the legend
        fig, grid = plt.subplots(2, 2, figsize=(W1, 5.2 * CM + 1.2), layout="constrained")
        axes = [grid[0, 0], grid[0, 1], grid[1, 0]]; leg_ax = grid[1, 1]; leg_ax.axis("off")
    for ax, (title, items, lim) in zip(axes, PANELS):
        for j, (t, k) in enumerate(items):
            for mode, off, col, mk in (("random", -0.14, GREY, "s"), ("group", 0.14, ORANGE, "o")):
                v = M[f"{mode}_{t}"]["seed0_ci"][k]
                ax.errorbar(j + off, v[0], yerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt=mk, color=col, ms=3.8, capsize=1.8, lw=0.9,
                            label=("random split (MOF-level)" if mode == "random" else "group-aware split") if j == 0 else None)
        ax.set_xticks(range(len(items))); ax.set_xticklabels([TITLE[t].split()[0] for t, _ in items]); ax.set_title(title); ax.set_xlim(-0.55, len(items) - 0.45)
        if lim: ax.set_ylim(*lim)
    letters(axes)
    h, l = axes[0].get_legend_handles_labels()
    if leg_ax is not None:
        leg_ax.legend(h, l, frameon=False, loc="center", ncol=1)
    else:
        fig.legend(h, l, frameon=False, loc="outside lower center", ncol=2)
    save(fig, "fig1_group_vs_random", variant)


for v in VARIANTS:
    draw(v)
