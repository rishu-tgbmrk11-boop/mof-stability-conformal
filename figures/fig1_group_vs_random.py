"""Fig 1: group-aware vs random split (Phase 2 baselines, seed 0, 95% bootstrap intervals: over groups for group-aware, over MOFs for random)."""
import sys
sys.path.insert(0, "figures")
from common import *

M = load(P2 + "metrics.json")["metrics"]
panels = [("AUC for 'truly stable'", [("thermal", "AUC_stable"), ("solvent", "AUC"), ("water", "AUC_stable")], (0.5, 0.9)),
          ("Accuracy of the stable / not-stable call", [("thermal", "acc_stable_at_350"), ("solvent", "accuracy"), ("water", "acc_stable_argmax_ge2")], (0.5, 0.8)),
          ("Thermal MAE (degC)", [("thermal", "MAE_degC")], None)]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [3, 3, 1.3]})
for ax, (title, items, lim) in zip(axes, panels):
    for j, (t, k) in enumerate(items):
        for mode, off, col, mk in (("random", -0.12, GREY, "s"), ("group", 0.12, ORANGE, "o")):
            v = M[f"{mode}_{t}"]["seed0_ci"][k]
            ax.errorbar(j + off, v[0], yerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt=mk, color=col, ms=4.5, capsize=2, lw=1, label=("random split" if mode == "random" else "group-aware split") if j == 0 else None)
    ax.set_xticks(range(len(items))); ax.set_xticklabels([TITLE[t].split()[0] for t, _ in items]); ax.set_title(title); ax.set_xlim(-0.5, len(items) - 0.5)
    if lim: ax.set_ylim(*lim)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout(); save(fig, "fig1_group_vs_random")
