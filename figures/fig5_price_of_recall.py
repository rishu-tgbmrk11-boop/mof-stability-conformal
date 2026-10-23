"""Fig 5: price of recall. Kept fraction vs AUC at target recall 0.80 / 0.90 / 0.95. Lines = analytic binormal curve; small squares with bars = SYNTHETIC scores
(plotted at the sample AUC, 95% group-bootstrap interval); large markers = REAL stage models (group-aware pooled calibration+test MOFs, 95% group-bootstrap boxes on both axes)."""
import sys
sys.path.insert(0, "figures")
from common import *

D = load(P4 + "price_of_recall.json")
R = D["recalls"]
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8), sharey=True)
for ax, t in zip(axes, TASKS):
    syn = D["synthetic"][t]
    for r in R:
        ax.plot([p["auc"] for p in syn["analytic"]], [p[f"kept_{r}"] for p in syn["analytic"]], color=RECALL_COL[r], lw=1.1, label=f"recall {r:.2f}" if t == "thermal" else None)
        xs = [p["auc_sample"] for p in syn["points"]]; k = [p[f"kept_{r}"] for p in syn["points"]]
        ax.errorbar(xs, [v[0] for v in k], yerr=[[v[0] - v[1] for v in k], [v[2] - v[0] for v in k]], fmt="s", color=RECALL_COL[r], ms=2.5, lw=0.7, capsize=0, alpha=0.8)
        for key, mk, fc in ((f"phase2_{t}", "s", "white"), (f"phase3_{t}", "o", RECALL_COL[r])):
            rr = D["real"][key]; a, kv = rr["auc"], rr[f"kept_{r}"]
            ax.errorbar(a[0], kv[0], xerr=[[a[0] - a[1]], [a[2] - a[0]]], yerr=[[kv[0] - kv[1]], [kv[2] - kv[0]]], fmt=mk, mfc=fc, mec=INK, ecolor=INK, color=INK, ms=5.5, lw=0.7, capsize=1.5, zorder=5)
    ax.set_title(f"{TITLE[t]} (prevalence {syn['prevalence']:.2f})"); ax.set_xlabel("AUC for 'truly stable'"); ax.set_xlim(0.5, 1.0); ax.set_ylim(0.3, 1.0)
axes[0].set_ylabel("fraction of MOFs kept")
from matplotlib.lines import Line2D
h = [Line2D([], [], color=RECALL_COL[r], lw=1.5, label=f"target recall {r:.2f} (analytic curve; colour of synthetic and real points)") for r in R]
h += [Line2D([], [], marker="s", color="w", mfc=GREY, ms=3, label="synthetic scores"), Line2D([], [], marker="s", color="w", mfc="white", mec=INK, ms=5, label="real stage model, Phase 2"),
      Line2D([], [], marker="o", color="w", mfc=GREY, mec=INK, ms=5, label="real stage model, Phase 3")]
fig.legend(handles=h, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.17), fontsize=6.5)
fig.tight_layout(); save(fig, "fig5_price_of_recall")
