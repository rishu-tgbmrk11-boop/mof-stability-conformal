"""Fig 3: fraction of test MOFs kept vs target recall for the per-stage split conformal rule (group-aware; 95% CI over test groups)."""
import sys
sys.path.insert(0, "figures")
from common import *

C2, C3 = load(P2 + "conformal.json"), load(P3 + "conformal.json")
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
for ax, t in zip(axes, TASKS):
    for C, col, mk, lab in ((C2, GREY, "s", "Phase 2 models"), (C3, ORANGE, "o", "Phase 3 models")):
        c = C[f"group_{t}"]["curve"]; x = [r["target_recall"] for r in c]
        ax.fill_between(x, [r["kept_lo"] for r in c], [r["kept_hi"] for r in c], color=col, alpha=0.18, lw=0)
        ax.plot(x, [r["kept_fraction"] for r in c], marker=mk, color=col, ms=3.5, lw=1.2, label=lab)
    ax.plot([0.5, 1], [0.5, 1], color=INK, lw=0.8, ls=":", label="no information (kept = recall)")
    ax.set_title(TITLE[t]); ax.set_xlabel("target recall (1 - alpha)")
axes[0].set_ylabel("fraction of test MOFs kept"); axes[0].legend(frameon=False, loc="lower right")
fig.tight_layout(); save(fig, "fig3_kept_vs_recall")
