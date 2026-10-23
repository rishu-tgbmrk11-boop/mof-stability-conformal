"""Fig 2: realised test recall vs target recall for the per-stage split conformal rule (group-aware test sets; single calibration draw; 95% CI over test groups)."""
import sys
sys.path.insert(0, "figures")
from common import *

C2, C3 = load(P2 + "conformal.json"), load(P3 + "conformal.json")


def draw(variant):
    fig, axes = panels(variant, 3, h2=2.3, h1=1.9, sharey=True)
    for ax, t in zip(axes, TASKS):
        ax.plot([0.5, 1], [0.5, 1], color=INK, lw=0.7, ls="--", label="target (y = x)")
        for C, col, mk, lab in ((C2, GREY, "s", "Phase 2 models"), (C3, ORANGE, "o", "Phase 3 models")):
            c = C[f"group_{t}"]["curve"]; x = [r["target_recall"] for r in c]
            ax.fill_between(x, [r["recall_lo"] for r in c], [r["recall_hi"] for r in c], color=col, alpha=0.18, lw=0)
            ax.plot(x, [r["recall"] for r in c], marker=mk, color=col, ms=3, lw=1.0, label=lab)
        ax.set_title(TITLE[t]); ax.set_ylim(0.2, 1.05)
        ax.set_ylabel("realised test recall") if (variant == "1col" or t == "thermal") else None
        ax.set_xlabel("target recall (1 − α)") if (variant == "2col" or t == "water") else None
    letters(axes, dx=-0.0)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="outside lower center", ncol=3 if variant == "2col" else 2)
    save(fig, "fig2_recall_vs_target", variant)


for v in VARIANTS:
    draw(v)
