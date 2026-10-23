"""Shared style and loaders for the paper figures. Every figure script reads ONLY saved results under data/processed/ (nothing is typed in)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P2, P3, P4 = "data/processed/phase2/", "data/processed/phase3/", "data/processed/phase4/"
TASKS = ("thermal", "solvent", "water")
TITLE = {"thermal": "Thermal stage", "solvent": "Solvent stage", "water": "Water stage"}
# Okabe-Ito colour-blind-safe colours; every series also differs in marker / line style
BLUE, ORANGE, GREEN, GREY, INK = "#0072B2", "#D55E00", "#009E73", "#6b6b6b", "#222222"
RECALL_COL = {0.80: "#56B4E9", 0.90: BLUE, 0.95: "#CC79A7"}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#888", "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "pdf.fonttype": 42, "savefig.bbox": "tight", "figure.dpi": 120})


def load(path):
    return json.load(open(path))


def save(fig, name):
    fig.savefig(f"figures/{name}.pdf")
    print("wrote", f"figures/{name}.pdf")
    if os.environ.get("FIG_PREVIEW_DIR"):                      # optional PNG preview for visual checks only (not a deliverable)
        fig.savefig(os.path.join(os.environ["FIG_PREVIEW_DIR"], name + ".png"), dpi=130)
