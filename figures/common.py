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
CM = 1 / 2.54
W1, W2 = 8.5 * CM, 17.5 * CM                                    # single- and double-column widths (inches)
FS = 7                                                          # one body font size everywhere (pt); tick/legend 6.5, panel letters 9 bold
plt.rcParams.update({"font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS, "legend.fontsize": 6.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#888", "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.5,
                     "axes.axisbelow": True, "axes.titlepad": 4, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                     "font.family": "DejaVu Sans", "pdf.fonttype": 42, "figure.dpi": 120})
VARIANTS = ("2col", "1col")                                     # 2col -> figN.pdf (17.5 cm); 1col -> figN_1col.pdf (8.5 cm)


def panels(variant, n, h2=2.3, h1=2.0, sharey=False, ratios=None):
    """n panels in a row at double-column width, stacked in a column at single-column width."""
    if variant == "2col":
        return plt.subplots(1, n, figsize=(W2, h2), sharey=sharey, gridspec_kw={"width_ratios": ratios} if ratios else None, layout="constrained")
    return plt.subplots(n, 1, figsize=(W1, h1 * n), sharey=sharey, layout="constrained")


def letters(axes, dx=-0.0):
    """Bold panel letters a, b, c at the top-left corner of each axes (above the plot area, left of the centred title)."""
    for ax, ch in zip(axes, "abcdefgh"):
        ax.text(dx, 1.0, ch, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right" if dx < 0 else "left")


def load(path):
    return json.load(open(path))


def save(fig, name, variant="2col"):
    out = f"figures/{name}.pdf" if variant == "2col" else f"figures/{name}_1col.pdf"
    fig.savefig(out)
    print("wrote", out, "(%.2f x %.2f cm)" % (fig.get_size_inches()[0] / CM, fig.get_size_inches()[1] / CM),)
    if os.environ.get("FIG_PREVIEW_DIR"):                      # optional PNG preview for visual checks only (not a deliverable)
        fig.savefig(os.path.join(os.environ["FIG_PREVIEW_DIR"], os.path.basename(out)[:-4] + ".png"), dpi=200)
    plt.close(fig)
