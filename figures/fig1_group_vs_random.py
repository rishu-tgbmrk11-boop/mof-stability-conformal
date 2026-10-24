"""Fig 1: group-aware vs random split, as RANGES over many splits (Phase 5 section 5). Median with the 2.5-97.5% range over splits.
Random forest, N = 100 splits (filled markers); released networks (fixed restore, thermal and solvent) + released forest (water), N = 10 splits (open markers).
Each split seed gives one random-group split and one MOF-level random split (60/20/20); test parts are about 20% of MOFs and differ between splits."""
import sys
sys.path.insert(0, "figures")
from common import *
import pandas as pd

RF, NN = pd.read_parquet(P5 + "split_ranges_rf.parquet"), pd.read_parquet(P5 + "split_ranges_nn.parquet")
PANELS = [("AUC, 'truly stable'", "auc_stable", ("thermal", "solvent", "water"), (0.4, 0.95)),
          ("Accuracy of stable call", "accuracy_stable", ("thermal", "solvent", "water"), (0.45, 0.85)),
          ("MAE (°C)", "mae_degC", ("thermal",), None)]
SPEC = [(RF, "random", -0.27, GREY, "s", True), (RF, "group", -0.09, ORANGE, "o", True), (NN, "random", 0.09, GREY, "s", False), (NN, "group", 0.27, ORANGE, "o", False)]


def draw(variant):
    fig, axes = panels(variant, 3, h2=2.5, h1=2.0, ratios=[3, 3, 1.3] if variant == "2col" else None)
    for ax, (title, key, stages, lim) in zip(axes, PANELS):
        for j, t in enumerate(stages):
            for df, mode, off, col, mk, filled in SPEC:
                v = df[(df.stage == t) & (df["mode"] == mode)][key].dropna()
                med, lo, hi = v.median(), v.quantile(0.025), v.quantile(0.975)
                ax.errorbar(j + off, med, yerr=[[med - lo], [hi - med]], fmt=mk, color=col, mfc=col if filled else "white", mec=col, ms=3.6, capsize=1.6, lw=0.9 if filled else 0.7, zorder=3 if filled else 2)
        ax.set_xticks(range(len(stages))); ax.set_xticklabels([TITLE[t].split()[0] for t in stages]); ax.set_title(title); ax.set_xlim(-0.6, len(stages) - 0.4)
        if lim: ax.set_ylim(*lim)
    letters(axes)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="s", color=GREY, ms=4, lw=0, label="random split (MOF-level)"), Line2D([], [], marker="o", color=ORANGE, ms=4, lw=0, label="group-aware split"),
         Line2D([], [], marker="o", color="#555", mfc="#555", ms=4, lw=0, label=f"random forest, {RF.split_seed.nunique()} splits"), Line2D([], [], marker="o", color="#555", mfc="white", ms=4, lw=0, label=f"released networks / forest, {NN.split_seed.nunique()} splits")]
    fig.legend(handles=h, frameon=False, loc="outside lower center", ncol=2 if variant == "2col" else 1)
    save(fig, "fig1_group_vs_random", variant)


for v in VARIANTS:
    draw(v)
