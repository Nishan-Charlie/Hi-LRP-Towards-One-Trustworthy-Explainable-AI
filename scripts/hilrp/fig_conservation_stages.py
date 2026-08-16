"""Figure: HiLRP relevance conservation across Swin-B resolution stages.

Reads results/hilrp_gate2real/conservation.csv (per-image relevance sum / logit at
each capture) and plots mean +/- std per stage against the exact-conservation line.
Directly visualizes the paper's central quantitative claim: conservation is exact at
the token level and deviates, monotonically and by a measured amount, only at the
convolutional pixel stem.

  python scripts/hilrp/fig_conservation_stages.py    (CWD = repo root)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))                       # <repo>/scripts
from figstyle import (apply_paper_style, save, stage_label, COL_W,
                      FS_ANNOT, GREEN, VERM, GREY)

apply_paper_style()
import matplotlib.pyplot as plt

cons = pd.read_csv("results/hilrp_gate2real/conservation.csv")
stages = ["tokens_56", "stage1_in_28", "stage2_in_14", "stage3_in_7", "final_7", "pixels"]
labels = [stage_label(s) for s in stages]
mean, std = cons[stages].mean().values, cons[stages].std().values
x = np.arange(len(stages))

# Authored at the printed column width so the 7-8 pt type stays 7-8 pt on paper.
fig, ax = plt.subplots(figsize=(COL_W, 2.5))
ax.axhline(1.0, color=GREY, ls="--", lw=1.0, zorder=1, label="Exact conservation")
ax.fill_between(x, mean - std, mean + std, color=GREEN, alpha=0.15, zorder=1)
ax.errorbar(x, mean, yerr=std, marker="o", ms=4, lw=1.4, color=GREEN, capsize=2,
            zorder=3, label=r"HiLRP  $\sum_i R_i / f_c(x)$")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("Relevance Sum / Logit"); ax.set_ylim(0.55, 1.55)
ax.set_title(f"Conservation Across Swin-B Resolution Stages ($n={len(cons)}$)")
ax.annotate("Bounded inflation\nat deep stages", (3, mean[3]), (1.85, 1.44),
            fontsize=FS_ANNOT, color=GREEN, ha="center",
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=0.8))
ax.annotate("$\\gamma$-rule deviation\nat the conv. stem", (5, mean[-1]), (3.9, 0.63),
            fontsize=FS_ANNOT, color=VERM, ha="center",
            arrowprops=dict(arrowstyle="->", color=VERM, lw=0.8))
ax.legend(frameon=False, loc="lower left", handlelength=1.6)
fig.tight_layout(pad=0.3)
save(fig, "figures/hilrp/conservation_stages")
print("stage means:", dict(zip(stages, np.round(mean, 3))))
