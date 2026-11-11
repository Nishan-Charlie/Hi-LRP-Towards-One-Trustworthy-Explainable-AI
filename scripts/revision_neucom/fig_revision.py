"""New figures for the NEUCOM revision, from results/revision_neucom/analysis/summary.json
and the per-image CSVs.

  figures/revision/gcpp_mechanism.pdf  (R3.4) per-backbone distance to the Grad-CAM++ pole
                                       and Max-Sensitivity of Grad-CAM, Grad-CAM++ and the
                                       bounded-alpha intervention
  figures/revision/capacity.pdf        (R3.1) Grad-CAM EBPG and Grad-CAM++ MS against
                                       parameter count, by architecture family
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common  # noqa: F401
from analyze_revision import NAME, HEAD, FAMILY

R = os.path.join("results", "revision_neucom")
FIG = os.path.join("figures", "revision")
S = json.load(open(os.path.join(R, "analysis", "summary.json")))
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 12.5})
HEAD_COL = {"GAP+linear": "#0072B2", "GAP+LN+linear": "#CC79A7", "GAP+LN+MLP": "#E69F00",
            "conv+GAP+LN-MLP": "#D55E00", "CLS attention": "#009E73"}
HEAD_LABEL = {"GAP+linear": "GAP+linear", "GAP+LN+linear": "GAP+LN+linear", "GAP+LN+MLP": "GAP+LN+MLP",
              "conv+GAP+LN-MLP": "Conv+GAP+MLP", "CLS attention": "CLS token"}
FAM_COL = {"CNN": "#555555", "Isotropic": "#009E73", "Hierarchical": "#0072B2",
           "Multi-axis": "#E69F00", "Hybrid": "#CC79A7", "Linear": "#D55E00"}

# ---------------------------------------------------------------- mechanism figure
G = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(R, "gcpp_*.csv"))]).drop_duplicates(["model", "idx"])
mech = S.get("mechanism", {})
order = [m for m in sorted(mech, key=lambda m: (list(HEAD_COL).index(HEAD.get(m, "GAP+linear")), mech[m].get("core_min", 0) * -1))]
ms_models = [m for m in order if "ms_grad_cam_plus_plus_med" in mech[m]]
if ms_models:
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.6), gridspec_kw=dict(width_ratios=[1.15, 1]))
else:  # robustness runs not available yet: diagnostic panel only
    fig, a0 = plt.subplots(1, 1, figsize=(9.5, 3.6))
    ax = [a0]
for i, m in enumerate(order):
    v = G[G.model == m].core_min.clip(lower=1e-3).values
    c = HEAD_COL.get(HEAD.get(m), "#999")
    ax[0].scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-0.25, 0.25, len(v)), v, s=2, color=c, alpha=0.25, lw=0)
    ax[0].plot([i - 0.32, i + 0.32], [np.median(v)] * 2, color="k", lw=1.2)
ax[0].axhline(0.5, ls="--", lw=0.8, color="#C1121F")
ax[0].text(len(order) - 0.5, 0.42, r"$|\alpha|>2$", color="#C1121F", ha="right", va="top", fontsize=11)
ax[0].set_yscale("log")
ax[0].set_ylabel(r"$\min_{k,i,j:\,g>0}\;|2 + S_k\, g^k_{ij}|$ per image")
ax[0].set_xticks(range(len(order)))
ax[0].set_xticklabels([NAME.get(m, m) for m in order], rotation=60, ha="right", fontsize=11.5)
ax[0].set_title("(a) Distance of the Grad-CAM++ weight from its pole", fontsize=13)
handles = [plt.Line2D([0], [0], marker="o", ls="", color=c, label=HEAD_LABEL[h], markersize=6) for h, c in HEAD_COL.items()
           if any(HEAD.get(m) == h for m in order)]
ax[0].legend(handles=handles, fontsize=11, frameon=False, loc="lower left", title="Class read-out", title_fontsize=11)

x = np.arange(len(ms_models))
w = 0.27
for j, (key, lab, col) in enumerate(() if not ms_models else (("ms_grad_cam_med", "Grad-CAM", "#888888"),
                                     ("ms_grad_cam_plus_plus_med", "Grad-CAM++", "#C1121F"),
                                     ("ms_grad_cam_plus_plus_stab_med", r"Grad-CAM++, $|2+S_k g|\geq 1$", "#0072B2"))):
    vals = [mech[m].get(key, np.nan) for m in ms_models]
    ax[1].bar(x + (j - 1) * w, vals, w, color=col, label=lab)
if ms_models:
    ax[1].set_yscale("log")
    ax[1].set_yticks([0.1, 0.2, 0.5, 1, 2, 5, 10, 20])
    ax[1].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax[1].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[1].set_ylabel("Max-Sensitivity, median\n(3 seeds, 10 perturbations)")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([NAME.get(m, m) for m in ms_models], rotation=60, ha="right", fontsize=11.5)
    ax[1].legend(fontsize=11, frameon=False, loc="upper left")
    ax[1].set_title("(b) Max-Sensitivity of the CAM methods", fontsize=13)
for a in ax:
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "gcpp_mechanism.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(FIG, "gcpp_mechanism.png"), dpi=160, bbox_inches="tight")
print("wrote gcpp_mechanism")

# ---------------------------------------------------------------- capacity figure
cap = S.get("capacity", {})
n_extra = sum(1 for m, v in cap.items() if m not in common.MAIN_MODELS and np.isfinite(v.get("ebpg_grad_cam") or np.nan))
if cap and n_extra >= 4:
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for k, (key, lab) in enumerate((("ebpg_grad_cam", "Grad-CAM EBPG (dense mask)"),
                                    ("ms_grad_cam_plus_plus_med", "Grad-CAM++ Max-Sensitivity (median)"))):
        for fam, col in FAM_COL.items():
            pts = sorted([(v["params"], v.get(key)) for m, v in cap.items() if v["family"] == fam
                          and v.get(key) is not None and np.isfinite(v.get(key))])
            if not pts:
                continue
            ax[k].plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=col, ms=5, lw=1.2, label=fam)
        for m, v in cap.items():
            if v.get(key) is not None and np.isfinite(v.get(key)):
                ax[k].annotate(NAME.get(m, m), (v["params"], v[key]), fontsize=8, xytext=(3, 3), textcoords="offset points")
        ax[k].set_xscale("log")
        if key.startswith("ms_"):
            ax[k].set_yscale("log")
        ax[k].set_xlabel("Parameters (M, log scale)")
        ax[k].set_ylabel(lab)
        ax[k].axvspan(18, 32, color="#eeeeee", zorder=0)
        for s in ("top", "right"):
            ax[k].spines[s].set_visible(False)
    ax[0].axhline(S.get("mask_area", {}).get("n1000", np.nan), ls="--", lw=0.8, color="#999")
    ax[0].legend(fontsize=9.5, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "capacity.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "capacity.png"), dpi=160, bbox_inches="tight")
    print("wrote capacity")
