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
import common  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analyze_revision import HEAD, NAME

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
ax[0].set_xticklabels([NAME.get(m, m) for m in order], rotation=90, ha="center", fontsize=10.5)
ax[0].set_title("(a) Distance of the Grad-CAM++ weight from its pole", fontsize=13)
handles = [plt.Line2D([0], [0], marker="o", ls="", color=c, label=HEAD_LABEL[h], markersize=6) for h, c in HEAD_COL.items()
           if any(HEAD.get(m) == h for m in order)]
ax[0].legend(handles=handles, fontsize=11, frameon=False, loc="lower left", title="Class read-out", title_fontsize=11)

x = np.arange(len(ms_models))
w = 0.27
TOP = 40.0   # an infinite median (the method fails on at least half the images) is drawn to here
inf_at = set()
for j, (key, lab, col) in enumerate(() if not ms_models else (("ms_grad_cam_med", "Grad-CAM", "#888888"),
                                     ("ms_grad_cam_plus_plus_med", "Grad-CAM++", "#C1121F"),
                                     ("ms_grad_cam_plus_plus_stab_med", r"Grad-CAM++, bounded", "#0072B2"))):
    vals = np.array([mech[m].get(key, np.nan) for m in ms_models], dtype=float)
    fin = np.where(np.isinf(vals), np.nan, vals)
    ax[1].bar(x + (j - 1) * w, fin, w, color=col, label=lab)
    for k in np.flatnonzero(np.isinf(vals)):
        ax[1].bar(x[k] + (j - 1) * w, TOP, w, color="white", edgecolor=col, hatch="////", lw=0.8)
        inf_at.add(int(k))
for k in inf_at:   # one label per backbone: median unbounded (fails on at least half the images)
    ax[1].text(x[k] + w / 2, TOP * 1.08, r"$\infty$", ha="center", va="bottom", fontsize=13)
if ms_models:
    ax[1].set_yscale("log")
    ax[1].set_ylim(top=TOP * 1.9)
    ax[1].set_yticks([0.1, 0.2, 0.5, 1, 2, 5, 10, 20])
    ax[1].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax[1].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[1].set_ylabel("Max-Sensitivity, median\n(3 seeds, 10 perturbations)")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([NAME.get(m, m) for m in ms_models], rotation=90, ha="center", fontsize=10.5)
    ax[1].legend(fontsize=10.5, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.93))
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
# Label offsets (points) where two backbones of similar size and score would overlap.
_E, _M = "ebpg_grad_cam", "ms_grad_cam_plus_plus_med"
CAP_OFFSET = {(_E, "resnet18"): (-40, 8), (_E, "deit_small_patch16_224"): (-38, 6),
              (_E, "convnext_tiny"): (5, -12), (_E, "mobilevitv2_200"): (-80, 5),
              (_E, "resnet50"): (6, 4), (_E, "maxvit_tiny_tf_224"): (5, -13),
              (_E, "vit_small_patch16_224"): (-54, -10), (_E, "swin_tiny_patch4_window7_224"): (5, 3),
              (_M, "resnet18"): (-22, 7), (_M, "resnet50"): (-20, 10), (_M, "mobilevitv2_200"): (-30, 13),
              (_M, "pvt_v2_b2"): (-12, -15), (_M, "swin_tiny_patch4_window7_224"): (5, 4),
              (_M, "resnet101"): (4, -15), (_M, "pvt_v2_b4"): (-12, 7),
              (_M, "deit_small_patch16_224"): (5, -13), (_M, "convnext_tiny"): (5, -4),
              (_M, "deit_base_patch16_224"): (6, 3), (_M, "convnext_base"): (4, -13),
              (_M, "maxvit_tiny_tf_224"): (-64, -15), (_M, "maxvit_small_tf_224"): (-30, -15)}
CAP_TOP = 40.0   # an infinite median (fails on at least half the images) is drawn at this level
if cap and n_extra >= 4:
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for k, (key, lab) in enumerate((("ebpg_grad_cam", "Grad-CAM EBPG (dense mask)"),
                                    ("ms_grad_cam_plus_plus_med", "Grad-CAM++ Max-Sensitivity (median)"))):
        for fam, col in FAM_COL.items():
            pts = sorted([(v["params"], v[key]) for m, v in cap.items() if v["family"] == fam
                          and v.get(key) is not None and not np.isnan(v[key])])
            fin = [p for p in pts if np.isfinite(p[1])]
            if fin:
                ax[k].plot([p[0] for p in fin], [p[1] for p in fin], "-o", color=col, ms=6, lw=1.4,
                           label=fam if k == 0 else None)
            inf = [p for p in pts if np.isinf(p[1])]
            if inf:   # open markers on the infinity line, not dropped from a log axis
                ax[k].plot([p[0] for p in inf], [CAP_TOP] * len(inf), "o", mfc="white", mec=col, mew=1.6, ms=7)
        for m, v in cap.items():
            val = v.get(key)
            if val is None or np.isnan(val):
                continue
            y = CAP_TOP if np.isinf(val) else val
            ax[k].annotate(NAME.get(m, m), (v["params"], y), fontsize=10,
                           xytext=CAP_OFFSET.get((key, m), (4, 4)), textcoords="offset points")
        ax[k].set_xscale("log")
        if key.startswith("ms_"):
            ax[k].set_yscale("log")
            if any(np.isinf(v.get(key) or 0) for v in cap.values()):
                ax[k].axhline(CAP_TOP, ls=":", lw=0.9, color="#999")
                ax[k].text(0.01, CAP_TOP * 1.25, r"$\infty$: fails on at least half the images",
                           transform=ax[k].get_yaxis_transform(), ha="left", fontsize=10, color="#555")
                ax[k].set_ylim(top=CAP_TOP * 2.2)
            ax[k].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax[k].set_xlabel("Parameters (M, log scale)")
        ax[k].set_ylabel(lab)
        ax[k].axvspan(18, 32, color="#eeeeee", zorder=0)
        for s in ("top", "right"):
            ax[k].spines[s].set_visible(False)
    ax[0].axhline(S.get("mask_area", {}).get("n1000", np.nan), ls="--", lw=0.8, color="#999")
    ax[0].text(1.0, S.get("mask_area", {}).get("n1000", np.nan) - 0.012, "random map",
               transform=ax[0].get_yaxis_transform(), ha="right", va="top", fontsize=10, color="#555")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=6, fontsize=11,
               frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "capacity.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "capacity.png"), dpi=160, bbox_inches="tight")
    print("wrote capacity")
