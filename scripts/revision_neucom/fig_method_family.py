"""Taxonomy figure (fig:method_family): the 13 benchmarked methods in their four families,
with the structural assumption each family places on the backbone. Sized for one column.

usage: python scripts/revision_neucom/fig_method_family.py OUT.pdf
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "pdf.fonttype": 42, "ps.fonttype": 42})

FAMILIES = [  # name, edge, fill, text colour, assumption, methods
    ("Gradient", "#9673a6", "#e1d5e7", "#4c0099",
     "differentiable path\nfrom input to logit",
     ["Saliency", "Integrated Grad.", "Input × Grad.", "SmoothGrad", "VarGrad", "GradientSHAP"]),
    ("CAM", "#6c8ebf", "#dae8fc", "#004c99",
     "logit pooled linearly\nfrom a spatial map",
     ["Grad-CAM", "Grad-CAM++"]),
    ("Attention", "#82b366", "#d5e8d4", "#006633",
     "global class-token\nattention matrix",
     ["Attention Rollout", "AttnLRP"]),
    ("Perturbation", "#d79b00", "#ffe6cc", "#994c00",
     "none (black box);\nmany forward passes",
     ["Occlusion", "RISE", "LIME"]),
]

fig = plt.figure(figsize=(5.6, 3.9))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0.1, 1)
ax.axis("off")


def box(xc, yc, w, h, edge, fill, text, color, size, weight="normal", lw=1.6):
    ax.add_patch(FancyBboxPatch((xc - w / 2, yc - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.012",
                                linewidth=lw, edgecolor=edge, facecolor=fill, zorder=2))
    if text:
        ax.text(xc, yc, text, ha="center", va="center", fontsize=size, color=color, fontweight=weight, zorder=3)


box(0.5, 0.935, 0.62, 0.085, "#3a3a3a", "#4d4d4d", "Post-hoc attribution methods (13)", "white", 14, "bold")
xs = [0.125, 0.375, 0.625, 0.875]
W = 0.24
ax.plot([xs[0], xs[-1]], [0.85, 0.85], color="#666666", lw=1.6, zorder=1)
ax.plot([0.5, 0.5], [0.892, 0.85], color="#666666", lw=1.6, zorder=1)
for x, (name, edge, fill, tcol, assumption, methods) in zip(xs, FAMILIES, strict=True):
    ax.annotate("", xy=(x, 0.795), xytext=(x, 0.85),
                arrowprops=dict(arrowstyle="-|>", color="#666666", lw=1.6, mutation_scale=12), zorder=1)
    # family box: name, then the structural assumption in italics
    box(x, 0.715, W, 0.155, edge, fill, "", tcol, 1, lw=2.0)
    ax.text(x, 0.764, name, ha="center", va="center", fontsize=13.5, color=tcol, fontweight="bold", zorder=3)
    ax.text(x, 0.69, assumption, ha="center", va="center", fontsize=10.5, fontstyle="italic",
            color="#222222", linespacing=1.1, zorder=3)
    # methods hang from a spine on the left of the column
    top, step = 0.565, 0.083
    ys = [top - i * step for i in range(len(methods))]
    sx = x - W / 2 + 0.012
    ax.plot([sx, sx], [0.637, ys[-1]], color=edge, lw=1.6, zorder=1)
    for y, m in zip(ys, methods, strict=True):
        ax.plot([sx, sx + 0.02], [y, y], color=edge, lw=1.6, zorder=1)
        box(x + 0.012, y, W - 0.025, 0.062, edge, "white", m, "black", 11 if len(m) > 15 else 12.5)

fig.savefig(sys.argv[1] if len(sys.argv) > 1 else "XAI_Method_family.pdf", bbox_inches="tight", pad_inches=0.03)
