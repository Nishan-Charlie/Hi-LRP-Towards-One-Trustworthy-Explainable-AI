"""Figures for the main HiLRP paper (paper.tex) that must NOT reuse the benchmark
paper's figure forms. Two replacements:

  hilrp_assumption_matrix.{pdf,png}  method-family x backbone-family support matrix
                                     (replaces bench_taxonomy in paper.tex only)
  hilrp_fc_null.{pdf,png}            FC cell means vs the per-image noise density
                                     (replaces bench_fc_noise in paper.tex only)

Same Okabe-Ito palette as the benchmark scripts (validated CVD-safe). Marks are
never color-only: holds/degraded/fails use distinct glyph shapes.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# benchmark_diagrams lives under scripts/bench/ after the 2026-07-15 split
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench"))
from benchmark_diagrams import load  # noqa: E402  (CSV + GradientSHAP override)

os.makedirs("figures", exist_ok=True)

C_HOLD, C_DEGR, C_FAIL, C_OURS = "#0072B2", "#E69F00", "#D55E00", "#009E73"


# ── 1. Assumption-coverage matrix ────────────────────────────────────────────
def fig_assumption_matrix():
    """Traffic-light support matrix: method families (rows) x backbone families
    (columns). Each cell carries a colour tint and a distinct glyph (colour-blind
    safe). The HiLRP row is the only one solid across the whole grid."""
    from matplotlib.patches import FancyBboxPatch

    # Short column key; full names live in the caption.
    cols = ["CNN", "ViT", "Swin", "PVT", "MaxViT", "MobViT", "EffViT"]
    # (family, assumption, per-column status)  H=holds  D=degraded  F=fails
    rows = [
        ("Gradient",         "smooth, stable input gradients",  "HDDDDDD"),
        ("CAM",              "a terminal spatial feature map",  "HDHHHHF"),
        ("Attention-native", "global softmax attention + CLS",  "FHFFFFF"),
        ("Perturbation",     "cheap output access ($10^2$x+)",  "HHHHHHH"),
        ("Classic LRP",      "a hand rule per module type",     "HFFFFFF"),
        ("HiLRP (ours)",     "four conserving primitives",      "HHHHHHH"),
    ]
    nr, nc = len(rows), len(cols)

    FACE = {"H": "#D6ECDB", "D": "#FBE7C6", "F": "#F6D6D0"}
    EDGE = {"H": "#1F7A44", "D": "#B4740C", "F": "#B23A2E"}
    GLYPH = {"H": "✓", "D": "◐", "F": "✗"}   # check / half / cross
    FACE_OURS, EDGE_OURS = "#BEE7CF", C_OURS

    cw = 1.0
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.set_xlim(-3.5, nc + 0.15)
    ax.set_ylim(nr + 0.9, -1.25)      # inverted: row 0 on top, header band above
    ax.axis("off")

    # Column header band.
    ax.add_patch(plt.Rectangle((-0.02, -1.02), nc * cw + 0.04, 0.9,
                               facecolor="#2B3A4A", edgecolor="none", zorder=1))
    for j, c in enumerate(cols):
        ax.text(j * cw + cw / 2, -0.57, c, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color="white", zorder=2)
    ax.text(-3.42, -0.57, "Backbone family", ha="left", va="center",
            fontsize=10.5, style="italic", color="#2B3A4A")

    for i, (fam, assume, states) in enumerate(rows):
        ours = fam.startswith("HiLRP")
        if ours:
            ax.add_patch(plt.Rectangle((-3.5, i - 0.02), nc * cw + 3.65, cw,
                                       facecolor=C_OURS, alpha=0.10,
                                       edgecolor="none", zorder=0))
        ax.text(-3.42, i + 0.30, fam, ha="left", va="center", fontsize=13,
                fontweight="bold", color=EDGE_OURS if ours else "#1A2433", zorder=2)
        ax.text(-3.42, i + 0.70, ("covers " if ours else "needs ") + assume,
                ha="left", va="center", fontsize=8.0, style="italic",
                color=("#2E7D50" if ours else "#7A828C"), zorder=2)
        for j, s in enumerate(states):
            face = FACE_OURS if ours else FACE[s]
            edge = EDGE_OURS if ours else EDGE[s]
            ax.add_patch(FancyBboxPatch(
                (j * cw + 0.07, i + 0.07), cw - 0.14, cw - 0.14,
                boxstyle="round,pad=0,rounding_size=0.10", mutation_aspect=1.0,
                facecolor=face, edgecolor=edge, linewidth=1.1, zorder=2))
            ax.text(j * cw + cw / 2, i + cw / 2, GLYPH[s], ha="center",
                    va="center", fontsize=17, fontweight="bold", color=edge,
                    zorder=3)

    # Legend row below the grid.
    ly = nr + 0.55
    for lx, s, txt in [(-3.42, "H", "assumption holds"),
                       (-0.55, "D", "runs, degraded / unstable"),
                       (2.75, "F", "undefined / fails to run")]:
        ax.add_patch(FancyBboxPatch((lx, ly - 0.22), 0.44, 0.44,
                     boxstyle="round,pad=0,rounding_size=0.08", facecolor=FACE[s],
                     edgecolor=EDGE[s], linewidth=1.1, clip_on=False, zorder=3))
        ax.text(lx + 0.22, ly, GLYPH[s], ha="center", va="center", fontsize=12,
                fontweight="bold", color=EDGE[s], clip_on=False, zorder=4)
        ax.text(lx + 0.58, ly, txt, ha="left", va="center", fontsize=9.5,
                color="#333", clip_on=False, zorder=4)

    fig.tight_layout(pad=0.2)
    fig.savefig("figures/hilrp/hilrp_assumption_matrix.pdf", bbox_inches="tight")
    fig.savefig("figures/hilrp/hilrp_assumption_matrix.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/hilrp/hilrp_assumption_matrix.pdf")


# ── 2. FC cell means vs the per-image noise density ─────────────────────────
MODELS_10 = ["resnet50", "convnext_base", "vit_base_patch16_224",
             "deit_base_patch16_224", "swin_base_patch4_window7_224",
             "pvt_v2_b2", "maxvit_small_tf_224", "mobilevitv2_100",
             "efficientvit_b1", "efficientvit_b2"]
METHOD_KEYS = ["saliency", "integrated_gradients", "input_x_gradient",
               "smoothgrad", "vargrad", "gradient_shap", "grad_cam",
               "grad_cam_plus_plus", "attention_rollout", "attention_gradients",
               "occlusion", "rise", "lime"]


def fig_fc_null(get):
    means, stds = [], []
    for mod in MODELS_10:
        for k in METHOD_KEYS:
            v = get(mod, k, "faithfulness_correlation_mean")
            if v is not None:
                means.append(v)
            s = get(mod, k, "faithfulness_correlation_std")
            if s is not None:
                stds.append(s)
    means = np.array(means)
    sigma = float(np.median(stds))
    print(f"  FC cells: n={len(means)}, mean={means.mean():+.4f}, "
          f"max|m|={np.abs(means).max():.3f}, median per-image std={sigma:.3f}")

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    x = np.linspace(-0.75, 0.75, 500)
    noise = np.exp(-0.5 * (x / sigma) ** 2)
    ax.fill_between(x, noise, color="#e4e4e4", zorder=1)
    ax.plot(x, noise, color="#b5b5b5", lw=1.0, zorder=2)
    # observed cell means, peak-normalized histogram
    h, edges = np.histogram(means, bins=np.arange(-0.75, 0.7501, 0.02))
    h = h / h.max()
    ax.bar((edges[:-1] + edges[1:]) / 2, h, width=0.019, color=C_HOLD,
           edgecolor="white", lw=0.3, zorder=3)
    lo, hi = means.min(), means.max()
    ax.annotate("All %d model-method cells\nland in $[%+.3f,\\,%+.3f]$"
                % (len(means), lo, hi),
                xy=(0.075, 0.32), xytext=(0.17, 0.80), fontsize=7,
                color=C_HOLD, ha="left", va="center", linespacing=1.3,
                arrowprops=dict(arrowstyle="-|>", color=C_HOLD, lw=0.9))
    ax.text(-0.44, 0.42, "Per-image score noise\n$\\sigma_{\\mathrm{med}}=%.2f$" % sigma,
            fontsize=7, color="#888", ha="center", va="center", linespacing=1.3)
    ax.set_xlim(-0.75, 0.75)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=7, length=0)
    ax.set_xlabel("Faithfulness Correlation", fontsize=8)
    ax.set_ylabel("Relative Density", fontsize=8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#bbb")
    fig.tight_layout(pad=0.3)
    fig.savefig("figures/hilrp/hilrp_fc_null.pdf", bbox_inches="tight")
    fig.savefig("figures/hilrp/hilrp_fc_null.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/hilrp/hilrp_fc_null.pdf")


if __name__ == "__main__":
    get = load()
    fig_assumption_matrix()
    fig_fc_null(get)
