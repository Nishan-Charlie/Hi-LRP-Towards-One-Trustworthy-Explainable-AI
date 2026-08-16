"""Build the paper's headline comparison figure from cached attribution maps.

Renders results/qual_tmp (4 backbones x {HiLRP, Grad-CAM, Grad-CAM++, IG,
SmoothGrad}, same input image) as a publication-quality overlay grid:

    rows = Swin-B, PVT-v2, EfficientViT-B2, MobileViT-v2
    cols = Input | HiLRP (ours) | Grad-CAM | Grad-CAM++ | Integrated Grad | SmoothGrad

The point of the figure is the EfficientViT row: Grad-CAM's terminal spatial
feature map disappears under linear cross-covariance attention, so its map
diffuses (dataset Pointing 0.551, below the 0.61 random-point prior) while HiLRP
still traces the object (0.960).

No model is run: this reads the cached .npy maps only. The input is recovered
from the ImageNet-S cache (index 0, label 915 "yurt"); the 100- and 1000-image
caches share a deterministic stream order, so index 0 is the same image.

  python scripts/bench/paper_comparison_figures.py

Assumes CWD = repo root.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

TMP = os.path.join("results", "qual_tmp")
OUT_DIR = os.path.join("figures", "bench")
SIZE = 224

MODELS = [
    ("swin_base_patch4_window7_224", "Swin-B"),
    ("pvt_v2_b2", "PVT-v2-b2"),
    ("efficientvit_b2", "EfficientViT-B2"),
    ("mobilevitv2_100", "MobileViT-v2"),
]
COLS = ["hilrp", "grad_cam", "grad_cam_plus_plus", "integrated_gradients", "smoothgrad"]
TITLES = ["Input", "HiLRP (ours)", "Grad-CAM", "Grad-CAM++", "Integrated Grad", "SmoothGrad"]

# Cell to flag: (row index, column index in the full grid) -> annotation.
COLLAPSE_CELL = (2, 2)  # EfficientViT-B2 x Grad-CAM


def _heat(mp):
    """Smoothed, percentile-normalized |relevance| in [0,1] for overlay."""
    from scipy.ndimage import gaussian_filter, zoom

    h = np.abs(np.asarray(mp, dtype=np.float64))
    if h.shape[0] != SIZE:
        h = zoom(h, SIZE / h.shape[0], order=1)
    h = gaussian_filter(h, sigma=2.5)
    return np.clip(h / (np.percentile(h, 99.0) + 1e-12), 0, 1)


def _overlay(ax, rgb, heat):
    gray = rgb.mean(2, keepdims=True).repeat(3, 2) * 0.55 + 0.15
    ax.imshow(gray)
    ax.imshow(heat, cmap="turbo", alpha=np.clip(heat * 1.1, 0, 1))


def main():
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["font.family"] = "Times New Roman"
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    os.makedirs(OUT_DIR, exist_ok=True)
    rgb = np.load(os.path.join(TMP, "__input.npy"))

    n_r, n_c = len(MODELS), len(COLS) + 1
    fig, ax = plt.subplots(n_r, n_c, figsize=(1.95 * n_c, 1.95 * n_r))
    ax = np.atleast_2d(ax)

    for r, (key, label) in enumerate(MODELS):
        ax[r, 0].imshow(rgb)
        ax[r, 0].set_xticks([]); ax[r, 0].set_yticks([])
        for s in ax[r, 0].spines.values():
            s.set_visible(False)
        ax[r, 0].set_ylabel(label, fontsize=13, fontweight="bold", labelpad=8)

        for c, meth in enumerate(COLS, start=1):
            a = ax[r, c]
            p = os.path.join(TMP, f"{key}__{meth}.npy")
            if not os.path.exists(p):
                a.axis("off")
                continue
            _overlay(a, rgb, _heat(np.load(p)))
            a.axis("off")

    for c, t in enumerate(TITLES):
        ax[0, c].set_title(t, fontsize=14, fontweight="bold", pad=6)

    # Flag the Grad-CAM collapse on linear attention. The scores are dataset-level
    # (Table II), not per-image, so they are labelled as such.
    def _flag(cell, color, text):
        a = ax[cell[0], cell[1]]
        a.add_patch(Rectangle((1, 1), SIZE - 3, SIZE - 3, fill=False,
                              edgecolor=color, linewidth=3.0, zorder=10))
        a.text(0.5, 0.965, text, transform=a.transAxes, ha="center", va="top",
               fontsize=10, color="white", fontweight="bold", zorder=11,
               bbox=dict(boxstyle="round,pad=0.22", facecolor=color, edgecolor="none"))

    _flag(COLLAPSE_CELL, "#d81b1b", "dataset Pointing 0.55")
    _flag((COLLAPSE_CELL[0], 1), "#1a7f37", "dataset Pointing 0.96")

    fig.subplots_adjust(wspace=0.04, hspace=0.06, left=0.06)
    out = os.path.join(OUT_DIR, "arch_method_comparison")
    fig.savefig(out + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    print("wrote", out + ".png")


if __name__ == "__main__":
    main()
