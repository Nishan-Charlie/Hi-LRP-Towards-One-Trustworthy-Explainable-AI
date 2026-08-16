"""Shared publication style for the paper figures.

Why this exists
---------------
The paper is IEEEtran (lettersize, journal, two-column), so a figure placed at
`width=\\linewidth` is rendered at 3.5 in and a `figure*` at `width=\\textwidth`
at 7.16 in. Several figures were authored 7-13 in wide and then scaled down by
LaTeX, which divides every font size by the same factor: an 8 pt tick label
authored on a 13 in canvas prints at ~2 pt. That is the "tiny unreadable text"
problem, and no amount of raising `fontsize=` fixes it on its own.

The rule this module enforces: AUTHOR AT THE PRINTED SIZE. Pick `COL_W` or
`FULL_W` as the figure width, then 8 pt in the script is 8 pt on paper. Font
sizes below are chosen against the 10 pt IEEEtran body text: labels and titles
at 8 pt, ticks at 7 pt, which is the smallest IEEE accepts.

Also holds the display-name maps, so a label reads "Swin-B" or "Stage 3
(14x14)" instead of the raw identifier `swin_base_patch4_window7_224` or
`stage2_in_14`.

Usage:
    import sys, os
    sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
    from figstyle import apply_paper_style, COL_W, FULL_W, model_label
    apply_paper_style()
"""
import os

# ---------------------------------------------------------------- canvas sizes
COL_W = 3.50    # IEEEtran lettersize \columnwidth, inches  (figure,  \linewidth)
FULL_W = 7.16   # IEEEtran lettersize \textwidth,   inches  (figure*, \textwidth)

# ------------------------------------------------------------------ font sizes
FS_TICK = 7.0
FS_LABEL = 8.0
FS_TITLE = 8.0     # per-axes titles (column headers in the map grids)
FS_SUPTITLE = 9.0
FS_LEGEND = 7.0
FS_ANNOT = 7.0

# --------------------------------------------------------------- colour tokens
GREEN, VERM, GREY, BLUE, RED = "#009E73", "#D55E00", "#8A8A8A", "#1F77B4", "#D62728"


def apply_paper_style(serif=True):
    """Set rcParams once per process. `serif` matches the IEEEtran body face."""
    import matplotlib
    matplotlib.use("Agg")
    family = (["Times New Roman", "DejaVu Serif"] if serif
              else ["Arial", "DejaVu Sans"])
    matplotlib.rcParams.update({
        "font.family": "serif" if serif else "sans-serif",
        ("font.serif" if serif else "font.sans-serif"): family,
        "mathtext.fontset": "cm",
        "font.size": FS_LABEL,
        "axes.labelsize": FS_LABEL,
        "axes.titlesize": FS_TITLE,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEGEND,
        "figure.titlesize": FS_SUPTITLE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.2,
        "savefig.dpi": 400,
        "figure.dpi": 400,
    })


def lxt_transformers_shim():
    """Restore `find_pruneable_heads_and_indices`, dropped in transformers 5.x.

    `lxt.efficient.core` imports `lxt.efficient.models`, whose bert module needs
    the symbol. HiLRP passes its own patch_map and never uses lxt's default
    maps, so re-supplying the helper is enough and avoids pinning transformers
    back. Same shim as scripts/hilrp/clip_custom_images.py.
    """
    import torch
    import transformers.pytorch_utils as _pu
    if hasattr(_pu, "find_pruneable_heads_and_indices"):
        return
    def _fp(heads, n_heads, head_size, already):
        mask = torch.ones(n_heads, head_size)
        heads = set(heads) - already
        for h in heads:
            h = h - sum(1 for x in already if x < h)
            mask[h] = 0
        mask = mask.view(-1).contiguous().eq(1)
        return heads, torch.arange(len(mask))[mask].long()
    _pu.find_pruneable_heads_and_indices = _fp


def repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------- display names
MODEL_LABELS = {
    "swin_base_patch4_window7_224": "Swin-B",
    "swin_tiny_patch4_window7_224": "Swin-T",
    "swin_small_patch4_window7_224": "Swin-S",
    "pvt_v2_b2": "PVT-v2-B2",
    "efficientvit_b1": "EfficientViT-B1",
    "efficientvit_b2": "EfficientViT-B2",
    "mobilevitv2_100": "MobileViT-v2",
    "maxvit_small_tf_224": "MaxViT-S",
    "vit_base_patch16_224": "ViT-B",
    "deit_base_patch16_224": "DeiT-B",
}

# Randomization blocks, output side first (sanity check). Numbering follows the
# module tree the script walks, so it matches the existing paper text.
BLOCK_LABELS = {
    "head": "Head",
    "stage3": "Stage 3",
    "stage2": "Stage 2",
    "stage1": "Stage 1",
    "stage0": "Stage 0",
    "patch_embed": "Patch Embed",
    "original": "Intact Model",
}

# Swin stage captures, with the token grid they live on. Stage numbering is the
# one already used in the paper (stage1_in_28 = "Stage 1"), not renumbered.
STAGE_LABELS = {
    "pixels": "Pixels\n224$\\times$224",
    "tokens_56": "Tokens\n56$\\times$56",
    "stage1_in_28": "Stage 1\n28$\\times$28",
    "stage2_in_14": "Stage 2\n14$\\times$14",
    "stage3_in_7": "Stage 3\n7$\\times$7",
    "final_7": "Head\n7$\\times$7",
}

MODE_LABELS = {"independent": "Independent", "cascading": "Cascading"}


def model_label(name):
    """Pretty model name; falls back to a Title-Cased identifier."""
    if name in MODEL_LABELS:
        return MODEL_LABELS[name]
    for key, val in MODEL_LABELS.items():
        if name.startswith(key.split("_")[0]):
            return val
    return name.replace("_", " ").title()


def block_label(name):
    return BLOCK_LABELS.get(name, name.replace("_", " ").title())


def stage_label(name):
    return STAGE_LABELS.get(name, name.replace("_", " ").title())


def mode_label(name):
    return MODE_LABELS.get(name, name.title())


def save(fig, stem, also_pdf=True, dpi=400):
    """Write `figures/hilrp/<stem>.png` (+ .pdf), CWD = repo root."""
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    fig.savefig(f"{stem}.png", dpi=dpi, bbox_inches="tight")
    if also_pdf:
        fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.png" + (f" + {stem}.pdf" if also_pdf else ""))
