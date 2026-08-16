"""Additional diagrams for the benchmark paper, built from all_results_sofar.csv
(same single source of truth and dedup logic as benchmark_figures.py).

Produces (figures/):
  bench_taxonomy.pdf    method-family taxonomy card (fills fig:method_family)
  attnlrp_flow.pdf      AttnLRP backward relevance flow through MHSA (fills fig:attn_lrp)
  bench_rankflip.pdf    Pointing-Game rank slopegraph across four backbone families
  bench_fc_noise.pdf    FC means vs the per-image noise band (metric-failure figure)
  bench_pipeline.pdf    benchmark framework pipeline overview (sec:framework)

Same Okabe-Ito family palette as benchmark_figures.py (validated CVD-safe);
every figure direct-labels methods, so color is never the only identity carrier.
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

os.makedirs("figures", exist_ok=True)

FAMILY_COLORS = {"Gradient": "#0072B2", "CAM": "#E69F00",
                 "Attention": "#009E73", "Perturb.": "#D55E00"}
METHODS = [
    ("saliency", "Saliency", "Gradient"),
    ("integrated_gradients", "Integrated Grad", "Gradient"),
    ("input_x_gradient", "Input$\\times$Grad", "Gradient"),
    ("smoothgrad", "SmoothGrad", "Gradient"),
    ("vargrad", "VarGrad", "Gradient"),
    ("gradient_shap", "GradientSHAP", "Gradient"),
    ("grad_cam", "Grad-CAM", "CAM"),
    ("grad_cam_plus_plus", "Grad-CAM++", "CAM"),
    ("attention_rollout", "Attn Rollout", "Attention"),
    ("attnlrp", "AttnLRP", "Attention"),
    ("occlusion", "Occlusion", "Perturb."),
    ("rise", "RISE", "Perturb."),
    ("lime", "LIME", "Perturb."),
]
MODELS = [
    ("resnet50", "RN-50"),
    ("vit_base_patch16_224", "ViT-B"),
    ("swin_base_patch4_window7_224", "Swin-B"),
    ("pvt_v2_b2", "PVT-v2"),
    ("maxvit_small_tf_224", "MaxViT-S"),
    ("mobilevitv2_100", "MobViT"),
    ("efficientvit_b1", "EffViT-B1"),
    ("efficientvit_b2", "EffViT-B2"),
]


# GradientSHAP rerun (2026-07-14): updated values from results_tables_generated.tex;
# the CSV still holds the old partial run, so the table row overrides it wholesale.
GRADIENTSHAP = {
    "pointing_game_mean": {
        "resnet50": .92, "vit_base_patch16_224": .70,
        "swin_base_patch4_window7_224": .73, "pvt_v2_b2": .70,
        "maxvit_small_tf_224": .65, "mobilevitv2_100": .88,
        "efficientvit_b1": .86, "efficientvit_b2": .84},
    "faithfulness_correlation_mean": {
        "resnet50": -.014, "vit_base_patch16_224": .018,
        "swin_base_patch4_window7_224": .005, "pvt_v2_b2": .050,
        "maxvit_small_tf_224": .030, "mobilevitv2_100": -.010,
        "efficientvit_b1": -.020, "efficientvit_b2": .015},
}


RESULTS_CSV = os.path.join("benchmark_results", "all_results_sofar.csv")


def load():
    rows = [r for r in csv.DictReader(open(RESULTS_CSV)) if r["model"] and r["method"]]
    d = {}
    for r in rows:
        prev = d.get(r["model"], {}).get(r["method"])
        if prev is not None and "rerun" in prev["source_run"] and "rerun" not in r["source_run"]:
            continue
        d.setdefault(r["model"], {})[r["method"]] = r

    def get(model, method, col):
        if method == "gradient_shap" and col in GRADIENTSHAP:
            return GRADIENTSHAP[col].get(model)
        r = d.get(model, {}).get(method)
        if not r:
            return None
        v = r.get(col, "")
        return float(v) if v not in ("", "nan", None) else None
    return get


# ── 1. Method-family taxonomy ────────────────────────────────────────────────
def fig_taxonomy():
    bands = [
        ("Gradient-based", "Gradient",
         ["Saliency", "Integrated Grad", "Input$\\times$Grad",
          "SmoothGrad", "VarGrad", "GradientSHAP"],
         "backpropagated class-score gradients; assumes smooth input gradients"),
        ("CAM-based", "CAM",
         ["Grad-CAM", "Grad-CAM++"],
         "gradient-weighted activations; assumes a terminal spatial feature map"),
        ("Attention-native", "Attention",
         ["Attention Rollout", "AttnLRP"],
         "traces the token graph; assumes explicit softmax attention matrices"),
        ("Perturbation-based", "Perturb.",
         ["Occlusion", "RISE", "LIME"],
         "black-box output probing; forward passes only, computationally costly"),
    ]
    fig, ax = plt.subplots(figsize=(3.5, 3.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    band_h = 0.235
    for bi, (name, fam, methods, note) in enumerate(bands):
        y0 = 0.995 - bi * (band_h + 0.018)
        c = FAMILY_COLORS[fam]
        ax.add_patch(FancyBboxPatch((0.005, y0 - band_h), 0.99, band_h,
                                    boxstyle="round,pad=0.004,rounding_size=0.012",
                                    facecolor=c, alpha=0.08, edgecolor=c, lw=1.0))
        ax.add_patch(plt.Rectangle((0.005, y0 - band_h), 0.014, band_h,
                                   facecolor=c, edgecolor="none"))
        ax.text(0.045, y0 - 0.035, name, fontsize=8.5, fontweight="bold",
                color=c, va="top")
        # method chips, wrapped at 3 per row
        cx, cy = 0.045, y0 - 0.105
        for mi, m in enumerate(methods):
            if mi and mi % 3 == 0:
                cx, cy = 0.045, cy - 0.062
            w = 0.032 + 0.0158 * len(m.replace("$\\times$", "x"))
            ax.add_patch(FancyBboxPatch((cx, cy - 0.024), w, 0.048,
                                        boxstyle="round,pad=0.003,rounding_size=0.01",
                                        facecolor="white", edgecolor=c, lw=0.8))
            ax.text(cx + w / 2, cy, m, fontsize=6.6, ha="center", va="center",
                    color="#222")
            cx += w + 0.022
        ax.text(0.045, y0 - band_h + 0.012, note, fontsize=6.0, style="italic",
                color="#666", va="bottom")
    fig.tight_layout(pad=0.2)
    fig.savefig("figures/bench/bench_taxonomy.pdf", bbox_inches="tight")
    fig.savefig("figures/bench/bench_taxonomy.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/bench/bench_taxonomy.pdf")


# ── 2. AttnLRP relevance flow through one attention head ────────────────────
def fig_attnlrp_flow():
    FWD, REL = "#8b9bb4", "#b2182b"
    fig, ax = plt.subplots(figsize=(3.5, 3.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x, y, w, h, text, fc="white", ec="#44546a", fs=7.5, bold=False):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0.004,rounding_size=0.012",
                                    facecolor=fc, edgecolor=ec, lw=1.0, zorder=3))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=4,
                fontweight="bold" if bold else "normal", color="#1a2433")

    def arrow(p0, p1, color, lw=1.4, style="-|>", ls="-", z=2):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, color=color,
                                     lw=lw, linestyle=ls, zorder=z,
                                     mutation_scale=9, shrinkA=2, shrinkB=2))

    # forward column (left) top->bottom
    xf = 0.30
    box(xf, 0.94, 0.42, 0.075, "input tokens $\\mathbf{X}$", fc="#eef2f7", bold=True)
    box(xf, 0.78, 0.50, 0.075, "$\\mathbf{Q},\\mathbf{K},\\mathbf{V}=\\mathbf{X}W_{Q,K,V}$")
    box(xf, 0.62, 0.50, 0.075, "$\\mathbf{Z}=\\mathbf{Q}\\mathbf{K}^{\\top}/\\sqrt{d}$")
    box(xf, 0.46, 0.42, 0.075, "$\\mathbf{A}=\\mathrm{softmax}(\\mathbf{Z})$")
    box(xf, 0.30, 0.42, 0.075, "$\\mathbf{O}=\\mathbf{A}\\mathbf{V}$")
    box(xf, 0.14, 0.42, 0.075, "output / logit $F_c$", fc="#eef2f7", bold=True)
    for y0, y1 in [(0.94, 0.78), (0.78, 0.62), (0.62, 0.46), (0.46, 0.30), (0.30, 0.14)]:
        arrow((xf, y0 - 0.038), (xf, y1 + 0.038), FWD)
    # V bypasses Z/softmax straight to O
    arrow((xf + 0.25, 0.78), (xf + 0.34, 0.78), FWD, lw=1.1, style="-")
    arrow((xf + 0.34, 0.78), (xf + 0.34, 0.30), FWD, lw=1.1, style="-")
    arrow((xf + 0.34, 0.30), (xf + 0.21, 0.30), FWD, lw=1.1)
    ax.text(xf + 0.365, 0.54, "$\\mathbf{V}$", fontsize=7, color=FWD)

    # relevance column (right) bottom->top with rule labels
    xr = 0.80
    rules = [
        (0.14, 0.30, "$R_{\\mathbf{O}}$"),
        (0.30, 0.46, "$\\varepsilon$-bilinear split:\n$R_{\\mathbf{A}}+R_{\\mathbf{V}} \\leftarrow R_{\\mathbf{O}}$"),
        (0.46, 0.62, "Taylor-linearized\nsoftmax: $R_{\\mathbf{A}}\\rightarrow R_{\\mathbf{Z}}$"),
        (0.62, 0.78, "$\\varepsilon$-bilinear split:\n$R_{\\mathbf{Q}}+R_{\\mathbf{K}} \\leftarrow R_{\\mathbf{Z}}$"),
        (0.78, 0.94, "$\\varepsilon$-LRP linear rule\n$\\rightarrow R_{\\mathbf{X}}$"),
    ]
    for y0, y1, lab in rules:
        arrow((xr, y0 + 0.02), (xr, y1 - 0.02), REL, lw=1.8)
        ax.text(xr + 0.025, (y0 + y1) / 2, lab, fontsize=6.3, color=REL,
                ha="left", va="center", linespacing=1.25)
    ax.text(xr, 0.10, "relevance $R$", fontsize=7.5, color=REL,
            ha="center", fontweight="bold")
    ax.text(xf, 0.035, "forward pass", fontsize=7.5, color=FWD,
            ha="center", fontweight="bold")
    ax.plot([0.585, 0.585], [0.08, 0.97], color="#dddddd", lw=0.8, ls=":")
    fig.tight_layout(pad=0.2)
    fig.savefig("figures/hilrp/attnlrp_flow.pdf", bbox_inches="tight")
    fig.savefig("figures/hilrp/attnlrp_flow.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/hilrp/attnlrp_flow.pdf")


# ── 3. Pointing-Game rank slopegraph across backbone families ────────────────
def fig_rankflip(get):
    cols = [("resnet50", "RN-50\n(CNN)"),
            ("vit_base_patch16_224", "ViT-B/16\n(isotropic)"),
            ("swin_base_patch4_window7_224", "Swin-B\n(hierarch.)"),
            ("efficientvit_b2", "EffViT-B2\n(linear)")]
    meths = [(k, lab, fam) for k, lab, fam in METHODS
             if k not in ("attention_rollout", "attnlrp")]
    pg = {k: [get(m, k, "pointing_game_mean") for m, _ in cols] for k, _, _ in meths}
    # rank per column (1 = best); ties broken by overall mean so lines stay stable
    overall = {k: np.mean(pg[k]) for k in pg}
    ranks = {k: [] for k in pg}
    for j in range(len(cols)):
        order = sorted(pg, key=lambda k: (-pg[k][j], -overall[k]))
        for r, k in enumerate(order, start=1):
            ranks[k].append(r)

    fig, ax = plt.subplots(figsize=(3.5, 3.6))
    x = np.arange(len(cols))
    for k, lab, fam in meths:
        c = FAMILY_COLORS[fam]
        cam = fam == "CAM"
        ax.plot(x, ranks[k], color=c, lw=2.4 if cam else 1.1,
                alpha=1.0 if cam else 0.55, zorder=3 if cam else 2,
                marker="o", ms=4 if cam else 2.6, mec="white", mew=0.6)
        ax.text(-0.14, ranks[k][0], lab, ha="right", va="center", fontsize=6.4,
                color="#222", fontweight="bold" if cam else "normal")
        ax.text(len(cols) - 0.86, ranks[k][-1], lab, ha="left", va="center",
                fontsize=6.4, color="#222", fontweight="bold" if cam else "normal")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in cols], fontsize=6.8)
    ax.set_xlim(-1.35, len(cols) - 1 + 1.35)
    ax.set_ylim(len(meths) + 0.5, 0.5)
    ax.set_yticks([])
    ax.set_ylabel("Pointing-Game rank (top = best)", fontsize=8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#bbb")
    ax.tick_params(length=0)
    ax.grid(axis="y", color="#eee", lw=0.5, zorder=0)
    fig.tight_layout(pad=0.3)
    fig.savefig("figures/bench/bench_rankflip.pdf", bbox_inches="tight")
    fig.savefig("figures/bench/bench_rankflip.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/bench/bench_rankflip.pdf")


# ── 4. Faithfulness Correlation vs the per-image noise band ─────────────────
def fig_fc_noise(get):
    meths = [(k, lab, fam) for k, lab, fam in METHODS if k != "attnlrp"]
    stds = []
    for mod, _ in MODELS:
        for k, _, _ in meths:
            s = get(mod, k, "faithfulness_correlation_std")
            if s is not None:
                stds.append(s)
    band = float(np.median(stds))

    fig, ax = plt.subplots(figsize=(3.5, 3.6))
    ax.axvspan(-band, band, color="#ececec", zorder=0)
    ax.axvline(0, color="#999", lw=0.8, zorder=1)
    ys, labels = [], []
    for i, (k, lab, fam) in enumerate(meths):
        y = len(meths) - i
        ys.append(y)
        labels.append(lab)
        for mod, _ in MODELS:
            v = get(mod, k, "faithfulness_correlation_mean")
            if v is not None:
                ax.scatter(v, y, s=14, color=FAMILY_COLORS[fam], alpha=0.85,
                           edgecolor="white", lw=0.4, zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlim(-0.30, 0.30)
    ax.set_ylim(0.3, len(meths) + 0.7)
    ax.tick_params(axis="x", labelsize=7)
    ax.set_xlabel("Faithfulness Correlation (mean per backbone)", fontsize=8)
    ax.text(band + 0.008, len(meths) + 0.45,
            f"$\\pm${band:.2f} median per-image std", fontsize=6.3,
            color="#777", ha="left", va="center")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#bbb")
    ax.tick_params(length=0)
    fig.tight_layout(pad=0.3)
    fig.savefig("figures/bench/bench_fc_noise.pdf", bbox_inches="tight")
    fig.savefig("figures/bench/bench_fc_noise.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/bench/bench_fc_noise.pdf  (band=%.3f)" % band)


# ── 5. Benchmark pipeline overview ───────────────────────────────────────────
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.1, 1.95))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    stages = [
        ("Dataset", "ImageNet-S val.\nfixed 1000 images\n+ GT masks", "#eef2f7"),
        ("Backbones (8)", "CNN  |  isotropic ViT\nSwin  |  PVT  |  MaxViT\nMobileViT  |  EffViT", "#eef2f7"),
        ("Methods (13)", "Gradient (6)\nCAM (2)  |  Attention (2)\nPerturbation (3)", "#eef2f7"),
        ("Normalization", "predicted-class target\n$|\\cdot|$ + min-max\nto $[0,1]$", "#eef2f7"),
        ("Quantus metrics", "FC  FE  (faithfulness)\nPG (localization)\nMS (robust.)  SP (compl.)", "#eef2f7"),
        ("Outputs", "metric tables\nfigures\nliving leaderboard", "#e8f2ec"),
    ]
    n = len(stages)
    w, gap = 0.148, (1 - 6 * 0.148) / (n - 1) - 0.0001
    fam_cols = list(FAMILY_COLORS.values())
    for i, (title, body, fc) in enumerate(stages):
        x0 = i * (w + gap)
        ax.add_patch(FancyBboxPatch((x0, 0.16), w, 0.66,
                                    boxstyle="round,pad=0.004,rounding_size=0.015",
                                    facecolor=fc, edgecolor="#44546a", lw=1.0))
        ax.text(x0 + w / 2, 0.90, title, ha="center", va="center",
                fontsize=8, fontweight="bold", color="#1a2433")
        ax.text(x0 + w / 2, 0.49, body, ha="center", va="center",
                fontsize=6.2, color="#333", linespacing=1.45)
        if i == 2:  # method-family color key under the methods box
            for fi, c in enumerate(fam_cols):
                ax.add_patch(plt.Rectangle((x0 + 0.012 + fi * 0.033, 0.055),
                                           0.024, 0.05, facecolor=c, edgecolor="none"))
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((x0 + w + 0.004, 0.49),
                                         (x0 + w + gap - 0.004, 0.49),
                                         arrowstyle="-|>", color="#44546a",
                                         lw=1.2, mutation_scale=10))
    fig.tight_layout(pad=0.2)
    fig.savefig("figures/bench/bench_pipeline.pdf", bbox_inches="tight")
    fig.savefig("figures/bench/bench_pipeline.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/bench/bench_pipeline.pdf")


if __name__ == "__main__":
    get = load()
    fig_taxonomy()
    fig_attnlrp_flow()
    fig_rankflip(get)
    fig_fc_noise(get)
    fig_pipeline()
