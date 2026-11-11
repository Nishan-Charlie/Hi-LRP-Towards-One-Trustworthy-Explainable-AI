"""Breaking-point figures for the benchmark paper, built from all_results_sofar.csv
(the single source of truth), including PVT-v2 and EfficientViT-B2.

Produces (figures/):
  bench_localization.pdf  Pointing-Game heatmap, diverging around the 0.61 random-point prior
  bench_robustness.pdf    Max-Sensitivity heatmap, log color (explosions = dark)
  bench_pareto.pdf        cost (log ms) vs mean Pointing, colored by method family

Colorblind-safe palettes: RdBu (diverging), YlOrRd+LogNorm (sequential), Okabe-Ito (categorical).
AttnLRP rows come from the dedicated attnlrp_grid_hier / attnlrp_grid_vit runs (n=100,
appended to the CSV); cost per method is from tab:cost. \\nd cells are drawn as hatched gray.
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LogNorm

os.makedirs("figures", exist_ok=True)
PRIOR = 0.61  # random-point Pointing prior on this eval set

# model CSV key -> (short label, family)
MODELS = [
    ("resnet50", "RN-50", "CNN"),
    ("vit_base_patch16_224", "ViT-B/16", "Isotropic"),
    ("swin_base_patch4_window7_224", "Swin-B", "Hierarch."),
    ("pvt_v2_b2", "PVT-v2", "Hierarch."),
    ("maxvit_small_tf_224", "MaxViT-S", "Multi-axis"),
    ("mobilevitv2_100", "MobileViT", "Hybrid"),
    ("efficientvit_b1", "EffViT-B1", "Linear"),
    ("efficientvit_b2", "EffViT-B2", "Linear"),
]
# method CSV key -> (label, family); order groups families top->bottom
METHODS = [
    ("saliency", "Saliency", "Gradient"),
    ("integrated_gradients", "Integrated Grad", "Gradient"),
    ("input_x_gradient", "Input$\\times$Grad", "Gradient"),
    ("smoothgrad", "SmoothGrad", "Gradient"),
    ("vargrad", "VarGrad", "Gradient"),
    ("gradient_shap", "GradientSHAP", "Gradient"),
    ("grad_cam", "Grad-CAM", "CAM"),
    ("grad_cam_plus_plus", "Grad-CAM++", "CAM"),
    ("attention_rollout", "Attention Rollout", "Attention"),
    ("attnlrp", "AttnLRP", "Attention"),
    ("occlusion", "Occlusion", "Perturb."),
    ("rise", "RISE", "Perturb."),
    ("lime", "LIME", "Perturb."),
]
FAMILY_COLORS = {"Gradient": "#0072B2", "CAM": "#E69F00",
                 "Attention": "#009E73", "Perturb.": "#D55E00"}
COST_MS = {  # tab:cost, mean over backbones
    "saliency": 31.9, "integrated_gradients": 479.6, "input_x_gradient": 22.1,
    "smoothgrad": 20.9, "vargrad": 40.4, "gradient_shap": 32.7,
    "grad_cam": 26.5, "grad_cam_plus_plus": 24.0,
    "attention_rollout": 41.0, "attnlrp": 57.4,
    "occlusion": 11011.1, "rise": 669.9, "lime": 1488.1,
}
# NEUCOM revision: synchronized re-timing on the eight backbones (cost_table.py)
_COST_CSV = os.path.join("results", "revision_neucom", "analysis", "cost_table.csv")
if os.path.exists(_COST_CSV):
    import csv as _csv
    with open(_COST_CSV) as _f:
        COST_MS.update({r["method"]: float(r["time_ms"]) for r in _csv.DictReader(_f)
                        if r["method"] in COST_MS})
# NEUCOM revision: GradientSHAP cells with no stored result (MaxViT-S, MobileViT-v2,
# EffViT-B1/B2) come from configs/revision_gradshap_fill.yaml; every other cell is the
# stored result. (The former hard-coded GradientSHAP row matched no stored run.)
GRADSHAP_FILL_SUMMARY = os.path.join("results", "revision_gradshap_fill", "summary.csv")

RESULTS_CSV = os.path.join("benchmark_results", "all_results_sofar.csv")

# Attention-native methods are defined only on the isotropic ViT-B/16 (Section 5.3).
# The CSV also holds AttnLRP rows from the naive generic-LXT fallback run
# (attnlrp_grid_hier) on the six non-isotropic backbones; those are NOT AttnLRP and
# must never be drawn (NEUCOM revision, R1.1/R3.3).
ATTENTION_NATIVE = {"attention_rollout", "attnlrp"}
ISOTROPIC = {"vit_base_patch16_224"}

# NEUCOM revision: the ViT-B/16 row of the grid is re-run with the model's native
# input statistics (configs/revision_vit_native.yaml). When present, it replaces
# the June full_benchmark_100 row for every method except AttnLRP, whose
# attnlrp_grid_vit run already used the native statistics.
VIT_NATIVE_SUMMARY = os.path.join("results", "revision_vit_native", "summary.csv")


def load_rows():
    """{model: {method: summary row}} for the five-metric grid, with the revision runs
    (native-statistics ViT-B/16 row, GradientSHAP fill) overlaid on the stored results."""
    rows = [r for r in csv.DictReader(open(RESULTS_CSV)) if r["model"] and r["method"]]
    d = {}
    for r in rows:
        # mobilevitv2 is duplicated: prefer the corrected rerun over the
        # preprocessing-artifact run (vit_suite_10, top-1~0 -> Grad-CAM 0.49).
        prev = d.get(r["model"], {}).get(r["method"])
        if prev is not None and "rerun" in prev["source_run"] and "rerun" not in r["source_run"]:
            continue
        d.setdefault(r["model"], {})[r["method"]] = r
    if os.path.exists(VIT_NATIVE_SUMMARY):
        for r in csv.DictReader(open(VIT_NATIVE_SUMMARY)):
            if r["method"] != "attnlrp":
                d.setdefault(r["model"], {})[r["method"]] = dict(r, source_run="revision_vit_native")
    if os.path.exists(GRADSHAP_FILL_SUMMARY):
        for r in csv.DictReader(open(GRADSHAP_FILL_SUMMARY)):
            d.setdefault(r["model"], {})[r["method"]] = dict(r, source_run="revision_gradshap_fill")
    return d


def load(stat="mean"):
    d = load_rows()

    def get(model, method, metric):
        if method in ATTENTION_NATIVE and model not in ISOTROPIC:
            return None                       # structurally undefined: hatched N/A
        r = d.get(model, {}).get(method)
        if not r:
            return None
        v = r.get(f"{metric}_{stat}", "")
        return float(v) if v not in ("", "nan", None) else None
    return get


def grid(get, metric):
    M = np.full((len(METHODS), len(MODELS)), np.nan)
    for i, (mk, _, _) in enumerate(METHODS):
        for j, (mod, _, _) in enumerate(MODELS):
            v = get(mod, mk, metric)
            if v is not None:
                M[i, j] = v
    return M


def _heat(ax, M, cmap, norm, fmt, title, cbar_label):
    ax.set_facecolor("white")
    im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor="#e8e8e8",
                                           hatch="///", edgecolor="white", lw=0))
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=9, color="#888")
            else:
                r, g, b, _ = im.cmap(im.norm(M[i, j]))
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                ax.text(j, i, fmt(M[i, j]), ha="center", va="center", fontsize=10,
                        color="white" if lum < 0.5 else "#222")
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([m[1] for m in MODELS], rotation=35, ha="right", fontsize=12)
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels([m[1] for m in METHODS], fontsize=12)
    for i, (_, _, fam) in enumerate(METHODS):  # family tick color
        ax.get_yticklabels()[i].set_color(FAMILY_COLORS[fam])
    ax.set_xticks(np.arange(-.5, len(MODELS), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(METHODS), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.5)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=13, pad=8)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(cbar_label, fontsize=12)
    cb.ax.tick_params(labelsize=11)


def fig_localization(get):
    M = grid(get, "pointing_game")
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    norm = TwoSlopeNorm(vmin=0.4, vcenter=PRIOR, vmax=1.0)
    _heat(ax, M, "RdBu", norm, lambda v: f"{v:.2f}",
          "Pointing Game across architectures", f"PG (blue $>$ {PRIOR} prior $>$ red)")
    fig.tight_layout()
    fig.savefig("figures/bench/bench_localization.pdf", bbox_inches="tight")
    fig.savefig("figures/bench/bench_localization.png", dpi=150, bbox_inches="tight")
    print("wrote figures/bench/bench_localization.pdf")


def fig_robustness(get):
    M = grid(get, "max_sensitivity")
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    norm = LogNorm(vmin=0.05, vmax=100)
    _heat(ax, M, "YlOrRd", norm, lambda v: f"{v:.1f}" if v >= 1 else f"{v:.2f}",
          "Max-Sensitivity across architectures (lower is better)", "Max-Sensitivity (log)")
    fig.tight_layout()
    fig.savefig("figures/bench/bench_robustness.pdf", bbox_inches="tight")
    fig.savefig("figures/bench/bench_robustness.png", dpi=150, bbox_inches="tight")
    print("wrote figures/bench/bench_robustness.pdf")


def fig_pareto(get):
    PG = grid(get, "pointing_game")
    mean_pg = np.nanmean(PG, axis=1)
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    # per-method label position to avoid collisions: (dx_points, dy_points, va, ha)
    L, R = (-7, 0, "center", "right"), (7, 0, "center", "left")
    LBL = {
        "smoothgrad": L, "grad_cam": L, "vargrad": L, "attnlrp": R,
        "gradient_shap": (0, -8, "top", "center"), "grad_cam_plus_plus": L, "saliency": R,
        "input_x_gradient": L, "attention_rollout": R,
        "integrated_gradients": (0, 8, "bottom", "center"), "rise": (0, 8, "bottom", "center"),
        "lime": (0, 8, "bottom", "center"), "occlusion": (0, 8, "bottom", "center"),
    }
    for i, (mk, lab, fam) in enumerate(METHODS):
        x, y = COST_MS[mk], mean_pg[i]
        # attention-native methods are defined on ViT-B/16 only: hollow marker, and the
        # value is that single backbone's PG, not an eight-backbone mean
        hollow = mk in ATTENTION_NATIVE
        ax.scatter(x, y, s=70, facecolor="white" if hollow else FAMILY_COLORS[fam],
                   edgecolor=FAMILY_COLORS[fam] if hollow else "white", lw=1.6 if hollow else 1.2, zorder=3)
        dx, dy, va, ha = LBL.get(mk, (0, 8, "bottom", "center"))
        ax.annotate(lab + ("$^{\\dagger}$" if hollow else ""), (x, y), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, va=va, fontsize=10.5, color="#222")
    ax.axhline(PRIOR, ls="--", lw=1, color="#999", zorder=1)
    ax.text(20000, PRIOR + 0.004, "Random prior", fontsize=10, color="#888", ha="right")
    ax.set_xscale("log")
    ax.set_xlim(5, 25000)
    ax.set_ylim(min(0.6, np.nanmin(mean_pg) - 0.03), max(0.91, np.nanmax(mean_pg) + 0.03))
    ax.tick_params(labelsize=11)
    ax.set_xlabel("Cost per explanation (ms, log scale)", fontsize=12)
    ax.set_ylabel("Mean Pointing Game\n(over applicable backbones)", fontsize=12)
    ax.set_title("Cost vs. localization: no fidelity gain from expensive methods", fontsize=13)
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=c, label=f, markersize=9,
                          markeredgecolor="white") for f, c in FAMILY_COLORS.items()]
    ax.legend(handles=handles, fontsize=11, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=4)
    ax.grid(True, which="both", axis="x", ls=":", lw=.5, color="#ddd")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig("figures/bench/bench_pareto.pdf", bbox_inches="tight")
    fig.savefig("figures/bench/bench_pareto.png", dpi=150, bbox_inches="tight")
    print("wrote figures/bench/bench_pareto.pdf")


if __name__ == "__main__":
    get = load()
    fig_localization(get)
    fig_robustness(get)
    fig_pareto(get)
