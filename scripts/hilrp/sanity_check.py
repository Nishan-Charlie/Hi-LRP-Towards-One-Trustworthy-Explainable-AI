"""Sanity check: model parameter randomization (Adebayo et al., NeurIPS 2018).

An explanation that does not change when the model's weights are destroyed is
not explaining the model. Edge detectors pass every localization test and fail
this one, which is the point of the check.

Two protocols, both from the original paper:

  independent  randomize exactly one block, restore it, move to the next
  cascading    randomize blocks cumulatively from the output backwards

For each stage we recompute the attribution for the SAME target class as the
original (the class the intact model predicted) and measure how much the map
changed. Low similarity = the attribution depends on the weights = the check
passes. Reported as Spearman rank correlation and SSIM against the intact map,
both on |R| so that the comparison is about structure, not sign.

Protocol note: as in the other scripts, every vanilla baseline runs BEFORE any
HiLRP call, because HiLRP's patches are class-level and process-global.

Run:  <mri-diffuser python> scripts/hilrp/sanity_check.py [n_images] [model ...]
Out:  results/sanity_check/{per_image.csv, summary.csv, sanity_check.png}

Figure-only paths (the curve is read from the cached summary.csv, so neither
touches the reported numbers):
  --figure-only   recompute just the displayed model's panel maps, then plot
  --replot        plot from results/figure_cache/sanity_check.npz, no model run
"""
import copy
import os
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))                       # <repo>/scripts

from figstyle import lxt_transformers_shim
lxt_transformers_shim()          # lxt.efficient vs transformers 5.x

import timm
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity

OUT = os.path.join("results", "sanity_check")
os.makedirs(OUT, exist_ok=True)

GAMMA = 0.25
CONV_GAMMA = {"efficientvit": 0.05, "mobilevit": 0.1}
DEFAULT_MODELS = ["swin_base_patch4_window7_224", "pvt_v2_b2"]
SEED = 0

CACHE = os.path.join("data", "ImageNetS", "cache_validation_1000.pt")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_cache(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def adapt_input(img, model):
    mean = torch.tensor(model.default_cfg["mean"]).view(3, 1, 1)
    std = torch.tensor(model.default_cfg["std"]).view(3, 1, 1)
    raw = (img * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)
    return (raw - mean) / std


def get_attributor(model_name):
    if model_name.startswith("swin"):
        from xai_bench.methods.hilrp.swin_lxt import attribute_swin
        return attribute_swin
    if model_name.startswith("pvt"):
        from xai_bench.methods.hilrp.pvt_lxt import attribute_pvt
        return attribute_pvt
    if model_name.startswith("efficientvit"):
        from xai_bench.methods.hilrp.efficientvit_lxt import attribute_efficientvit
        return attribute_efficientvit
    raise ValueError(f"no HiLRP attributor for {model_name}")


def conv_gamma_for(model_name):
    for prefix, g in CONV_GAMMA.items():
        if model_name.startswith(prefix):
            return g
    return None


def randomization_blocks(model):
    """Ordered output-to-input list of (name, [modules]) to randomize.

    Swin and PVT both expose a stem, a list of stages, and a head, so the
    grouping is derived from the module tree rather than hard-coded per model.
    """
    blocks = []
    if hasattr(model, "head") and model.head is not None:
        blocks.append(("head", [model.head]))
    stages = None
    for attr in ("layers", "stages", "blocks"):
        if hasattr(model, attr):
            cand = getattr(model, attr)
            if isinstance(cand, (nn.ModuleList, nn.Sequential)) and len(cand) > 1:
                stages = cand
                break
    if stages is not None:
        for i in reversed(range(len(stages))):
            blocks.append((f"stage{i}", [stages[i]]))
    if hasattr(model, "patch_embed"):
        blocks.append(("patch_embed", [model.patch_embed]))
    return blocks


def randomize_(modules, generator):
    """Re-initialize every parameter in `modules` from N(0, std) matched to the
    parameter's own current standard deviation, so scale is preserved and the
    change is in the learned structure rather than the magnitude."""
    for m in modules:
        for p in m.parameters(recurse=True):
            std = float(p.detach().std()) if p.numel() > 1 else 1.0
            if not np.isfinite(std) or std == 0.0:
                std = 0.02
            with torch.no_grad():
                p.copy_(torch.randn(p.shape, generator=generator,
                                    device=generator.device, dtype=p.dtype) * std)


def similarity(a, b):
    """Agreement between two attribution maps.

    Returns (spearman on |R|, SSIM on |R|, spearman on signed R). The standard
    sanity check compares magnitude maps, but for a signed attribution the class
    identity lives in the SIGN (Sec. class-sensitivity), so a magnitude-only
    comparison discards exactly the part the classifier head controls. We report
    both and let the difference speak.
    """
    a_s = np.asarray(a, dtype=np.float64)
    b_s = np.asarray(b, dtype=np.float64)
    a_m, b_m = np.abs(a_s), np.abs(b_s)
    if a_m.std() < 1e-12 or b_m.std() < 1e-12:
        return float("nan"), float("nan"), float("nan")
    rho = float(spearmanr(a_m.flatten(), b_m.flatten()).correlation)
    rng = max(a_m.max(), b_m.max()) - min(a_m.min(), b_m.min())
    ssim = float(structural_similarity(a_m, b_m, data_range=rng)) if rng > 0 else float("nan")
    rho_signed = float(spearmanr(a_s.flatten(), b_s.flatten()).correlation)
    return rho, ssim, rho_signed


def run_model(model_name, data, n, mode):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(model_name, pretrained=True).to(device).eval()
    attribute = get_attributor(model_name)
    cg = conv_gamma_for(model_name)
    kw = dict(gamma=GAMMA)
    if cg is not None:
        kw["conv_gamma"] = cg

    pristine = copy.deepcopy(model.state_dict())
    blocks = randomization_blocks(model)
    print(f"  blocks (output to input): {[b[0] for b in blocks]}")

    rows = []
    demo = {}
    for idx in range(n):
        x = adapt_input(data[idx]["image"], model).unsqueeze(0).to(device)
        model.load_state_dict(pristine)
        with torch.no_grad():
            target = int(model(x)[0].argmax())
        ref = attribute(model, x, target=target, **kw)["pixel_map"].numpy()
        if idx == 0:
            demo["original"] = ref

        gen = torch.Generator(device=device).manual_seed(SEED + idx)
        if mode == "independent":
            for name, mods in blocks:
                model.load_state_dict(pristine)
                randomize_(mods, gen)
                r = attribute(model, x, target=target, **kw)["pixel_map"].numpy()
                rho, ssim, rho_s = similarity(ref, r)
                rows.append(dict(model=model_name, mode=mode, idx=idx, block=name,
                                 spearman=rho, ssim=ssim, spearman_signed=rho_s))
                if idx == 0:
                    demo[name] = r
        else:  # cascading
            model.load_state_dict(pristine)
            for name, mods in blocks:
                randomize_(mods, gen)          # cumulative: never restored
                r = attribute(model, x, target=target, **kw)["pixel_map"].numpy()
                rho, ssim, rho_s = similarity(ref, r)
                rows.append(dict(model=model_name, mode=mode, idx=idx, block=name,
                                 spearman=rho, ssim=ssim, spearman_signed=rho_s))
                if idx == 0:
                    demo[name] = r

        if (idx + 1) % 5 == 0:
            print(f"    {idx + 1} images")

    model.load_state_dict(pristine)
    del model
    torch.cuda.empty_cache()
    return rows, demo, [b[0] for b in blocks]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    models = sys.argv[2:] or DEFAULT_MODELS
    data = load_cache(CACHE)
    n = min(n, len(data))

    all_rows, demos, orders = [], {}, {}
    for m in models:
        for mode in ("independent", "cascading"):
            print(f"\n=== {m} / {mode} ({n} images) ===")
            rows, demo, order = run_model(m, data, n, mode)
            all_rows += rows
            demos[(m, mode)] = demo
            orders[m] = order

    fields = ["model", "mode", "idx", "block", "spearman", "ssim", "spearman_signed"]
    with open(os.path.join(OUT, "per_image.csv"), "w") as f:
        f.write(",".join(fields) + "\n")
        for r in all_rows:
            f.write(",".join(str(r[k]) for k in fields) + "\n")

    print("\n" + "=" * 74)
    print(f"{'model':<30s} {'mode':<12s} {'block':<12s} {'rho|R|':>9s} {'SSIM':>7s} {'rho signed':>11s}")
    print("=" * 74)
    summary = []
    for m in models:
        for mode in ("independent", "cascading"):
            for block in orders[m]:
                sel = [r for r in all_rows if r["model"] == m and r["mode"] == mode
                       and r["block"] == block]
                if not sel:
                    continue
                rho = np.nanmean([r["spearman"] for r in sel])
                ss = np.nanmean([r["ssim"] for r in sel])
                rs = np.nanmean([r["spearman_signed"] for r in sel])
                print(f"{m:<30s} {mode:<12s} {block:<12s} {rho:>+9.3f} {ss:>7.3f} {rs:>+11.3f}")
                summary.append(dict(model=m, mode=mode, block=block, n=len(sel),
                                    spearman=rho, ssim=ss, spearman_signed=rs))
    print("=" * 74)

    with open(os.path.join(OUT, "summary.csv"), "w") as f:
        f.write("model,mode,block,n,spearman,ssim,spearman_signed\n")
        for r in summary:
            f.write(f"{r['model']},{r['mode']},{r['block']},{r['n']},"
                    f"{r['spearman']:.4f},{r['ssim']:.4f},"
                    f"{r['spearman_signed']:.4f}\n")

    make_figure(demos, orders, summary, models, data)
    print(f"\nwrote {OUT}/per_image.csv, summary.csv, sanity_check.png")


def make_figure(demos, orders, summary, models, data):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))                   # <repo>/scripts
    from figstyle import (apply_paper_style, save, block_label, model_label,
                          mode_label, FULL_W, FS_TITLE, FS_LABEL)
    apply_paper_style()
    import matplotlib.pyplot as plt
    from xai_bench.methods.hilrp.viz import normalize_for_display

    m = models[0]
    order = orders[m]
    n_col = len(order) + 2                      # input + original + blocks
    # 8 columns cannot survive a 3.5 in column, so this is a figure* authored at
    # \textwidth: the panel titles print at the 8 pt they are written at.
    panel = FULL_W / n_col
    fig = plt.figure(figsize=(FULL_W, 2 * panel + 1.85))
    gs = fig.add_gridspec(3, n_col, height_ratios=[1, 1, 1.9], hspace=0.30,
                          wspace=0.06)

    raw = (data[0]["image"] * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1).permute(1, 2, 0).numpy()

    for row, mode in enumerate(("independent", "cascading")):
        demo = demos[(m, mode)]
        ax = fig.add_subplot(gs[row, 0])
        ax.imshow(raw); ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(mode_label(mode), fontsize=FS_LABEL)
        if row == 0:
            ax.set_title("Input", fontsize=FS_TITLE)
        for j, name in enumerate(["original"] + order):
            if name not in demo:
                continue
            ax = fig.add_subplot(gs[row, j + 1])
            v, vmax = normalize_for_display(demo[name])
            ax.imshow(v, cmap="bwr", vmin=-vmax, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(block_label(name), fontsize=FS_TITLE)

    ax = fig.add_subplot(gs[2, :])
    xs = np.arange(len(order))
    for m2 in models:
        for mode, ls in (("independent", "--"), ("cascading", "-")):
            ys = [next((r["spearman"] for r in summary
                        if r["model"] == m2 and r["mode"] == mode and r["block"] == b),
                       np.nan) for b in orders[m2]]
            ax.plot(xs[:len(ys)], ys, ls, marker="o", ms=3, lw=1.1,
                    label=f"{model_label(m2)} / {mode_label(mode)}")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels([block_label(b).replace("\n", " ") for b in order])
    ax.set_ylabel("Spearman vs.\nIntact Map")
    ax.set_xlabel("Randomized Block (Output Side on the Left)")
    ax.set_ylim(-0.10, 1.10)
    ax.legend(ncol=2, frameon=False, handlelength=1.8, columnspacing=1.2)
    ax.grid(alpha=0.3, lw=0.5)

    fig.suptitle("Parameter-Randomization Sanity Check: HiLRP Maps Degrade as the "
                 "Model's Weights Are Destroyed", y=0.99)
    fig.savefig(os.path.join(OUT, "sanity_check.png"), dpi=400, bbox_inches="tight")
    save(fig, "figures/hilrp/sanity_check", also_pdf=False)


# ----------------------------------------------------------- figure-only paths
DEMO_CACHE = os.path.join("results", "figure_cache", "sanity_check.npz")


def _save_demo_cache(demos, orders):
    """Persist the single-image demo maps so restyling never re-runs the model.

    Plain float arrays plus one comma-joined string per model, so the cache
    loads without allow_pickle.
    """
    store = {}
    for (mdl, mode), demo in demos.items():
        for name, arr in demo.items():
            store[f"map~{mdl}~{mode}~{name}"] = np.asarray(arr)
    for mdl, order in orders.items():
        store[f"order~{mdl}"] = np.array(",".join(order))
    os.makedirs(os.path.dirname(DEMO_CACHE), exist_ok=True)
    np.savez_compressed(DEMO_CACHE, **store)
    print(f"cached demo maps -> {DEMO_CACHE}")


def _load_demo_cache():
    z = np.load(DEMO_CACHE)
    demos, orders = {}, {}
    for k in z.files:
        if k.startswith("order~"):
            orders[k.split("~", 1)[1]] = str(z[k]).split(",")
        else:
            _, mdl, mode, name = k.split("~")
            demos.setdefault((mdl, mode), {})[name] = z[k]
    return demos, orders


def _load_summary():
    """Read the cached per-block curve (n=50) instead of recomputing it."""
    import csv
    with open(os.path.join(OUT, "summary.csv")) as f:
        return [dict(model=r["model"], mode=r["mode"], block=r["block"],
                     n=int(r["n"]), spearman=float(r["spearman"]),
                     ssim=float(r["ssim"]),
                     spearman_signed=float(r["spearman_signed"]))
                for r in csv.DictReader(f)]


def _orders_from_summary(summary):
    """Per-model block order, output side first, as written by the full run."""
    orders = {}
    for r in summary:
        seq = orders.setdefault(r["model"], [])
        if r["block"] not in seq:
            seq.append(r["block"])
    return orders


def demo_only(models):
    """Recompute ONLY the image-panel maps for the displayed model.

    The curve underneath comes from the existing summary.csv and the panels use
    the first cached image, so this costs 2 x (n_blocks + 1) attributions
    instead of a full n-image sweep over every model.
    """
    data = load_cache(CACHE)
    demos, orders = {}, {}
    mdl = models[0]
    for mode in ("independent", "cascading"):
        print(f"=== {mdl} / {mode} (demo image only) ===")
        _, demo, order = run_model(mdl, data, 1, mode)
        demos[(mdl, mode)] = demo
        orders[mdl] = order
    _save_demo_cache(demos, orders)
    return demos, data


def _plot_from_parts(demos, models):
    summary = _load_summary()
    orders = _orders_from_summary(summary)       # every model's curve, from cache
    orders.setdefault(models[0], list(next(iter(demos.values())).keys()))
    make_figure(demos, orders, summary, models, load_cache(CACHE))


if __name__ == "__main__":
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    mdls = argv[1:] or DEFAULT_MODELS
    if "--replot" in flags and os.path.exists(DEMO_CACHE):
        _plot_from_parts(_load_demo_cache()[0], mdls)
    elif "--figure-only" in flags:
        _plot_from_parts(demo_only(mdls)[0], mdls)
    else:
        main()
