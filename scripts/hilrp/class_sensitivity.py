"""Class sensitivity and misclassified-sample attribution.

Two questions a localization score cannot answer:

  (1) Is the map class-specific, or does it just find the salient object?
      Probe: attribute the predicted class and then the model's LEAST likely
      class (bottom-1 logit) on the SAME image. A class-specific signed
      attribution should put positive evidence on the object for the predicted
      class and NEGATIVE evidence there for the counterfactual class. Note that
      top-1 vs top-2 is the wrong probe on single-object images: both classes are
      supported by the same object, so high agreement is expected and means
      nothing.

  (2) When the model is wrong, does the explanation follow the MODEL or the
      LABEL? Probe: on misclassified images, attribute both the predicted class
      and the ground-truth class and compare.

Protocol note (two-pass): HiLRP's patches are class-level and process-global
(nn.LayerNorm, nn.GELU, and the per-family attention classes). Every vanilla
baseline is therefore computed for EVERY model first, before any HiLRP call
patches anything. Running the baselines inside the patched context measures the
patched gradient, not the baseline.

Run:  <mri-diffuser python> scripts/hilrp/class_sensitivity.py [n_images] [model ...]
Out:  results/class_sensitivity/{summary.csv, per_image.csv, class_sensitivity.png}
"""
import os
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))                       # <repo>/scripts

from figstyle import lxt_transformers_shim
lxt_transformers_shim()          # lxt.efficient vs transformers 5.x

import timm

OUT = os.path.join("results", "class_sensitivity")
os.makedirs(OUT, exist_ok=True)

GAMMA = 0.25
CONV_GAMMA = {"efficientvit": 0.05, "mobilevit": 0.1}
DEFAULT_MODELS = ["swin_base_patch4_window7_224", "pvt_v2_b2", "efficientvit_b2"]

CACHE = os.path.join("data", "ImageNetS", "cache_validation_1000.pt")
MASKS = os.path.join("data", "ImageNetS", "cache_masks_1000.pt")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_cache(path):
    """Local caches hold only tensors and plain containers, so the safe loader
    is enough; fall back only if an older cache carries a custom class."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def adapt_input(img, model):
    """Cached tensors are ImageNet-normalized; convert to the model's own stats."""
    mean = torch.tensor(model.default_cfg["mean"]).view(3, 1, 1)
    std = torch.tensor(model.default_cfg["std"]).view(3, 1, 1)
    raw = (img * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)
    return (raw - mean) / std


def get_attributor(model_name):
    if model_name.startswith("swin"):
        from hilrp.swin_lxt import attribute_swin
        return attribute_swin
    if model_name.startswith("pvt"):
        from hilrp.pvt_lxt import attribute_pvt
        return attribute_pvt
    if model_name.startswith("efficientvit"):
        from hilrp.efficientvit_lxt import attribute_efficientvit
        return attribute_efficientvit
    if model_name.startswith("mobilevit"):
        from hilrp.mobilevit_lxt import attribute_mobilevit
        return attribute_mobilevit
    raise ValueError(f"no HiLRP attributor for {model_name}")


def conv_gamma_for(model_name):
    for prefix, g in CONV_GAMMA.items():
        if model_name.startswith(prefix):
            return g
    return None


def load_masks():
    """idx -> boolean [H, W] object mask (any labeled pixel)."""
    raw = load_cache(MASKS)
    out = {}
    for idx, rec in raw.items():
        m = rec["mask"]
        m = m.numpy() if hasattr(m, "numpy") else np.asarray(m)
        if m.ndim == 3:
            m = m.sum(-1)
        out[int(idx)] = m > 0
    return out


def corr(a, b):
    a, b = a.flatten(), b.flatten()
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def in_mask_mean(rel, mask):
    """Mean relevance inside the object, normalized by the map's absolute scale
    so values are comparable across images and methods."""
    scale = np.abs(rel).mean()
    if scale < 1e-12:
        return float("nan")
    return float(rel[mask].mean() / scale)


# ---------------------------------------------------------------- vanilla pass
def gradcam_pass(model_name, data, masks, n):
    """Grad-CAM class sensitivity. MUST run before any HiLRP patching."""
    from xai_bench.models.timm_models import create_timm_model
    from xai_bench.registry import METHODS
    import xai_bench.methods  # noqa: F401  (populates the registry)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    wrapper = create_timm_model(model_name)
    net = wrapper.model.to(device).eval()
    cam = METHODS.get("grad_cam")(model_wrapper=wrapper, model=net)

    rows = []
    for idx in range(n):
        if idx not in masks:
            continue
        x = data[idx]["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            logits = net(x)[0]
        pred = int(logits.argmax())
        bottom = int(logits.argmin())

        m_pred = cam(x, pred).squeeze().detach().cpu().numpy()
        m_cf = cam(x, bottom).squeeze().detach().cpu().numpy()
        rows.append(dict(
            model=model_name, method="grad_cam", idx=idx,
            label=int(data[idx]["label"]), pred=pred, counter=bottom,
            map_corr=corr(m_pred, m_cf),
            inmask_pred=in_mask_mean(m_pred, masks[idx]),
            inmask_counter=in_mask_mean(m_cf, masks[idx]),
        ))
    del cam, net, wrapper
    torch.cuda.empty_cache()
    return rows


# ---------------------------------------------------------------- HiLRP pass
def hilrp_pass(model_name, data, masks, n):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(model_name, pretrained=True).to(device).eval()
    attribute = get_attributor(model_name)
    cg = conv_gamma_for(model_name)
    kw = dict(gamma=GAMMA)
    if cg is not None:
        kw["conv_gamma"] = cg

    rows, demo = [], []
    for idx in range(n):
        if idx not in masks:
            continue
        x = adapt_input(data[idx]["image"], model).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)[0]
        pred = int(logits.argmax())
        bottom = int(logits.argmin())
        label = int(data[idx]["label"])

        r_pred = attribute(model, x, target=pred, **kw)["pixel_map"].numpy()
        r_cf = attribute(model, x, target=bottom, **kw)["pixel_map"].numpy()

        row = dict(
            model=model_name, method="hilrp", idx=idx,
            label=label, pred=pred, counter=bottom,
            map_corr=corr(r_pred, r_cf),
            inmask_pred=in_mask_mean(r_pred, masks[idx]),
            inmask_counter=in_mask_mean(r_cf, masks[idx]),
            inmask_gt=float("nan"), gt_corr=float("nan"),
        )

        # misclassified: explain the GT class too, and compare
        if pred != label:
            r_gt = attribute(model, x, target=label, **kw)["pixel_map"].numpy()
            row["inmask_gt"] = in_mask_mean(r_gt, masks[idx])
            row["gt_corr"] = corr(r_pred, r_gt)

        rows.append(row)
        # keep demo images on distinct classes, otherwise the figure repeats
        if len(demo) < 4 and label not in {d[6] for d in demo}:
            demo.append((idx, data[idx]["image"], r_pred, r_cf, pred, bottom, label))
        if (idx + 1) % 25 == 0:
            print(f"    {idx + 1} images")

    del model
    torch.cuda.empty_cache()
    return rows, demo


def summarize(rows, key):
    vals = np.array([r[key] for r in rows], dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan"), 0
    return float(vals.mean()), float(vals.std()), len(vals)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    models = sys.argv[2:] or DEFAULT_MODELS

    data = load_cache(CACHE)
    masks = load_masks()
    n = min(n, len(data))
    print(f"{len(data)} cached images, {len(masks)} with masks; using first {n}\n")

    all_rows, demos = [], {}

    # ---- pass 1: every vanilla baseline, before anything is patched ---------
    print("=== pass 1: vanilla baselines (no HiLRP patches applied yet) ===")
    for m in models:
        print(f"  {m}")
        all_rows += gradcam_pass(m, data, masks, n)

    # ---- pass 2: HiLRP (patches the process from here on) ------------------
    print("\n=== pass 2: HiLRP ===")
    for m in models:
        print(f"  {m}")
        rows, demo = hilrp_pass(m, data, masks, n)
        all_rows += rows
        demos[m] = demo

    # ---------------------------------------------------------------- output
    fields = ["model", "method", "idx", "label", "pred", "counter",
              "map_corr", "inmask_pred", "inmask_counter", "inmask_gt", "gt_corr"]
    with open(os.path.join(OUT, "per_image.csv"), "w") as f:
        f.write(",".join(fields) + "\n")
        for r in all_rows:
            f.write(",".join(str(r.get(k, "")) for k in fields) + "\n")

    print("\n" + "=" * 78)
    print(f"{'model':<34s} {'method':<9s} {'corr':>14s} {'in-mask pred':>13s} {'in-mask cf':>12s}")
    print("=" * 78)
    summary = []
    for m in models:
        for meth in ("grad_cam", "hilrp"):
            rows = [r for r in all_rows if r["model"] == m and r["method"] == meth]
            if not rows:
                continue
            c, cs, _ = summarize(rows, "map_corr")
            p, ps, _ = summarize(rows, "inmask_pred")
            q, qs, _ = summarize(rows, "inmask_counter")
            flip = np.mean([
                1.0 for r in rows
                if not np.isnan(r["inmask_pred"]) and not np.isnan(r["inmask_counter"])
                and r["inmask_pred"] > 0 > r["inmask_counter"]
            ] or [0.0])
            print(f"{m:<34s} {meth:<9s} {c:+.3f}+-{cs:.3f} {p:+13.3f} {q:+12.3f}")
            summary.append(dict(model=m, method=meth, n=len(rows),
                                map_corr=c, map_corr_std=cs,
                                inmask_pred=p, inmask_pred_std=ps,
                                inmask_counter=q, inmask_counter_std=qs,
                                sign_flip_rate=float(flip)))

        mis = [r for r in all_rows if r["model"] == m and r["method"] == "hilrp"
               and not np.isnan(r["gt_corr"])]
        if mis:
            g, gs, k = summarize(mis, "gt_corr")
            ig, _, _ = summarize(mis, "inmask_gt")
            ip, _, _ = summarize(mis, "inmask_pred")
            print(f"{'':<34s} {'(misclf)':<9s} pred-vs-GT corr {g:+.3f}  "
                  f"in-mask pred {ip:+.3f} / GT {ig:+.3f}  (n={k})")
            summary.append(dict(model=m, method="hilrp_misclassified", n=k,
                                map_corr=g, map_corr_std=gs,
                                inmask_pred=ip, inmask_pred_std=float("nan"),
                                inmask_counter=ig, inmask_counter_std=float("nan"),
                                sign_flip_rate=float("nan")))
    print("=" * 78)

    sfields = ["model", "method", "n", "map_corr", "map_corr_std", "inmask_pred",
               "inmask_pred_std", "inmask_counter", "inmask_counter_std",
               "sign_flip_rate"]
    with open(os.path.join(OUT, "summary.csv"), "w") as f:
        f.write(",".join(sfields) + "\n")
        for r in summary:
            f.write(",".join(f"{r.get(k, '')}" for k in sfields) + "\n")

    make_figure(demos, models)
    print(f"\nwrote {OUT}/per_image.csv, summary.csv, class_sensitivity.png")


def make_figure(demos, models):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))                   # <repo>/scripts
    from figstyle import apply_paper_style, save, model_label, COL_W, FS_LABEL
    apply_paper_style()
    import matplotlib.pyplot as plt
    from hilrp.viz import normalize_for_display

    rows = [(m, d) for m in models for d in demos.get(m, [])[:2]]
    if not rows:
        return
    # Column width (this goes in at \linewidth), so 7-8 pt here is 7-8 pt in
    # print. Class ids move inside the panels: a per-row title carrying the id
    # does not fit in a 1.17 in column.
    panel = COL_W / 3
    fig_h = panel * len(rows) + 0.55
    fig, axes = plt.subplots(len(rows), 3, figsize=(COL_W, fig_h))
    axes = np.atleast_2d(axes)
    headers = ("Input", "Predicted Class", "Least-Likely Class")
    for i, (m, (idx, img, r_pred, r_cf, pred, bottom, label)) in enumerate(rows):
        raw = (img * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1).permute(1, 2, 0).numpy()
        axes[i, 0].imshow(raw)
        axes[i, 0].set_ylabel(model_label(m), fontsize=FS_LABEL)
        # each panel on its own scale: the claim is about the SIGN of the
        # evidence, which a shared scale washes out on the weaker panel
        for ax, rel, cls in ((axes[i, 1], r_pred, pred),
                             (axes[i, 2], r_cf, bottom)):
            v, vmax = normalize_for_display(rel)
            ax.imshow(raw, alpha=0.30)
            ax.imshow(v, cmap="bwr", vmin=-vmax, vmax=vmax, alpha=0.80)
            ax.text(0.03, 0.97, f"class {cls}", transform=ax.transAxes,
                    fontsize=5.5, ha="left", va="top",
                    bbox=dict(boxstyle="square,pad=0.15", fc="white",
                              ec="none", alpha=0.75))
        for c, ax in enumerate(axes[i]):
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(headers[c], fontsize=7)
    fig.tight_layout(pad=0.25, h_pad=0.5)
    # y is in figure fractions, so on a ~7.5 in tall canvas a fixed 1.02 still
    # lands on top of the column headers: offset by a physical 0.42 in instead.
    fig.suptitle("HiLRP Class Sensitivity: Evidence For the Predicted Class,\n"
                 "Against the Same Image Explained For the Least-Likely Class",
                 y=1 + 0.42 / fig_h)
    fig.savefig(os.path.join(OUT, "class_sensitivity.png"), dpi=400,
                bbox_inches="tight")
    save(fig, "figures/hilrp/class_sensitivity", also_pdf=False)


# ----------------------------------------------------------- figure-only paths
DEMO_CACHE = os.path.join("results", "figure_cache", "class_sensitivity.npz")


def _save_demo_cache(demos):
    """Persist the demo panels (images + both attributions) as plain arrays."""
    store = {}
    for m, items in demos.items():
        for k, (idx, img, r_pred, r_cf, pred, bottom, label) in enumerate(items[:2]):
            store[f"img~{m}~{k}"] = img.numpy()
            store[f"pred~{m}~{k}"] = r_pred
            store[f"cf~{m}~{k}"] = r_cf
            store[f"meta~{m}~{k}"] = np.array([idx, pred, bottom, label])
    os.makedirs(os.path.dirname(DEMO_CACHE), exist_ok=True)
    np.savez_compressed(DEMO_CACHE, **store)
    print(f"cached demo maps -> {DEMO_CACHE}")


def _load_demo_cache():
    z = np.load(DEMO_CACHE)
    demos = {}
    for key in z.files:
        if not key.startswith("meta~"):
            continue
        _, m, k = key.split("~")
        idx, pred, bottom, label = (int(v) for v in z[key])
        demos.setdefault(m, []).append(
            (idx, torch.from_numpy(z[f"img~{m}~{k}"]), z[f"pred~{m}~{k}"],
             z[f"cf~{m}~{k}"], pred, bottom, label))
    return demos


def demo_only(models, n=15):
    """Recompute ONLY the demo panels; the reported table stays summary.csv.

    `n` just has to be large enough for the first two distinct classes per
    model to show up, which is the same pair the full run picks.
    """
    data = load_cache(CACHE)
    masks = load_masks()
    demos = {}
    for m in models:
        print(f"  {m} (demo panels only)")
        _, demo = hilrp_pass(m, data, masks, min(n, len(data)))
        demos[m] = demo
    _save_demo_cache(demos)
    return demos


if __name__ == "__main__":
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    mdls = argv[1:] or DEFAULT_MODELS
    if "--replot" in flags and os.path.exists(DEMO_CACHE):
        make_figure(_load_demo_cache(), mdls)
    elif "--figure-only" in flags:
        make_figure(demo_only(mdls), mdls)
    else:
        main()
