"""RQ2: Does the pretraining objective determine what a model finds explanatory?

A controlled factorial that only HiLRP's label-free attribution makes possible.
We fix the architecture (ViT-B/16, identical 197x768 token layout) and vary only
the pretraining objective across four checkpoints:

    supervised (AugReg IN21k->IN1k) | DINO | MAE | CLIP

and explain each model under one common, label-free scalar: the view-invariance
similarity s = cos(cls(x), cls(flip(x)).detach()). Because the architecture,
scalar, and inputs are identical, any difference in the attribution is caused by
the pretraining objective alone. This is the question a linear-probe pipeline
cannot ask, because probing replaces every model's objective with a shared
classifier; HiLRP attributes each model in its own representational objective.

Per objective we report: the view-similarity value, label-free Pointing (does the
invariance evidence localize the object, with no labels used?), and the fraction
of relevance that reaches the pixels. We then compute the cross-objective
agreement of the maps (mean pairwise Spearman): low agreement means the objective
strongly determines where view-invariance evidence lives.

Run:  <mri-diffuser python> scripts/hilrp/rq2_pretraining.py [n_images]
"""
import os
import sys
import warnings
import itertools

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
import timm

OUT = os.path.join("results", "rq2_pretraining")
os.makedirs(OUT, exist_ok=True)

# DINOv2 variants are ViT-B/14 checkpoints instantiated at img_size=224 (16x16
# patch grid, interpolated pos-embed): architecture matched up to patch size.
# The reg4 variant adds 4 register tokens (timm order [cls, reg, patches], so
# the CLS index used by the similarity scalar is unchanged).
MODELS = {
    "supervised": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
    "DINO":       "vit_base_patch16_224.dino",
    "DINOv2":     "vit_base_patch14_dinov2.lvd142m",
    "DINOv2-reg": "vit_base_patch14_reg4_dinov2.lvd142m",
    "MAE":        "vit_base_patch16_224.mae",
    "CLIP":       "vit_base_patch16_clip_224.openai",
}
MEAN = np.array([0.485, 0.456, 0.406]); STD = np.array([0.229, 0.224, 0.225])


def denorm(t):
    return np.clip(t.permute(1, 2, 0).numpy() * STD + MEAN, 0, 1)


def pointing(pm, bboxes):
    h, w = pm.shape
    py, px = np.unravel_index(np.argmax(pm), pm.shape)
    return any(bb[0] * w <= px <= bb[2] * w and bb[1] * h <= py <= bb[3] * h for bb in bboxes)


def main():
    n_images = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    device = "cuda" if torch.cuda.is_available() else "cpu"
    import glob
    caches = glob.glob(os.path.join("data", "ImageNetS", "cache_validation_*.pt"))
    cache = max(caches, key=lambda p: int(p.rsplit("_", 1)[-1].split(".")[0]))
    data = torch.load(cache, map_location="cpu", weights_only=False)

    from xai_bench.methods.hilrp.vit_lxt import attribute_ssl_similarity, ensure_patched
    from xai_bench.methods.hilrp.pvt_lxt import audit_norm_patches
    ensure_patched()

    models = {}
    for tag, name in MODELS.items():
        m = timm.create_model(name, pretrained=True, num_classes=0,
                              img_size=224).eval().to(device)
        assert not audit_norm_patches(m, verbose=True), f"unpatched norms in {tag}"
        models[tag] = m

    tags = list(MODELS)
    per = {t: dict(sim=[], point=0, cons=[]) for t in tags}
    maps_by_img = []          # list of {tag: flat map}
    demo_maps = {t: [] for t in tags}

    for idx in range(n_images):
        x = data[idx]["image"].unsqueeze(0).to(device)
        x_ref = torch.flip(x, dims=(3,))
        maps = {}
        for t in tags:
            res = attribute_ssl_similarity(models[t], x, x_ref)
            pm = res["pixel_map"]
            per[t]["sim"].append(res["similarity"])
            per[t]["point"] += pointing(pm.numpy(), data[idx]["metadata"]["bboxes"])
            per[t]["cons"].append(res["stage_sums"][0][1])
            maps[t] = pm.numpy().flatten()
            if idx < 5:
                demo_maps[t].append(pm.numpy())
        maps_by_img.append(maps)

    # ---- per-objective summary ----
    print(f"RQ2: frozen ViT-B backbones ({len(tags)} pretrainings), {n_images} images, "
          f"label-free view-similarity scalar\n")
    print(f"{'objective':12s} {'similarity':>11s} {'labelfree-point':>16s} {'pixelR/scalar':>14s}")
    for t in tags:
        print(f"{t:12s} {np.mean(per[t]['sim']):>11.3f} "
              f"{per[t]['point']/n_images:>16.2f} "
              f"{np.mean(per[t]['cons']):>+14.3f}")

    # ---- cross-objective map agreement (mean pairwise Spearman) ----
    print("\nCross-objective map agreement (mean pairwise Spearman, low = objective matters):")
    M = np.zeros((len(tags), len(tags)))
    for i, a in enumerate(tags):
        for j, b in enumerate(tags):
            if i <= j:
                rs = [spearmanr(m[a], m[b]).statistic for m in maps_by_img]
                M[i, j] = M[j, i] = np.nanmean(rs)
    hdr = "            " + " ".join(f"{t[:6]:>7s}" for t in tags)
    print(hdr)
    for i, a in enumerate(tags):
        print(f"{a:12s}" + " ".join(f"{M[i,j]:>7.2f}" for j in range(len(tags))))
    off = M[np.triu_indices(len(tags), 1)]
    print(f"\nmean off-diagonal agreement: {off.mean():.3f} "
          f"(vs ~1.0 if the objective were irrelevant)")

    with open(os.path.join(OUT, "summary.csv"), "w") as f:
        f.write("objective,similarity,labelfree_pointing,pixel_cons\n")
        for t in tags:
            f.write(f"{t},{np.mean(per[t]['sim']):.4f},{per[t]['point']/n_images:.3f},"
                    f"{np.mean(per[t]['cons']):.4f}\n")
    np.save(os.path.join(OUT, "cross_objective_spearman.npy"), M)

    # ---- figure: same images, four objectives ----
    ns = min(5, n_images)
    fig, axes = plt.subplots(ns, 1 + len(tags), figsize=(2.1 * (1 + len(tags)), 2.1 * ns))
    for i in range(ns):
        axes[i, 0].imshow(denorm(data[i]["image"])); axes[i, 0].axis("off")
        if i == 0: axes[i, 0].set_title("input", fontsize=8)
        for j, t in enumerate(tags):
            m = demo_maps[t][i]; v = np.abs(m).max() + 1e-12
            axes[i, 1 + j].imshow(m, cmap="bwr", vmin=-v, vmax=v); axes[i, 1 + j].axis("off")
            if i == 0: axes[i, 1 + j].set_title(t, fontsize=8)
    fig.suptitle(f"RQ2: view-invariance evidence under {len(tags)} pretraining "
                 "objectives (frozen ViT-B)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "rq2_maps.png"), dpi=150)
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
