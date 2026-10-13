"""Regenerate the RQ2 label-free figure (view-invariance scalar under different
pretraining objectives) on clean single-object images (koala, panda, owl).
Figure only; the quantitative RQ2 table stays the n=30 result. -> figures/hilrp/rq2_maps.png

The maps are cached to results/figure_cache/rq2_maps.npz on the first run, so
restyling the figure never costs a second pass over five frozen ViT-Bs:

  python scripts/hilrp/rq2_custom_images.py            compute + plot
  python scripts/hilrp/rq2_custom_images.py --replot   plot from the cache
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))                       # <repo>/scripts

import numpy as np
from PIL import Image

from figstyle import apply_paper_style, lxt_transformers_shim, save, FULL_W
from hilrp.viz import normalize_for_display

lxt_transformers_shim()
import torch
import timm

apply_paper_style()
import matplotlib.pyplot as plt

OUT = "figures/hilrp"
CACHE = os.path.join("results", "figure_cache", "rq2_maps.npz")
os.makedirs(OUT, exist_ok=True)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
MODELS = {
    "Supervised": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
    "DINO":       "vit_base_patch16_224.dino",
    "DINOv2":     "vit_base_patch14_dinov2.lvd142m",
    "DINOv2-reg": "vit_base_patch14_reg4_dinov2.lvd142m",
    "MAE":        "vit_base_patch16_224.mae",
}
IMAGES = ["images/koala.jpg", "images/panda.jpg", "images/owl.jpg"]


def load(path):
    pil = Image.open(path).convert("RGB").resize((224, 224), Image.BICUBIC)
    x = torch.from_numpy(np.array(pil)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return (x - MEAN) / STD, np.array(pil) / 255.0


def compute():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from hilrp.vit_lxt import attribute_ssl_similarity, ensure_patched
    ensure_patched()
    models = {t: timm.create_model(n, pretrained=True, num_classes=0, img_size=224).eval().to(device)
              for t, n in MODELS.items()}
    tags = list(MODELS)
    inputs = [load(p) for p in IMAGES]

    demo = {t: [] for t in tags}
    for x, _ in inputs:
        x = x.to(device); x_ref = torch.flip(x, dims=(3,))
        for t in tags:
            res = attribute_ssl_similarity(models[t], x, x_ref)
            demo[t].append(res["pixel_map"].numpy())
        print("done one image", flush=True)

    store = {f"img{i}": inputs[i][1] for i in range(len(inputs))}
    for t in tags:
        for i, m in enumerate(demo[t]):
            store[f"map:{t}:{i}"] = m
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, **store)
    print(f"cached maps -> {CACHE}")
    return [inp[1] for inp in inputs], demo, tags


def from_cache():
    z = np.load(CACHE)
    tags = list(MODELS)
    n = sum(1 for k in z.files if k.startswith("img"))
    raws = [z[f"img{i}"] for i in range(n)]
    demo = {t: [z[f"map:{t}:{i}"] for i in range(n)] for t in tags}
    return raws, demo, tags


def make_figure(raws, demo, tags):
    n = len(raws)
    # Wide 6-column grid: authored at \textwidth for a figure* so the 8 pt
    # column headers print at 8 pt instead of being scaled to ~2 pt.
    panel = FULL_W / (1 + len(tags))
    fig_h = panel * n + 0.35
    fig, ax = plt.subplots(n, 1 + len(tags), figsize=(FULL_W, fig_h))
    for i in range(n):
        ax[i, 0].imshow(raws[i]); ax[i, 0].axis("off")
        if i == 0:
            ax[i, 0].set_title("Input")
        for j, t in enumerate(tags):
            # Percentile-99 clipping, not symmetric-max: SSL relevance is
            # heavy-tailed and a raw max washes the panel out to near-white
            # (hilrp/viz.py).
            m, v = normalize_for_display(demo[t][i], percentile=99.0)
            ax[i, 1 + j].imshow(m, cmap="bwr", vmin=-v, vmax=v)
            ax[i, 1 + j].axis("off")
            if i == 0:
                ax[i, 1 + j].set_title(t)
    fig.tight_layout(pad=0.15)
    # physical offset, so clearance does not shrink as the grid gets taller
    fig.suptitle(f"RQ2: View-Invariance Evidence Under {len(tags)} Pretraining "
                 "Objectives (frozen ViT-B)", y=1 + 0.22 / fig_h)
    save(fig, os.path.join(OUT, "rq2_maps"), also_pdf=False)


if __name__ == "__main__":
    if "--replot" in sys.argv and os.path.exists(CACHE):
        make_figure(*from_cache())
    else:
        make_figure(*compute())
