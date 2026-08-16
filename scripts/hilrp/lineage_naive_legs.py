"""AttnLRP-naive legs on the hierarchical/hybrid models (lineage completion).

"Naive" = what a user gets today by applying LXT's generic flat-transformer
recipe to an unsupported architecture: nn.GELU and nn.LayerNorm identity
patches plus zennit Gamma composites, and NOTHING architecture-specific. We are
deliberately GENEROUS: the naive baseline also gets the BN-merge canonizer and
correct preprocessing (both are generic zennit/bench hygiene). What it does not
get is exactly our contribution set:

  * window / separable / linear attention rules (CP or uniform)
  * the norm-subclass guard (timm LayerNorm subclass, GroupNorm1)
  * activation identity rules beyond GELU (SiLU, Hardswish)

MUST run in its own process: our ports patch classes globally, so this script
never imports the hilrp port modules. Reports pointing plus conservation at the
deepest capture (stem/patch-embed output) and the head-adjacent capture, same
protocol as scripts/hilrp/lineage_comparison.py.

Run:  <mri-diffuser python> scripts/hilrp/lineage_naive_legs.py
"""
import os
import sys
import warnings
from functools import partial

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import timm
from lxt.efficient.patches import patch_method, non_linear_forward, layer_norm_forward
from lxt.efficient.zennit_patches import monkey_patch_zennit
from zennit.canonizers import NamedMergeBatchNorm
from zennit.composites import LayerMapComposite
from zennit.rules import Gamma

N_IMAGES = 1000

MODELS = ["swin_base_patch4_window7_224", "pvt_v2_b2", "efficientvit_b1",
          "efficientvit_b2", "maxvit_small_tf_224", "mobilevitv2_100"]

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def naive_patch_once():
    """LXT's generic recipe, class-level: GELU + nn.LayerNorm only."""
    patch_method(non_linear_forward, nn.GELU, keep_original=True)
    patch_method(layer_norm_forward, nn.LayerNorm)
    monkey_patch_zennit()


def bn_pairs(model):
    pairs = []
    for name, mod in model.named_modules():
        conv = getattr(mod, "conv", None)
        if not isinstance(conv, nn.Conv2d):
            continue
        for bn_attr in ("bn", "norm"):
            bn = getattr(mod, bn_attr, None)
            if isinstance(bn, nn.BatchNorm2d):
                pairs.append(([f"{name}.conv"], f"{name}.{bn_attr}"))
                break
    return pairs


def adapt_input(img, model):
    mean = torch.tensor(model.pretrained_cfg["mean"]).view(3, 1, 1)
    std = torch.tensor(model.pretrained_cfg["std"]).view(3, 1, 1)
    raw = (img * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)
    return (raw - mean) / std


def deepest_module(model):
    for attr in ("patch_embed", "stem"):
        if hasattr(model, attr):
            return getattr(model, attr)
    raise ValueError("no stem/patch_embed")


def attribute_naive(model, x, gamma=0.25):
    captures, hooks = [], []

    def grab(name):
        def hook(m, a, o):
            t = o[0] if isinstance(o, tuple) else o
            t.retain_grad()
            captures.append((name, t))
        return hook

    hooks.append(deepest_module(model).register_forward_hook(grab("deep")))
    last_stage = model.stages[-1] if hasattr(model, "stages") else model.layers[-1]
    hooks.append(last_stage.register_forward_hook(grab("head_adjacent")))

    x = x.clone().detach().requires_grad_(True)
    comp = LayerMapComposite(
        [(nn.Conv2d, Gamma(gamma)), (nn.Linear, Gamma(gamma))],
        canonizers=[NamedMergeBatchNorm(bn_pairs(model))],
    )
    try:
        with comp.context(model) as mod:
            logits = mod(x)
            t = int(logits[0].argmax())
            mod.zero_grad(set_to_none=True)
            logits[0, t].backward()
    finally:
        for h in hooks:
            h.remove()

    denom = logits[0, t].item() or 1.0
    sums = {n: (tn * tn.grad)[0].sum().item() / denom
            for n, tn in captures if tn.grad is not None}
    pixel_map = (x * x.grad)[0].sum(0).detach().cpu()
    return pixel_map, sums


def pointing(pm, bboxes):
    h, w = pm.shape
    py, px = np.unravel_index(np.argmax(pm), pm.shape)
    return any(bb[0] * w <= px <= bb[2] * w and bb[1] * h <= py <= bb[3] * h
               for bb in bboxes)


def main():
    global N_IMAGES
    naive_patch_once()
    import glob
    caches = glob.glob(os.path.join("data", "ImageNetS", "cache_validation_*.pt"))
    path = max(caches, key=lambda p: int(p.rsplit("_", 1)[-1].split(".")[0]))
    data = torch.load(path, map_location="cpu", weights_only=False)
    N_IMAGES = min(N_IMAGES, len(data))
    print(f"cache: {path} ({len(data)} imgs), N_IMAGES={N_IMAGES}")

    print("AttnLRP-naive legs (generic LXT recipe, generous: canonizer + correct stats)")
    for name in MODELS:
        model = timm.create_model(name, pretrained=True).eval().cuda()
        hits, deep, head = 0, [], []
        hit_array = np.zeros(N_IMAGES, dtype=bool)
        for idx in range(N_IMAGES):
            x = adapt_input(data[idx]["image"], model).unsqueeze(0).cuda()
            pm, sums = attribute_naive(model, x)
            hit_bool = pointing(pm.numpy(), data[idx]["metadata"]["bboxes"])
            hits += hit_bool
            hit_array[idx] = hit_bool
            deep.append(sums.get("deep", np.nan))
            head.append(sums.get("head_adjacent", np.nan))
        print(f"  {name:22s} pointing {hits}/{N_IMAGES} = {hits / N_IMAGES:.3f}   "
              f"cons deep {np.nanmean(deep):+.2f} +- {np.nanstd(deep):.2f}   "
              f"head {np.nanmean(head):+.2f} +- {np.nanstd(head):.2f}")
        
        # Save hit array
        OUT = os.path.join("results", "hilrp_lineage")
        os.makedirs(OUT, exist_ok=True)
        np.save(os.path.join(OUT, f"hits_{name}_attnlrp_naive.npy"), hit_array)

    print("\nHiLRP reference (same protocol, gate3):")
    print("  pvt_v2_b2              0.980   cons smooth monotone (deep ~+0.39, head +1.01)")
    print("  efficientvit_b2        0.960   (head-adjacent trace inflates, see NOTES)")
    print("  mobilevitv2_100        0.860   cons head +1.05")


if __name__ == "__main__":
    main()
