"""Grad-CAM target-layer sensitivity: is a low Pointing score the architecture or the layer?

The bench's generic Grad-CAM reads the output of ``forward_features``. On
EfficientViT that is the last stage output, a real [B, 384, 7, 7] spatial map,
but it sits UPSTREAM of the head's own 1x1 expansion conv (``head.in_conv``,
384 -> 2304 + BN + Hardswish) and of a two-layer MLP classifier. The spatial map
that actually feeds global pooling is ``head.in_conv``'s output, so the generic
protocol and the textbook "last conv map before pooling" target differ on this
family. This script re-runs the same Grad-CAM at every candidate spatial layer
so the README can report the protocol number and the best-layer number side by
side instead of attributing the gap to the architecture.

  python scripts/bench/gradcam_target_layers.py efficientvit_b2 1000
  python scripts/bench/gradcam_target_layers.py efficientvit_b2 1000 voc

VANILLA PROCESS ONLY: nothing here may import the HiLRP ports (class-level
patches would alter these gradients). Per-image hits are written to
results/scaled/<model>_gradcam@<layer>.npy on the same images as scaled_eval.py,
so they pair with its cached HiLRP hits.
"""
import os
import sys
import warnings

import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.bench.scaled_eval import DEVICE, load_cache, npy_name, pointing  # noqa: E402


def candidate_layers(model):
    """Every spatial map a CAM could target, coarse to fine, as (name, module).

    ``forward_features`` ends at the last entry of ``stages``; ``head.in_conv``
    (EfficientViT) is the map that global pooling actually consumes.
    """
    layers = [(f"stages.{i}", s) for i, s in enumerate(model.stages)]
    in_conv = getattr(getattr(model, "head", None), "in_conv", None)
    if in_conv is not None:
        layers.append(("head.in_conv", in_conv))
    return layers


def cams_at_layers(model, layers, x, target):
    """Grad-CAM at every layer from ONE forward and ONE backward."""
    acts, hooks = {}, []
    for name, module in layers:
        def hook(_m, _i, out, name=name):
            out.retain_grad()
            acts[name] = out
        hooks.append(module.register_forward_hook(hook))
    try:
        model.zero_grad(set_to_none=True)
        model(x)[0, target].backward()
    finally:
        for h in hooks:
            h.remove()

    cams = {}
    for name, A in acts.items():
        weights = A.grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * A).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cams[name] = cam[0, 0].detach().cpu().numpy()
    return cams


def ci(hits):
    rng = np.random.RandomState(0)
    boot = [hits[rng.randint(0, len(hits), len(hits))].mean() for _ in range(2000)]
    return hits.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def mcnemar(a, b):
    from scipy.stats import binomtest
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(min(only_a, only_b), only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return only_a, only_b, p


def main(model_name, n, dataset="imagenets"):
    from xai_bench.registry import MODELS
    data, path = load_cache(dataset)
    n = min(n, len(data))
    mw = MODELS.get(model_name)()
    mw.model.eval().to(DEVICE)
    layers = candidate_layers(mw.model)
    protocol_layer = f"stages.{len(mw.model.stages) - 1}"   # == forward_features

    hits = {name: np.zeros(n, dtype=bool) for name, _ in layers}
    blank = {name: 0 for name, _ in layers}                 # all-zero CAM after ReLU
    for i in range(n):
        x = data[i]["image"].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            t = int(mw(x).argmax())
        for name, cam in cams_at_layers(mw.model, layers, x, t).items():
            hits[name][i] = pointing(cam, data[i]["metadata"]["bboxes"])
            blank[name] += int(cam.max() <= 0)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{n}", flush=True)

    for name in hits:
        np.save(npy_name(model_name, f"gradcam@{name}", dataset), hits[name])

    print(f"=== Grad-CAM target layers, {model_name} [{dataset}] (n={n}, cache={path}) ===")
    hilrp_path = npy_name(model_name, "hilrp", dataset)
    hilrp = np.load(hilrp_path)[:n] if os.path.exists(hilrp_path) else None
    for name, _ in layers:
        m, lo, hi = ci(hits[name])
        tag = "  <- bench protocol (forward_features)" if name == protocol_layer else ""
        line = f"  {name:14s} {m:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  blank maps {blank[name]:4d}/{n}"
        if hilrp is not None and len(hilrp) == n:
            h_only, g_only, p = mcnemar(hilrp, hits[name])
            line += f"  | vs HiLRP {hilrp.mean():.3f}: HiLRP-only={h_only}, CAM-only={g_only}, p={p:.1e}"
        print(line + tag)

    legacy = npy_name(model_name, "gradcam", dataset)
    if os.path.exists(legacy):
        same = np.array_equal(np.load(legacy)[:n], hits[protocol_layer])
        print(f"  protocol-layer hits identical to cached scaled_eval.py run: {same}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "imagenets")
