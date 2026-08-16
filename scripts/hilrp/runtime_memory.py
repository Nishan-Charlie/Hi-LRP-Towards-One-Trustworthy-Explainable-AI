"""Runtime + peak-memory of HiLRP vs Grad-CAM, Integrated Gradients, Attention
Rollout on ViT-B/16 (single 224x224 image, forward+backward). Reports mean ms
over timed iterations and peak GPU memory. Addresses the overhead question.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
import transformers.pytorch_utils as _pu
if not hasattr(_pu, "find_pruneable_heads_and_indices"):
    def _fp(heads, n_heads, head_size, already):
        mask = torch.ones(n_heads, head_size); heads = set(heads) - already
        for h in heads:
            h = h - sum(1 for x in already if x < h); mask[h] = 0
        return heads, torch.arange(len(mask.view(-1)))[mask.view(-1).eq(1)].long()
    _pu.find_pruneable_heads_and_indices = _fp

import numpy as np, timm
from xai_bench.registry import MODELS
import xai_bench.methods  # noqa
from xai_bench.methods.hilrp.vit_lxt import attribute_vit, ensure_patched

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def timeit(fn, n=20, warmup=3):
    for _ in range(warmup):
        fn()
    if DEV == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(n):
        fn()
    if DEV == "cuda":
        torch.cuda.synchronize()
    ms = (time.time() - t0) / n * 1000
    peak = torch.cuda.max_memory_allocated() / 1e6 if DEV == "cuda" else float("nan")
    return ms, peak


def main():
    mw = MODELS.get("vit_base_patch16_224")()
    mw.model.to(DEV).eval()
    inner = mw.model.model if hasattr(mw.model, "model") else mw.model
    x = torch.randn(1, 3, 224, 224, device=DEV)

    results = {}

    # HiLRP (forward + backward, CP rule)
    ensure_patched()
    def hilrp():
        r = attribute_vit(inner, x.clone(), target=1)
        inner.zero_grad(set_to_none=True)
        return r
    results["HiLRP (ours)"] = timeit(hilrp)

    # Grad-CAM
    gc = MODELS.get("vit_base_patch16_224")(); gc.model.to(DEV).eval()
    from xai_bench.registry import METHODS
    gcm = METHODS.get("grad_cam")(model_wrapper=gc, model=gc.model)
    results["Grad-CAM"] = timeit(lambda: gcm(x.clone(), target=1))

    # Integrated Gradients (captum, default 50 steps)
    try:
        from captum.attr import IntegratedGradients
        ig_model = MODELS.get("vit_base_patch16_224")(); ig_model.model.to(DEV).eval()
        ig = IntegratedGradients(ig_model.model)
        results["Integrated Grads"] = timeit(
            lambda: ig.attribute(x.clone().requires_grad_(True), target=1, n_steps=50), n=5)
    except Exception as e:
        print("IG failed:", str(e)[:80])

    # Attention Rollout
    try:
        ar = MODELS.get("vit_base_patch16_224")(); ar.model.to(DEV).eval()
        arm = METHODS.get("attention_rollout")(model_wrapper=ar, model=ar.model)
        results["Attention Rollout"] = timeit(lambda: arm(x.clone(), target=1))
    except Exception as e:
        print("Rollout failed:", str(e)[:80])

    print(f"\nViT-B/16, single image, {DEV}:")
    print(f"{'method':20s} {'time (ms)':>10s} {'peak mem (MB)':>14s}")
    for k, (ms, mem) in results.items():
        print(f"{k:20s} {ms:>10.1f} {mem:>14.0f}")


if __name__ == "__main__":
    main()
