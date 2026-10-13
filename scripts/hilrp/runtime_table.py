"""Runtime and peak memory per explanation, across the full method set.

Measured on ViT-B/16 at batch size 1, which is the worst case per image: methods
that amortize over a batch (HiLRP, the gradient family) look worst here, and
perturbation methods that are already sequential look unchanged.

Ordering matters and is not cosmetic. HiLRP's rule patches are class-level and
process-global, so timing a baseline after HiLRP has been imported measures the
PATCHED forward, not the baseline's own. Every vanilla method is therefore timed
first, and HiLRP last. (The earlier version of this script timed HiLRP first,
which contaminated every baseline number it produced.)

Methods use the same configuration as the benchmark grid, taken from the shared
registry rather than re-specified here, so the costs correspond to the numbers
reported elsewhere in the paper.

Run:  <mri-diffuser python> scripts/hilrp/runtime_table.py
Out:  results/runtime/runtime.csv  (+ a LaTeX-ready dump on stdout)
"""
import os
import sys
import time
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# transformers shim required by the LXT vision models on this version pin
import transformers.pytorch_utils as _pu
if not hasattr(_pu, "find_pruneable_heads_and_indices"):
    def _fp(heads, n_heads, head_size, already):
        mask = torch.ones(n_heads, head_size); heads = set(heads) - already
        for h in heads:
            h = h - sum(1 for x in already if x < h); mask[h] = 0
        return heads, torch.arange(len(mask.view(-1)))[mask.view(-1).eq(1)].long()
    _pu.find_pruneable_heads_and_indices = _fp

from xai_bench.registry import MODELS, METHODS
import xai_bench.methods  # noqa: F401

OUT = os.path.join("results", "runtime")
os.makedirs(OUT, exist_ok=True)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "vit_base_patch16_224"
TARGET = 1

# (registry name, display name, timed iterations, forward passes per explanation)
VANILLA = [
    ("saliency",            "Saliency",             20, "1"),
    ("input_x_gradient",    "Input$\\times$Grad",   20, "1"),
    ("grad_cam",            "Grad-CAM",             20, "1"),
    ("grad_cam_plus_plus",  "Grad-CAM++",           20, "1"),
    ("attention_rollout",   "Attention Rollout",    20, "1"),
    ("smoothgrad",          "SmoothGrad",           10, "20"),
    ("vargrad",             "VarGrad",              10, "20"),
    ("gradient_shap",       "GradientSHAP",         10, "20"),
    ("integrated_gradients", "Integrated Gradients", 5, "50"),
    ("occlusion",           "Occlusion",             3, "784"),
    ("lime",                "LIME",                  3, "100"),
    ("rise",                "RISE",                  3, "2000"),
]


def timeit(fn, n=20, warmup=3):
    for _ in range(warmup):
        fn()
    if DEV == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    if DEV == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / n * 1000
    peak = torch.cuda.max_memory_allocated() / 1e9 if DEV == "cuda" else float("nan")
    return ms, peak


def main():
    if DEV == "cuda":
        print(f"device: {torch.cuda.get_device_name(0)}")
    x = torch.randn(1, 3, 224, 224, device=DEV)
    results = []

    # ---- pass 1: every vanilla method, before HiLRP patches anything --------
    for name, disp, iters, passes in VANILLA:
        try:
            wrapper = MODELS.get(MODEL)()
            wrapper.model.to(DEV).eval()
            fn = METHODS.get(name)(model_wrapper=wrapper, model=wrapper.model)
            ms, mem = timeit(lambda: fn(x.clone(), TARGET), n=iters)
            results.append((disp, ms, mem, passes))
            print(f"  {disp:22s} {ms:9.1f} ms  {mem:6.2f} GB")
            del fn, wrapper
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  {disp:22s} FAILED: {str(e)[:70]}")
            results.append((disp, float("nan"), float("nan"), passes))

    # ---- pass 2: HiLRP (patches the process from here on) ------------------
    from hilrp.vit_lxt import attribute_vit, ensure_patched
    ensure_patched()
    wrapper = MODELS.get(MODEL)()
    wrapper.model.to(DEV).eval()
    inner = wrapper.model.model if hasattr(wrapper.model, "model") else wrapper.model

    def hilrp():
        r = attribute_vit(inner, x.clone(), target=TARGET)
        inner.zero_grad(set_to_none=True)
        return r

    ms, mem = timeit(hilrp, n=20)
    results.append(("\\textbf{HiLRP (ours)}", ms, mem, "1"))
    print(f"  {'HiLRP (ours)':22s} {ms:9.1f} ms  {mem:6.2f} GB")

    with open(os.path.join(OUT, "runtime.csv"), "w") as f:
        f.write("method,time_ms,peak_mem_gb,forward_passes\n")
        for disp, t, m, p in results:
            f.write(f"\"{disp}\",{t:.1f},{m:.3f},{p}\n")

    print("\n% LaTeX rows (sorted by time)")
    for disp, t, m, p in sorted(results, key=lambda r: (np.isnan(r[1]), r[1])):
        print(f"    {disp:26s} & ${t:,.0f}$ & ${m:.2f}$ & ${p}$ \\\\")
    print(f"\nwrote {OUT}/runtime.csv")


if __name__ == "__main__":
    main()
