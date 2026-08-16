"""Equivariance consistency test (empirical validation for the theorem).

Claim under test: HiLRP attribution inherits the forward's symmetry group.
For Swin (no absolute position embedding), the forward is exactly equivariant
under CYCLIC translations by multiples of patch_size x window_size = 28 px.
The attribution should then satisfy R(roll(x, v)) == roll(R(x), v) at those
shifts, up to float error, and should NOT be consistent at unaligned shifts,
because the forward itself changes there (the explanation correctly tracks the
model, not the image).

Metrics per shift v:
  forward drift   |logit(roll(x)) - logit(x)| / |logit(x)|   (the reference line)
  map consistency Spearman( R(roll(x)), roll(R(x)) )  and relative L2

Cyclic shifts avoid boundary confounds. Baselines for contrast: saliency
(deterministic, should track forward), smoothgrad (stochastic sampling, breaks
consistency by construction unless seeds are shared).

Run:  <mri-diffuser python> scripts/hilrp/equivariance_test.py
"""
import os
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scipy.stats import spearmanr
import timm

MODEL = "swin_tiny_patch4_window7_224"
SHIFTS = [(28, 28), (56, 28), (4, 4), (13, 13)]   # aligned, aligned, patch-only, unaligned
N_IMAGES = 8


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = torch.load(os.path.join("data", "ImageNetS", "cache_validation_100.pt"),
                      map_location="cpu", weights_only=False)

    from xai_bench.methods.hilrp.swin_lxt import attribute_swin, ensure_patched
    ensure_patched()
    model = timm.create_model(MODEL, pretrained=True).eval().to(device)

    def hilrp_map(x, target):
        return attribute_swin(model, x, target=target)["pixel_map"].numpy()

    def saliency_map(x, target):
        xg = x.clone().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        model(xg)[0, target].backward()
        return xg.grad[0].abs().sum(0).cpu().numpy()

    methods = {"hilrp": hilrp_map, "saliency": saliency_map}

    print(f"{'shift':>9s} {'fwd drift':>10s}", end="")
    for name in methods:
        print(f"  {name+'_rho':>12s} {name+'_L2':>10s}", end="")
    print()

    for v in SHIFTS:
        drifts, stats = [], {n: [[], []] for n in methods}
        for idx in range(N_IMAGES):
            x = data[idx]["image"].unsqueeze(0).to(device)
            xs = torch.roll(x, shifts=v, dims=(2, 3))
            with torch.no_grad():
                l0, l1 = model(x), model(xs)
            t = int(l0[0].argmax())
            drifts.append(abs(l1[0, t].item() - l0[0, t].item()) / (abs(l0[0, t].item()) + 1e-12))

            for name, fn in methods.items():
                m0 = fn(x, t)
                m1 = fn(xs, t)
                m0s = np.roll(m0, v, axis=(0, 1))          # roll the base map
                stats[name][0].append(spearmanr(m1.flatten(), m0s.flatten()).statistic)
                stats[name][1].append(rel_l2(m1, m0s))

        print(f"{str(v):>9s} {np.mean(drifts):>10.2e}", end="")
        for name in methods:
            print(f"  {np.mean(stats[name][0]):>12.4f} {np.mean(stats[name][1]):>10.4f}", end="")
        print()

    print("\nprediction: at (28,28)/(56,28) forward drift ~0 and hilrp rho ~1.0;")
    print("at (4,4)/(13,13) the forward itself changes, so consistency drops for")
    print("every faithful method: the attribution tracks the model, not the image.")


if __name__ == "__main__":
    main()
