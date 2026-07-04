"""Sampled permutation Shapley reference: pilot run (NOTES item: the slow track).

This is the PRIMARY evidence yardstick (CLAUDE.md validation design): axiomatic
uniqueness, unbiased, 1/sqrt(m) convergence. The estimator is generic over any
region partition. The pilot uses SLIC superpixels (no checkpoint download, fast)
to de-risk the estimator: convergence, masking strategy, batching, agreement
computation. The paper run swaps in SAM segments; all importance signal comes
from the target model either way, so there is no second-model faithfulness trap.

Masking uses Gaussian-blur infill, not black patches (the TIS OOD-token critique).

Per image, writes to results/shapley_pilot/:
  regions_{idx}.npy     [H, W] int region labels
  shapley_{idx}.npy     [K] sampled Shapley values (target-class logit game)
  mc_std_{idx}.npy      [K] Monte Carlo std of the estimate
Plus agreement.csv: Spearman agreement of HiLRP / grad_cam / saliency maps with
the Shapley reference, aggregated per region.

Run:  <mri-diffuser python> scripts/shapley_pilot.py [n_images] [n_perms]
"""
import os
import sys
import warnings

import numpy as np
import torch
import torchvision.transforms.functional as TF

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skimage.segmentation import slic
from scipy.stats import spearmanr

import timm

OUT = os.path.join("results", "shapley_pilot")
os.makedirs(OUT, exist_ok=True)

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

MODEL = "swin_tiny_patch4_window7_224"
N_SEGMENTS = 24
BLUR_KERNEL = 51          # heavy blur = information removal without OOD black
BATCH = 32


def make_regions(img_norm):
    """SLIC superpixels on the denormalized image. Returns [H, W] int labels."""
    raw = (img_norm * STD + MEAN).clamp(0, 1).permute(1, 2, 0).numpy()
    labels = slic(raw, n_segments=N_SEGMENTS, compactness=20, start_label=0)
    return labels


def shapley_sampled(model, img_norm, labels, target, n_perms, device):
    """Sampled permutation Shapley over regions for the target-class logit.

    v(T) = logit_target(image with regions in T intact, rest blur-filled).
    One permutation contributes K marginal terms; all K+1 prefix states of a
    permutation are evaluated in batched forward passes.
    """
    K = labels.max() + 1
    x = img_norm.to(device)
    blurred = TF.gaussian_blur(x, BLUR_KERNEL).to(device)
    masks = torch.stack([torch.from_numpy(labels == k) for k in range(K)]).to(device)

    vals = np.zeros((n_perms, K), dtype=np.float64)
    rng = np.random.RandomState(0)

    for p in range(n_perms):
        perm = rng.permutation(K)
        # build the K+1 prefix images: start all-blurred, add one region at a time
        imgs = torch.empty(K + 1, *x.shape, device=device)
        cur = blurred.clone()
        imgs[0] = cur
        for i, k in enumerate(perm):
            m = masks[k]
            cur = torch.where(m.unsqueeze(0), x, cur)
            imgs[i + 1] = cur
        logits = []
        with torch.no_grad():
            for b in range(0, K + 1, BATCH):
                logits.append(model(imgs[b:b + BATCH])[:, target])
        logits = torch.cat(logits).double().cpu().numpy()
        vals[p, perm] = np.diff(logits)          # marginal contribution per region

    shap = vals.mean(0)
    mc_std = vals.std(0) / np.sqrt(n_perms)
    return shap, mc_std


def region_sums(pixel_map, labels):
    K = labels.max() + 1
    return np.array([pixel_map[labels == k].sum() for k in range(K)])


def main():
    n_images = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    n_perms = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = timm.create_model(MODEL, pretrained=True).eval().to(device)
    data = torch.load(os.path.join("data", "ImageNetS", "cache_validation_100.pt"),
                      map_location="cpu", weights_only=False)

    # attribution methods to score against the reference
    from xai_bench.methods.hilrp.swin_lxt import attribute_swin, ensure_patched
    ensure_patched()

    rows = []
    for idx in range(n_images):
        img = data[idx]["image"]
        labels = make_regions(img)
        x = img.unsqueeze(0).to(device)
        with torch.no_grad():
            target = int(model(x)[0].argmax())

        shap, mc_std = shapley_sampled(model, img, labels, target, n_perms, device)
        np.save(os.path.join(OUT, f"regions_{idx}.npy"), labels)
        np.save(os.path.join(OUT, f"shapley_{idx}.npy"), shap)
        np.save(os.path.join(OUT, f"mc_std_{idx}.npy"), mc_std)

        # HiLRP agreement (patched forward changes logits negligibly; same target)
        res = attribute_swin(model, x, target=target)
        hilrp_r = region_sums(res["pixel_map"].numpy(), labels)
        rho_hilrp = spearmanr(shap, hilrp_r).statistic

        # simple gradient baseline for contrast (saliency = |grad| sum)
        xg = x.clone().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        model(xg)[0, target].backward()
        sal = xg.grad[0].abs().sum(0).cpu().numpy()
        rho_sal = spearmanr(shap, region_sums(sal, labels)).statistic

        snr = np.abs(shap).mean() / (mc_std.mean() + 1e-12)
        rows.append((idx, target, rho_hilrp, rho_sal, snr))
        print(f"img{idx}: target {target}  rho(HiLRP) {rho_hilrp:+.3f}  "
              f"rho(saliency) {rho_sal:+.3f}  shapley SNR {snr:.1f}")

    with open(os.path.join(OUT, "agreement.csv"), "w") as f:
        f.write("idx,target,rho_hilrp,rho_saliency,snr\n")
        for r in rows:
            f.write(",".join(str(v) for v in r) + "\n")
    arr = np.array([[r[2], r[3]] for r in rows], dtype=float)
    print(f"\nmean Spearman vs Shapley: HiLRP {np.nanmean(arr[:,0]):+.3f}   "
          f"saliency {np.nanmean(arr[:,1]):+.3f}   (n={len(rows)}, m={n_perms} perms)")
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
