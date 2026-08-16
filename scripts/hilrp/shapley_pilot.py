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

Run:  <mri-diffuser python> scripts/hilrp/shapley_pilot.py [n_images] [n_perms]
"""
import os
import sys
import warnings

import numpy as np
import torch
import torchvision.transforms.functional as TF

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from skimage.segmentation import slic
from scipy.stats import spearmanr

import timm

OUT = os.path.join("results", "shapley_pilot")
os.makedirs(OUT, exist_ok=True)

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

MODEL = "swin_tiny_patch4_window7_224"     # override with argv[3]
N_SEGMENTS = 24
BLUR_KERNEL = 51          # heavy blur = information removal without OOD black
BATCH = 32


def get_hilrp_attributor(model_name):
    """Same dispatch as gate3: pick the HiLRP attributor for the architecture."""
    if model_name.startswith("swin"):
        from xai_bench.methods.hilrp.swin_lxt import attribute_swin, ensure_patched
        return attribute_swin, ensure_patched
    if model_name.startswith("mobilevit"):
        from xai_bench.methods.hilrp.mobilevit_lxt import attribute_mobilevit, ensure_patched
        return attribute_mobilevit, ensure_patched
    if model_name.startswith("efficientvit"):
        from xai_bench.methods.hilrp.efficientvit_lxt import attribute_efficientvit, ensure_patched
        return attribute_efficientvit, ensure_patched
    if model_name.startswith("pvt"):
        from xai_bench.methods.hilrp.pvt_lxt import attribute_pvt, ensure_patched
        return attribute_pvt, ensure_patched
    raise ValueError(model_name)


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

    model_name = sys.argv[3] if len(sys.argv) > 3 else MODEL
    # mobilevit needs raw [0,1] inputs (see gate3 adapt_input); pass 2 below uses
    # a bare timm model without the stats adapter, so guard until wired.
    assert not model_name.startswith("mobilevit"), "wire adapt_input before using mobilevit here"
    out = os.path.join(OUT, model_name)
    os.makedirs(out, exist_ok=True)
    data = torch.load(os.path.join("data", "ImageNetS", "cache_validation_100.pt"),
                      map_location="cpu", weights_only=False)

    # TWO PASSES, because the HiLRP patches are CLASS-level and would otherwise
    # alter the gradients of every baseline (identical forward numerics, but
    # detached-std LN and CP-detached attention change all gradient paths).
    # Pass 1 computes the Shapley reference and ALL baseline maps on the truly
    # vanilla classes; only then are the classes patched for the HiLRP pass.
    from xai_bench.models.timm_models import create_timm_model
    from xai_bench.registry import METHODS
    import captum.attr as ca

    wrapper = create_timm_model(model_name)     # vanilla pass
    wrapper.model.eval().to(device)
    vanilla = wrapper.model
    grad_cam = METHODS.get("grad_cam")(model_wrapper=wrapper, model=vanilla)
    ig = ca.IntegratedGradients(vanilla)
    sg = ca.NoiseTunnel(ca.Saliency(vanilla))

    def baseline_maps(x, target):
        maps = {}
        xg = x.clone().requires_grad_(True)
        vanilla.zero_grad(set_to_none=True)
        vanilla(xg)[0, target].backward()
        maps["saliency"] = xg.grad[0].abs().sum(0).cpu().numpy()
        maps["ig"] = ig.attribute(x, target=target, n_steps=32)[0].sum(0).detach().cpu().numpy()
        maps["smoothgrad"] = sg.attribute(x, target=target, nt_type="smoothgrad",
                                          nt_samples=16, stdevs=0.15)[0].abs().sum(0).detach().cpu().numpy()
        gc = grad_cam(x, target)
        maps["grad_cam"] = (gc[0] if gc.ndim == 3 else gc).detach().cpu().numpy()
        return maps

    method_names = ["hilrp", "saliency", "ig", "smoothgrad", "grad_cam"]

    # ---- pass 1: vanilla classes -- Shapley reference + baseline maps --------
    per_image = []
    for idx in range(n_images):
        img = data[idx]["image"]
        labels = make_regions(img)
        x = img.unsqueeze(0).to(device)
        with torch.no_grad():
            target = int(vanilla(x)[0].argmax())

        shap, mc_std = shapley_sampled(vanilla, img, labels, target, n_perms, device)
        np.save(os.path.join(out, f"regions_{idx}.npy"), labels)
        np.save(os.path.join(out, f"shapley_{idx}.npy"), shap)
        np.save(os.path.join(out, f"mc_std_{idx}.npy"), mc_std)

        base_r = {name: region_sums(m, labels)
                  for name, m in baseline_maps(x, target).items()}
        per_image.append((idx, labels, target, shap, mc_std, base_r))
        print(f"pass1 img{idx}: target {target}  SNR "
              f"{np.abs(shap).mean() / (mc_std.mean() + 1e-12):.1f}")

    del vanilla, wrapper, grad_cam, ig, sg      # nothing vanilla survives pass 2

    # ---- pass 2: patch the classes, HiLRP on a fresh instance -----------------
    attribute, ensure_patched = get_hilrp_attributor(model_name)
    ensure_patched()
    hmodel = timm.create_model(model_name, pretrained=True).eval().to(device)

    rows = []
    for idx, labels, target, shap, mc_std, base_r in per_image:
        x = data[idx]["image"].unsqueeze(0).to(device)
        res = attribute(hmodel, x, target=target)
        rhos = {"hilrp": spearmanr(shap, region_sums(res["pixel_map"].numpy(), labels)).statistic}
        for name, r in base_r.items():
            rhos[name] = spearmanr(shap, r).statistic

        snr = np.abs(shap).mean() / (mc_std.mean() + 1e-12)
        rows.append([idx, target] + [rhos[n] for n in method_names] + [snr])
        print(f"img{idx}: target {target}  " +
              "  ".join(f"{n} {rhos[n]:+.3f}" for n in method_names) + f"  SNR {snr:.1f}")

    with open(os.path.join(out, "agreement.csv"), "w") as f:
        f.write("idx,target," + ",".join("rho_" + n for n in method_names) + ",snr\n")
        for r in rows:
            f.write(",".join(str(v) for v in r) + "\n")
    arr = np.array([r[2:2 + len(method_names)] for r in rows], dtype=float)
    print(f"\n[{model_name}] mean Spearman vs Shapley (n={len(rows)}, m={n_perms} perms):")
    for j, n in enumerate(method_names):
        print(f"  {n:12s} {np.nanmean(arr[:, j]):+.3f}")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
