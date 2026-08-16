"""Sampled permutation Shapley reference: Gate 4 (SAM segments).

This script uses the Segment Anything Model (SAM) to partition the image into
semantically meaningful segments, then evaluates the sampled Shapley value
of each segment for the target-class logit.

The SAM generator typically returns overlapping or unassigned pixels. This script
resolves them into a strict non-overlapping HxW partition matrix so that every
pixel belongs to exactly one region.

Run:  <mri-diffuser python> scripts/hilrp/gate4_shapley_sam.py [n_images] [n_perms] [model_name]
"""
import os
import sys
import warnings

import numpy as np
import torch
import torchvision.transforms.functional as TF
from scipy.stats import spearmanr
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

import timm

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

OUT = os.path.join("results", "gate4_shapley_sam")
os.makedirs(OUT, exist_ok=True)

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

MODEL = "swin_tiny_patch4_window7_224"     
BLUR_KERNEL = 51          
BATCH = 32

# Hardcap for number of SAM segments to prevent computational explosion
# (If SAM generates > 32 segments, we only use the 32 largest, dumping the rest into a background mask)
MAX_SEGMENTS = 32

def get_hilrp_attributor(model_name):
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

def make_sam_regions(img_norm, mask_generator):
    """
    Generate non-overlapping segments using SAM.
    Returns [H, W] int labels.
    """
    raw = (img_norm * STD + MEAN).clamp(0, 1).permute(1, 2, 0).numpy()
    raw_uint8 = (raw * 255).astype(np.uint8)
    
    sam_masks = mask_generator.generate(raw_uint8)
    
    # Sort masks by area (largest first)
    sam_masks = sorted(sam_masks, key=lambda x: x['area'], reverse=True)
    
    H, W = raw_uint8.shape[:2]
    labels = np.full((H, W), -1, dtype=np.int32)
    
    current_label = 0
    # Apply up to MAX_SEGMENTS-1 masks
    for mask_data in sam_masks[:MAX_SEGMENTS - 1]:
        m = mask_data['segmentation']
        # Only assign if not already assigned
        labels[(m == True) & (labels == -1)] = current_label
        current_label += 1
        
    # Any unassigned pixels become the final background segment
    labels[labels == -1] = current_label
    
    # Ensure sequential IDs (in case some masks were completely occluded)
    unique_ids = np.unique(labels)
    final_labels = np.zeros_like(labels)
    for new_id, old_id in enumerate(unique_ids):
        final_labels[labels == old_id] = new_id
        
    return final_labels

def shapley_sampled(model, img_norm, labels, target, n_perms, device, is_mobilevit=False):
    K = labels.max() + 1
    x = img_norm.to(device)
    
    # mobilevit uses raw [0,1] inputs internally via its own adapter, but the input passed
    # here is standardized for baseline maps.
    
    blurred = TF.gaussian_blur(x, BLUR_KERNEL).to(device)
    masks = torch.stack([torch.from_numpy(labels == k) for k in range(K)]).to(device)

    vals = np.zeros((n_perms, K), dtype=np.float64)
    rng = np.random.RandomState(0)

    for p in range(n_perms):
        perm = rng.permutation(K)
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
                b_img = imgs[b:b + BATCH]
                if is_mobilevit:
                    # mobilevit expects [0,1] raw pixels in the timm wrapper
                    # (it applies its own standardisation inside)
                    b_img = (b_img * STD.to(device) + MEAN.to(device))
                logits.append(model(b_img)[:, target])
        logits = torch.cat(logits).double().cpu().numpy()
        vals[p, perm] = np.diff(logits)

    shap = vals.mean(0)
    mc_std = vals.std(0) / np.sqrt(n_perms)
    return shap, mc_std

def region_sums(pixel_map, labels):
    K = labels.max() + 1
    return np.array([pixel_map[labels == k].sum() for k in range(K)])

def main():
    n_images = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    n_perms = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = sys.argv[3] if len(sys.argv) > 3 else MODEL
    is_mobilevit = model_name.startswith("mobilevit")
    
    out = os.path.join(OUT, model_name)
    os.makedirs(out, exist_ok=True)
    
    data = torch.load(os.path.join("data", "ImageNetS", "cache_validation_100.pt"),
                      map_location="cpu", weights_only=False)

    print("Loading SAM (vit_b)...")
    sam_checkpoint = os.path.join("checkpoints", "sam_vit_b_01ec64.pth")
    if not os.path.exists(sam_checkpoint):
        print(f"Error: SAM checkpoint not found at {sam_checkpoint}")
        return
        
    sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    
    from xai_bench.models.timm_models import create_timm_model
    from xai_bench.registry import METHODS
    import captum.attr as ca

    wrapper = create_timm_model(model_name)
    wrapper.model.eval().to(device)
    vanilla = wrapper.model
    
    grad_cam = METHODS.get("grad_cam")(model_wrapper=wrapper, model=vanilla)
    ig = ca.IntegratedGradients(vanilla)
    sg = ca.NoiseTunnel(ca.Saliency(vanilla))

    def baseline_maps(x, target):
        maps = {}
        # if mobilevit, baseline methods evaluate on the normalized input, 
        # but the wrapper handles the denorm internally.
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

    per_image = []
    for idx in range(n_images):
        img = data[idx]["image"]
        
        # 1. Generate SAM segments
        labels = make_sam_regions(img, mask_generator)
        K = labels.max() + 1
        
        x = img.unsqueeze(0).to(device)
        with torch.no_grad():
            fwd_x = x
            if is_mobilevit:
                fwd_x = (x * STD.to(device) + MEAN.to(device))
            target = int(vanilla(fwd_x)[0].argmax())

        # 2. Compute Shapley on vanilla model
        shap, mc_std = shapley_sampled(vanilla, img, labels, target, n_perms, device, is_mobilevit)
        
        np.save(os.path.join(out, f"regions_{idx}.npy"), labels)
        np.save(os.path.join(out, f"shapley_{idx}.npy"), shap)
        np.save(os.path.join(out, f"mc_std_{idx}.npy"), mc_std)

        # 3. Compute baseline maps
        base_r = {name: region_sums(m, labels)
                  for name, m in baseline_maps(x, target).items()}
                  
        per_image.append((idx, labels, target, shap, mc_std, base_r))
        print(f"pass1 img{idx}: K={K} target {target}  SNR "
              f"{np.abs(shap).mean() / (mc_std.mean() + 1e-12):.1f}")

    # Free SAM and vanilla model memory before loading HiLRP
    del mask_generator, sam, vanilla, wrapper, grad_cam, ig, sg      
    torch.cuda.empty_cache()

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
    print(f"\n[{model_name}] mean Spearman vs SAM-Shapley (n={len(rows)}, m={n_perms} perms):")
    for j, n in enumerate(method_names):
        print(f"  {n:12s} {np.nanmean(arr[:, j]):+.3f}")
    print(f"written to {out}")

if __name__ == "__main__":
    main()
