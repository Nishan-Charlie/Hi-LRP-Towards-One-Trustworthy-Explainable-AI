"""Lineage comparison: LRP / AttnLRP / Chefer-proxy vs HiLRP.

Leg A, plain ViT (vit_base_patch16_224, AttnLRP's home turf):
  - HiLRP-CP (ours, default attention rule)
  - AttnLRP proper (softmax Taylor + uniform bilinear, via attn_mode flag;
    on a flat ViT this IS the LXT/Achtibat method, which we inherit)
  - bench baselines from the existing table: attention_rollout,
    attention_gradient (Chefer-proxy), grad_cam, saliency, IG
  - captum LRP: documented failure on timm models (0 rows in the bench)

Leg B, Swin (swin_base, our home turf):
  - HiLRP (hierarchical rules + guards)
  - "AttnLRP-naive": LXT's flat-ViT recipe applied to Swin as-is, i.e. LN/GELU
    identity patches and Gamma composites but NO window-attention rule (vanilla
    softmax gradients through W-MSA/SW-MSA). What a user without hierarchical
    rules would get today.

Protocol: pointing game, same argmax-in-bbox rule as gate3, cached ImageNet-S.

Run:  <mri-diffuser python> scripts/hilrp/lineage_comparison.py
"""
import os
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import glob
import timm

N_IMAGES = 1000
OUT = os.path.join("results", "hilrp_lineage")
os.makedirs(OUT, exist_ok=True)


def load_cache():
    """Load the largest available cache_validation_*.pt (so N_IMAGES scales as
    the cache grows) and return (data, n) with n clamped to what exists."""
    global N_IMAGES
    caches = glob.glob(os.path.join("data", "ImageNetS", "cache_validation_*.pt"))
    if not caches:
        raise FileNotFoundError("no ImageNet-S cache; run scripts/data/cache_imagenets.py --num 500")
    path = max(caches, key=lambda p: int(p.rsplit("_", 1)[-1].split(".")[0]))
    data = torch.load(path, map_location="cpu", weights_only=False)
    N_IMAGES = min(N_IMAGES, len(data))
    print(f"cache: {path} ({len(data)} imgs), using N_IMAGES={N_IMAGES}")
    return data


def pointing(pm, bboxes):
    h, w = pm.shape
    py, px = np.unravel_index(np.argmax(pm), pm.shape)
    return any(bb[0] * w <= px <= bb[2] * w and bb[1] * h <= py <= bb[3] * h
               for bb in bboxes)


def run_leg(tag, attributor, model, data, **kw):
    """Pointing + conservation at BOTH ends: the deepest capture (first entry,
    e.g. tokens_56 for Swin, pixels for flat ViT) discriminates rule quality;
    the head-adjacent capture (last entry) is enforced by the head hook and
    stays ~1 for everyone."""
    hits, cons_deep, cons_head = 0, [], []
    hit_array = np.zeros(N_IMAGES, dtype=bool)
    for idx in range(N_IMAGES):
        x = data[idx]["image"].unsqueeze(0).cuda()
        res = attributor(model, x, **kw)
        hit_bool = pointing(res["pixel_map"].numpy(), data[idx]["metadata"]["bboxes"])
        hits += hit_bool
        hit_array[idx] = hit_bool
        cons_deep.append(res["stage_sums"][0][1])
        cons_head.append(res["stage_sums"][-1][1])
    print(f"  {tag:32s} pointing {hits}/{N_IMAGES} = {hits / N_IMAGES:.3f}   "
          f"cons deep {np.mean(cons_deep):+.2f} +- {np.std(cons_deep):.2f}   "
          f"head {np.mean(cons_head):+.2f}")
    
    # Save the hit array
    safe_tag = tag.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_").lower()
    model_name = getattr(model, "default_cfg", {}).get("architecture", "unknown_model")
    np.save(os.path.join(OUT, f"hits_{model_name}_{safe_tag}.npy"), hit_array)
    
    return hits / N_IMAGES


def main():
    data = load_cache()

    # ---------------- Leg A: plain ViT ----------------
    print("=== Leg A: vit_base_patch16_224 (AttnLRP's home turf) ===")
    from xai_bench.methods.hilrp.vit_lxt import attribute_vit, ensure_patched as vit_patch
    vit_patch()
    vit = timm.create_model("vit_base_patch16_224", pretrained=True).eval().cuda()
    run_leg("HiLRP-CP (ours)", attribute_vit, vit, data, attn_mode="cp")
    run_leg("AttnLRP (inherited rules)", attribute_vit, vit, data, attn_mode="attnlrp")
    print("  bench baselines (100 imgs, same protocol family):")
    print("    grad_cam 0.927 | attention_gradient (Chefer-proxy) 0.740 | "
          "attention_rollout 0.640 | saliency 0.650 | IG 0.640")
    print("    captum LRP: FAILS on timm transformers (unsupported Identity "
          "layers, 0 rows in the bench)")

    # ---------------- Leg B: Swin ----------------
    print("\n=== Leg B: swin_base_patch4_window7_224 (hierarchical, our turf) ===")
    from xai_bench.methods.hilrp.swin_lxt import attribute_swin, ensure_patched as swin_patch
    swin_patch()
    swin = timm.create_model("swin_base_patch4_window7_224", pretrained=True).eval().cuda()
    run_leg("HiLRP (ours)", attribute_swin, swin, data)

    # AttnLRP-naive: LN/GELU identity patches and the Gamma composite stay
    # (LXT's generic flat-transformer recipe), but the window attention runs
    # timm's ORIGINAL explicit path: vanilla softmax gradients, no CP detach,
    # no uniform rule. Implemented as an explicit vanilla forward and swapped
    # in/out on the class (no importlib.reload class confusion).
    from timm.models.swin_transformer import WindowAttention

    def vanilla_window_attention_forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn + self._get_rel_pos_bias()
        if mask is not None:
            num_win = mask.shape[0]
            attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = attn @ v                                 # vanilla: gradient flows everywhere
        x = x.transpose(1, 2).reshape(B_, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    hilrp_fwd = WindowAttention.forward
    WindowAttention.forward = vanilla_window_attention_forward
    try:
        run_leg("AttnLRP-naive (no window rule)", attribute_swin, swin, data)
    finally:
        WindowAttention.forward = hilrp_fwd


if __name__ == "__main__":
    main()
