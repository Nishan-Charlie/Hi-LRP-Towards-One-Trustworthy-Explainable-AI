"""Deep / head / pixel conservation for HiLRP, ONE backbone per process.

Class-level patches contaminate across *_lxt modules, so each backbone must run
in a fresh interpreter (same protocol note as clip_demo / attnlrp_method).
Drive with:  for m in ...; do python scripts/hilrp/conservation_grid.py $m 100; done

Reports sum(R)/logit at the input-adjacent capture (deep), the head-adjacent
capture, and the pixel level, mean +- std over n images.
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import transformers.pytorch_utils as _pu
if not hasattr(_pu, "find_pruneable_heads_and_indices"):
    def _fp(heads, n_heads, head_size, already):
        mask = torch.ones(n_heads, head_size)
        heads = set(heads) - already
        for h in heads:
            h = h - sum(1 for x in already if x < h)
            mask[h] = 0
        mask = mask.view(-1).contiguous().eq(1)
        return heads, torch.arange(len(mask))[mask].long()
    _pu.find_pruneable_heads_and_indices = _fp

import glob
import importlib
import numpy as np
import timm

LEGS = {
    "swin_base_patch4_window7_224": ("swin_lxt", "attribute_swin"),
    "pvt_v2_b2": ("pvt_lxt", "attribute_pvt"),
    "efficientvit_b1": ("efficientvit_lxt", "attribute_efficientvit"),
    "efficientvit_b2": ("efficientvit_lxt", "attribute_efficientvit"),
    "maxvit_small_tf_224": ("maxvit_lxt", "attribute_maxvit"),
    "mobilevitv2_100": ("mobilevit_lxt", "attribute_mobilevit"),
}


def main():
    model_name = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    mod_name, fn_name = LEGS[model_name]

    caches = glob.glob(os.path.join("data", "ImageNetS", "cache_validation_*.pt"))
    path = max(caches, key=lambda p: int(p.rsplit("_", 1)[-1].split(".")[0]))
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        data = torch.load(path, map_location="cpu", weights_only=False)  # trusted local cache
    n = min(n, len(data))

    mod = importlib.import_module(f"xai_bench.methods.hilrp.{mod_name}")
    if hasattr(mod, "ensure_patched"):
        mod.ensure_patched()
    attributor = getattr(mod, fn_name)
    model = timm.create_model(model_name, pretrained=True).eval().cuda()

    deep, head, pix = [], [], []
    for i in range(n):
        x = data[i]["image"].unsqueeze(0).cuda()
        res = attributor(model, x)
        ss = res["stage_sums"]
        deep.append(ss[0][1])
        head.append(ss[-1][1])
        logit = res.get("logit")
        if logit is not None and abs(logit) > 1e-12:
            pix.append(float(res["pixel_map"].sum()) / logit)
    names = [s[0] for s in res["stage_sums"]]
    print(f"{model_name}  n={n}  captures={names}")
    print(f"  deep({names[0]}): {np.mean(deep):+.3f} +- {np.std(deep):.3f}")
    print(f"  head({names[-1]}): {np.mean(head):+.3f} +- {np.std(head):.3f}")
    if pix:
        print(f"  pixel: {np.mean(pix):+.3f} +- {np.std(pix):.3f}")
    print("LEG_DONE", flush=True)


if __name__ == "__main__":
    main()
