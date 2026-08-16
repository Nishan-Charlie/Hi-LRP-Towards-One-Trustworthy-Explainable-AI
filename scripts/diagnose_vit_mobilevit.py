"""Diagnose and improve HiLRP on the two weak backbones (ViT-B, MobileViT-v2).

ViT-B:      CP-LRP detaches the softmax, discarding the CLS->patch attention where a
            flat ViT's localization lives. Compare attn_mode cp vs attnlrp.
MobileViT:  GroupNorm1's live global mean smears relevance across the background.
            Compare group-norm mean live vs detached (mobilevit_lxt.DETACH_MEAN).

Both toggles are read at call time, so we A/B on the SAME images in one process.
Reports mean Pointing and mean |pixel conservation - 1| for each variant.

Usage:  <mri-python> scripts/diagnose_vit_mobilevit.py [N]
"""
import os
import sys
import glob
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
IM = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IS = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_cache():
    p = max(glob.glob("data/ImageNetS/cache_validation_*.pt"),
            key=lambda q: int(q.rsplit("_", 1)[-1].split(".")[0]))
    return torch.load(p, map_location="cpu", weights_only=False)


def pointing(pm, bboxes):
    h, w = pm.shape
    py, px = np.unravel_index(np.argmax(pm), pm.shape)
    return any(b[0] * w <= px <= b[2] * w and b[1] * h <= py <= b[3] * h for b in bboxes)


def summarize(tag, hits, cerr):
    print(f"  {tag:28s} Pointing={np.mean(hits):.3f}  |cons-1|={np.mean(cerr):.3f}", flush=True)


def run_vit(data, n, device):
    import timm
    from xai_bench.methods.hilrp.vit_lxt import attribute_vit
    model = timm.create_model("vit_base_patch16_224", pretrained=True).eval().to(device)
    res = {"cp": ([], []), "attnlrp": ([], [])}
    for i in range(n):
        x = data[i]["image"].unsqueeze(0).to(device)
        bb = data[i]["metadata"]["bboxes"]
        for mode in ("cp", "attnlrp"):
            r = attribute_vit(model, x, gamma=0.25, attn_mode=mode)
            res[mode][0].append(pointing(r["pixel_map"].numpy(), bb))
            res[mode][1].append(abs(r["stage_sums"][-1][1] - 1.0))
    print("== ViT-B/16 (attention rule) ==")
    summarize("CP-LRP (current)", *res["cp"])
    summarize("AttnLRP mode", *res["attnlrp"])


def run_mobilevit(data, n, device):
    import timm
    import xai_bench.methods.hilrp.mobilevit_lxt as mv
    model = timm.create_model("mobilevitv2_100", pretrained=True).eval().to(device)
    cfg = model.pretrained_cfg
    need_raw = any(abs(a - b) > 1e-6 for a, b in zip(cfg["mean"], (0.485, 0.456, 0.406)))
    mean = torch.tensor(cfg["mean"]).view(3, 1, 1); std = torch.tensor(cfg["std"]).view(3, 1, 1)
    res = {False: ([], []), True: ([], [])}
    for i in range(n):
        img = data[i]["image"]
        if need_raw:
            img = ((img * IS + IM).clamp(0, 1) - mean) / std
        x = img.unsqueeze(0).to(device)
        bb = data[i]["metadata"]["bboxes"]
        for detach in (False, True):
            mv.DETACH_MEAN = detach
            r = mv.attribute_mobilevit(model, x, gamma=0.25)
            pm = r["pixel_map"]
            cons = pm.sum().item() / (r["logit"] if abs(r["logit"]) > 1e-12 else 1.0)
            res[detach][0].append(pointing(pm.numpy(), bb))
            res[detach][1].append(abs(cons - 1.0))
    mv.DETACH_MEAN = False
    print("== MobileViT-v2 (GroupNorm1 mean) ==")
    summarize("mean live (current)", *res[False])
    summarize("mean detached", *res[True])


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_cache()
    n = min(N, len(data))
    print(f"n={n}, device={device}")
    run_vit(data, n, device)
    run_mobilevit(data, n, device)
