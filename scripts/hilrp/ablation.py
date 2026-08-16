"""Ablation of HiLRP's relevance-rule design choices on Swin-B.

Swin exposes both toggles (attn_mode in {cp, attnlrp}, gamma), so we can isolate:
  Section A - attention rule (gamma fixed 0.25): AttnLRP-through vs CP-LRP (ours)
  Section B - gamma schedule (attn_mode fixed cp): 0.0 / 0.25 (ours) / 0.5

For each config we report, over N cached ImageNet-S images:
  Pointing  - localization (peak in GT bbox)
  ConsErr   - mean |sum(pixel relevance)/logit - 1|  (conservation validity)

Usage:  <mri-python> scripts/hilrp/ablation.py [N]
"""
import os
import sys
import glob
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

MODEL = "swin_base_patch4_window7_224"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
CONFIGS = [  # (label, section, attn_mode, gamma, is_ours)
    ("AttnLRP-through", "A: Attention rule (gamma=0.25)", "attnlrp", 0.25, False),
    ("CP-LRP (ours)",   "A: Attention rule (gamma=0.25)", "cp",      0.25, True),
    ("gamma = 0.0",     "B: gamma schedule (attn=CP)",    "cp",      0.0,  False),
    ("gamma = 0.25 (ours)", "B: gamma schedule (attn=CP)","cp",      0.25, True),
    ("gamma = 0.5",     "B: gamma schedule (attn=CP)",    "cp",      0.5,  False),
]


def load_cache():
    p = max(glob.glob("data/ImageNetS/cache_validation_*.pt"),
            key=lambda q: int(q.rsplit("_", 1)[-1].split(".")[0]))
    return torch.load(p, map_location="cpu", weights_only=False)


def pointing(pm, bboxes):
    h, w = pm.shape
    py, px = np.unravel_index(np.argmax(pm), pm.shape)
    return any(b[0]*w <= px <= b[2]*w and b[1]*h <= py <= b[3]*h for b in bboxes)


def main():
    import timm
    from xai_bench.methods.hilrp.swin_lxt import attribute_swin
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_cache()
    n = min(N, len(data))
    model = timm.create_model(MODEL, pretrained=True).eval().to(device)

    rows = []
    for label, section, mode, gamma, ours in CONFIGS:
        hits = np.zeros(n, dtype=bool)
        cerr = np.zeros(n)
        for i in range(n):
            x = data[i]["image"].unsqueeze(0).to(device)
            r = attribute_swin(model, x, gamma=gamma, attn_mode=mode)
            hits[i] = pointing(r["pixel_map"].numpy(), data[i]["metadata"]["bboxes"])
            cerr[i] = abs(r["stage_sums"][-1][1] - 1.0)   # |sum R / logit - 1|
        rows.append((section, label, ours, hits.mean(), cerr.mean()))
        line = f"[{section}] {label:22s}  Pointing={hits.mean():.3f}  ConsErr={cerr.mean():.2e}"
        print(line, flush=True)
        os.makedirs("results", exist_ok=True)
        with open("results/ablation_swin.txt", "a") as fh:
            fh.write(line + "\n")

    print("\n=== LaTeX-ready (n={}) ===".format(n))
    for section, label, ours, pg, ce in rows:
        star = "  <- ours" if ours else ""
        print(f"{section} | {label} | {pg:.3f} | {ce:.1e}{star}")


if __name__ == "__main__":
    main()
