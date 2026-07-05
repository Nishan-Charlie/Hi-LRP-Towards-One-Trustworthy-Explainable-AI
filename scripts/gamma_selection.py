"""Gamma-selection ablation for HiLRP (paper Table).

The gamma-rule on Linear/Conv2d controls the trade-off between conservation
(gamma -> 0 recovers the pure epsilon rule, exact but noisy) and denoising
(larger gamma emphasizes positive evidence, smoother but more biased). We sweep
gamma and report Pointing (localization) and the head-adjacent conservation
ratio (sum(R)/logit) on three clean-conservation backbones, to justify a single
default and to quantify sensitivity.

Run:  <mri-diffuser python> scripts/gamma_selection.py [n_images]
"""
import os, sys, warnings
import numpy as np, torch
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import timm

GAMMAS = [0.1, 0.25, 0.5, 1.0]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def pointing(pm, bb):
    h, w = pm.shape; py, px = np.unravel_index(np.argmax(pm), pm.shape)
    return any(b[0] * w <= px <= b[2] * w and b[1] * h <= py <= b[3] * h for b in bb)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    data = torch.load(os.path.join("data", "ImageNetS", "cache_validation_100.pt"),
                      map_location="cpu", weights_only=False)
    from xai_bench.methods.hilrp.swin_lxt import attribute_swin, ensure_patched as sp
    from xai_bench.methods.hilrp.pvt_lxt import attribute_pvt, ensure_patched as pp
    from xai_bench.methods.hilrp.efficientvit_lxt import attribute_efficientvit, ensure_patched as ep
    sp(); pp(); ep()
    models = [
        ("Swin-B", timm.create_model("swin_base_patch4_window7_224", pretrained=True).eval(), attribute_swin, False),
        ("PVT-v2", timm.create_model("pvt_v2_b2", pretrained=True).eval(), attribute_pvt, False),
        ("EffViT-B2", timm.create_model("efficientvit_b2", pretrained=True).eval(), attribute_efficientvit, False),
    ]
    print(f"Gamma selection ({n} images). Cells: Pointing / head-conservation.\n")
    print(f"{'model':10s} " + " ".join(f"g={g:<9g}" for g in GAMMAS))
    for name, model, attr, raw in models:
        cells = []
        for g in GAMMAS:
            hits, cons = 0, []
            for i in range(n):
                x = data[i]["image"].unsqueeze(0)
                if raw:
                    x = (x[0] * STD + MEAN).clamp(0, 1).unsqueeze(0)
                r = attr(model, x, gamma=g)
                hits += pointing(r["pixel_map"].numpy(), data[i]["metadata"]["bboxes"])
                cons.append(r["stage_sums"][-1][1])
            cells.append(f"{hits/n:.2f}/{np.mean(cons):.2f}")
        print(f"{name:10s} " + " ".join(f"{c:<11s}" for c in cells))


if __name__ == "__main__":
    main()
