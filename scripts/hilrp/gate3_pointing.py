"""Gate 3: pointing game for HiLRP vs the existing baseline table.

Win condition (CLAUDE.md): pointing game on the hybrid/linear-attention models
where Grad-CAM collapsed (mobilevitv2_100: 0.49), not deletion AUC. swin_base is
the parity leg (Grad-CAM is saturated at 1.0 there).

Protocol: same rule as the bench's pointing_game, the attribution argmax must
fall inside a ground-truth box. Runs on the same cached ImageNet-S images the
baseline table used. Writes results/hilrp_gate3/<model>.csv and prints the
comparison rows.

Run:  <mri-diffuser python> scripts/hilrp/gate3_pointing.py [model ...]
      models: swin_base_patch4_window7_224 (default), mobilevitv2_100
"""
import os
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import timm

OUT = os.path.join("results", "hilrp_gate3")
os.makedirs(OUT, exist_ok=True)
GAMMA = 0.25
N_IMAGES = 100

BASELINE_CSV = os.path.join("results", "combined_summary.csv")

# the cache is ImageNet-normalized; some models (mobilevit family) expect raw
# [0,1] inputs. NOTE: the bench itself hardcoded ImageNet normalization for all
# models (xai_bench/datasets/imagenet_s.py), so its mobilevitv2 baseline rows ran
# on out-of-distribution inputs (top-1 was ~0). Those rows need a re-run before
# the baseline comparison there is fair.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def adapt_input(img, model):
    """Convert an ImageNet-normalized cached tensor to the model's expected stats."""
    mean = torch.tensor(model.default_cfg["mean"]).view(3, 1, 1)
    std = torch.tensor(model.default_cfg["std"]).view(3, 1, 1)
    raw = (img * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)
    return (raw - mean) / std


def get_attributor(model_name):
    if model_name.startswith("swin"):
        from xai_bench.methods.hilrp.swin_lxt import attribute_swin
        return attribute_swin
    if model_name.startswith("mobilevit"):
        from xai_bench.methods.hilrp.mobilevit_lxt import attribute_mobilevit
        return attribute_mobilevit
    if model_name.startswith("efficientvit"):
        from xai_bench.methods.hilrp.efficientvit_lxt import attribute_efficientvit
        return attribute_efficientvit
    if model_name.startswith("pvt"):
        from xai_bench.methods.hilrp.pvt_lxt import attribute_pvt
        return attribute_pvt
    raise ValueError(f"no HiLRP attributor for {model_name}")


def pointing_hit(pixel_map, bboxes):
    pm = pixel_map.numpy()
    py, px = np.unravel_index(np.argmax(pm), pm.shape)
    h, w = pm.shape
    return any(bb[0] * w <= px <= bb[2] * w and bb[1] * h <= py <= bb[3] * h
               for bb in bboxes)


def baseline_rows(model_name):
    if not os.path.exists(BASELINE_CSV):
        return []
    rows = []
    with open(BASELINE_CSV) as f:
        header = f.readline().strip().split(",")
        pg_i = header.index("pointing_game_mean")
        for line in f:
            parts = line.strip().split(",")
            if parts[0] == model_name:
                rows.append((parts[1], float(parts[pg_i])))
    return sorted(rows, key=lambda r: -r[1])


def main():
    models = sys.argv[1:] or ["swin_base_patch4_window7_224"]
    data = torch.load(os.path.join("data", "ImageNetS", "cache_validation_100.pt"),
                      map_location="cpu", weights_only=False)

    for model_name in models:
        print(f"\n=== {model_name} ===")
        model = timm.create_model(model_name, pretrained=True).eval()
        attributor = get_attributor(model_name)

        hits, rows = 0, []
        cons_last = []
        for idx in range(min(N_IMAGES, len(data))):
            item = data[idx]
            x = adapt_input(item["image"], model).unsqueeze(0)
            res = attributor(model, x, gamma=GAMMA)
            hit = pointing_hit(res["pixel_map"], item["metadata"]["bboxes"])
            hits += hit
            rows.append((idx, item["label"], res["target"], res["logit"], int(hit)))
            cons_last.append(res["stage_sums"][-1][1])
            if (idx + 1) % 25 == 0:
                print(f"  {idx + 1} imgs, running pointing = {hits / (idx + 1):.3f}")

        n = len(rows)
        pg = hits / n
        with open(os.path.join(OUT, f"{model_name}.csv"), "w") as f:
            f.write("idx,label,pred,logit,pointing_hit\n")
            for r in rows:
                f.write(",".join(str(v) for v in r) + "\n")

        print(f"HiLRP (gamma={GAMMA}, CP) pointing game: {hits}/{n} = {pg:.3f}")
        print(f"conservation at last capture: {np.mean(cons_last):+.3f} +- {np.std(cons_last):.3f}")
        print("baselines (same protocol family, from the bench):")
        for meth, v in baseline_rows(model_name):
            marker = " <-- HiLRP beats" if pg > v else ""
            print(f"  {meth:22s} {v:.3f}{marker}")


if __name__ == "__main__":
    main()
