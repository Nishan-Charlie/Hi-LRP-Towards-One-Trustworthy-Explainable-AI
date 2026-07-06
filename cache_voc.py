#!/usr/bin/env python
"""Materialise N Pascal VOC 2007 test images to a local .pt cache, in the SAME
record schema as cache_imagenets.py, so scaled_eval.py can run the Pointing Game
on a second, independent dataset.

VOC is the right second dataset: its objects are frequently off-center and
multi-instance (mean object area far below ImageNet-S), so it directly tests
whether the localization gap survives away from ImageNet-S's center-point prior.
We store every object's bounding box (native VOC annotation, not mask-derived),
normalized to [0,1] by the original image size.

Each record: {"image": (3,224,224) ImageNet-normalized tensor,
              "label": -1 (VOC classes are not ImageNet-1k; unused by Pointing),
              "metadata": {"img_name": str, "bboxes": [[x0,y0,x1,y1], ...]}}

Usage:  python cache_voc.py --num 1000
"""
import argparse
import os

import torch
from torchvision import transforms
from torchvision.datasets import VOCDetection

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def boxes_from_target(target):
    ann = target["annotation"]
    W = float(ann["size"]["width"])
    H = float(ann["size"]["height"])
    objs = ann["object"]
    if isinstance(objs, dict):
        objs = [objs]
    out = []
    for o in objs:
        b = o["bndbox"]
        x0 = float(b["xmin"]) / W
        y0 = float(b["ymin"]) / H
        x1 = float(b["xmax"]) / W
        y1 = float(b["ymax"]) / H
        out.append([x0, y0, x1, y1])
    return out


def build(num_samples, out_path, root):
    os.makedirs(root, exist_ok=True)
    ds = VOCDetection(root=root, year="2007", image_set="test", download=True)
    n = min(num_samples, len(ds))
    records = []
    for i in range(n):
        img, target = ds[i]
        if img.mode != "RGB":
            img = img.convert("RGB")
        records.append({
            "image": TRANSFORM(img),
            "label": -1,
            "metadata": {"img_name": target["annotation"]["filename"],
                         "bboxes": boxes_from_target(target)},
        })
        if (i + 1) % 100 == 0:
            print(f"  cached {i+1}/{n}", flush=True)
    torch.save(records, out_path)
    n_box = sum(1 for r in records if r["metadata"]["bboxes"])
    print(f"Done: {len(records)} samples -> {out_path} ({n_box} with bboxes)")
    return len(records)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=1000)
    ap.add_argument("--root", type=str, default="data/VOC")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    out = args.out or f"data/VOC/cache_voc_{args.num}.pt"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    build(args.num, out, args.root)
