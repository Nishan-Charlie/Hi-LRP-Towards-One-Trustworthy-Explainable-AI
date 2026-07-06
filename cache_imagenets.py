#!/usr/bin/env python
"""Materialise N ImageNet-S validation samples to a local .pt cache.

HuggingFace streaming is flaky over long runs; caching once up front makes the
benchmark network-independent and restart-safe. Each record stores the
pre-transformed image tensor, label, and bbox metadata (derived from the mask),
exactly what the dataset would otherwise return per __getitem__.

Usage:  python cache_imagenets.py --num 100 --split validation
"""
import argparse
import os
import time
import ssl
import certifi

def _custom_create_default_context(*args, **kwargs):
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.load_verify_locations(cafile=certifi.where())
    return ctx

ssl.create_default_context = _custom_create_default_context

import numpy as np
import torch
from datasets import load_dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
MASK_RESIZE = transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST)


def bboxes_from_mask(mask_pil):
    m = np.array(MASK_RESIZE(mask_pil))
    rows = np.any(m > 0, axis=1)
    cols = np.any(m > 0, axis=0)
    if not np.any(rows):
        return []
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return [[float(xmin) / 224.0, float(ymin) / 224.0,
             float(xmax) / 224.0, float(ymax) / 224.0]]


def build(num_samples, split, out_path, max_retries=8):
    records = []
    if os.path.exists(out_path):
        records = torch.load(out_path, weights_only=False)
        print(f"Resuming: {len(records)} samples already cached.")

    attempt = 0
    while len(records) < num_samples and attempt < max_retries:
        attempt += 1
        try:
            print(f"[attempt {attempt}] loading {split}, have {len(records)}/{num_samples}...")
            ds = load_dataset("braceletboy/imagenet-s", split=split, streaming=True)
            for i, item in enumerate(ds):
                if i < len(records):
                    continue  # already have it
                if len(records) >= num_samples:
                    break
                img = item["image"]
                if img.mode != "RGB":
                    img = img.convert("RGB")
                meta = {"img_name": f"sample_{i}.jpg", "bboxes": []}
                if "mask" in item and item["mask"] is not None:
                    meta["bboxes"] = bboxes_from_mask(item["mask"])
                records.append({
                    "image": TRANSFORM(img),
                    "label": int(item["label"]),
                    "metadata": meta,
                })
                if len(records) % 10 == 0:
                    torch.save(records, out_path)
                    print(f"  cached {len(records)}/{num_samples}")
            torch.save(records, out_path)
        except Exception as e:
            torch.save(records, out_path)
            wait = min(2 ** attempt, 30)
            print(f"  stream error: {e}\n  saved {len(records)}, retrying in {wait}s...")
            time.sleep(wait)

    torch.save(records, out_path)
    n_with_box = sum(1 for r in records if r["metadata"]["bboxes"])
    print(f"Done: {len(records)} samples -> {out_path} ({n_with_box} with bboxes)")
    return len(records)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=100)
    ap.add_argument("--split", type=str, default="validation")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    out = args.out or f"data/ImageNetS/cache_{args.split}_{args.num}.pt"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    build(args.num, args.split, out)
