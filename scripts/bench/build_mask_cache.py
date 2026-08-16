"""Build a dense-mask cache aligned to data/ImageNetS/cache_validation_1000.pt.

Streams the same braceletboy/imagenet-s validation split (same order as the image
cache), resizes each mask to 224x224 (nearest), stores the boolean foreground mask
(mask>0, the object silhouette) plus the label for alignment checking. Saves
incrementally every 50 items so a slow/flaky stream still leaves usable progress,
and is restart-safe (skips already-cached indices).
"""
import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

OUT = "data/ImageNetS/cache_masks_1000.pt"
N = 1000
resize = transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST)


def main():
    records = {}
    if os.path.exists(OUT):
        records = torch.load(OUT, weights_only=True)
        print(f"resuming: {len(records)} masks already cached")
    if len(records) >= N:
        print("already complete")
        return

    from datasets import load_dataset
    ds = load_dataset("braceletboy/imagenet-s", split="validation", streaming=True)
    print("stream opened, iterating...", flush=True)

    for i, item in enumerate(ds):
        if i >= N:
            break
        if i in records:
            continue
        m = item.get("mask")
        if m is None:
            records[i] = dict(mask=None, label=item.get("label"))
        else:
            marr = np.array(resize(m))
            fg = (marr > 0).astype(np.uint8)   # object silhouette
            records[i] = dict(mask=torch.from_numpy(fg), label=item.get("label"),
                              area=float(fg.mean()))
        if (i + 1) % 50 == 0:
            torch.save(records, OUT)
            print(f"  cached {i+1}/{N} (last area={records[i].get('area')})", flush=True)

    torch.save(records, OUT)
    print(f"DONE: {len(records)} masks saved to {OUT}", flush=True)


if __name__ == "__main__":
    main()
