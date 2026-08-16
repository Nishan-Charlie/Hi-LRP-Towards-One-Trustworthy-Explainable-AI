"""Multi-modal HiLRP demo: text-conditioned attribution on CLIP (ViT-B/16).

Proves HiLRP performs conservation-based attribution on a real multi-modal model
with a cross-modal scalar (image-text similarity), and that the explanation is
text-conditioned: a caption that matches the image concentrates more evidence on
the object than a mismatched caption. This is the multi-attention / multi-modal
leg of the "any attention" claim (the Chefer ICCV'21 differentiator), reusing the
label-free SSL-scalar machinery.

Run in its own process (class-level patches):  <mri-diffuser python> scripts/hilrp/clip_demo.py
"""
import os
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import open_clip
from xai_bench.methods.hilrp.clip_lxt import attribute_clip_image_text, ensure_patched
from xai_bench.methods.hilrp.viz import normalize_for_display

OUT = "figures"
os.makedirs(OUT, exist_ok=True)
IM = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IS = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
CM = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
CS = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
CORRECT, WRONG = "a photo of a yurt", "a photo of a dog"


def energy_in_box(pm, bb):
    h, w = pm.shape
    m = np.zeros((h, w), bool)
    for b in bb:
        m[int(b[1] * h):int(b[3] * h) + 1, int(b[0] * w):int(b[2] * w) + 1] = True
    pos = np.maximum(pm, 0)
    return pos[m].sum() / (pos.sum() + 1e-12)


def main():
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    model.eval()
    tok = open_clip.get_tokenizer("ViT-B-16")
    ensure_patched()
    data = torch.load("data/ImageNetS/cache_validation_100.pt", map_location="cpu", weights_only=False)

    with torch.no_grad():
        te = model.encode_text(tok([CORRECT, WRONG]))
        te = te / te.norm(dim=-1, keepdim=True)

    yurts = [i for i in range(len(data)) if data[i]["label"] == 915][:6]
    ec, ew = [], []
    rows = []
    for i in yurts:
        raw = (data[i]["image"] * IS + IM).clamp(0, 1)
        x = ((raw - CM) / CS).unsqueeze(0)
        rc = attribute_clip_image_text(model, x, te[0].detach())
        rw = attribute_clip_image_text(model, x, te[1].detach())
        ec.append(energy_in_box(rc["pixel_map"].numpy(), data[i]["metadata"]["bboxes"]))
        ew.append(energy_in_box(rw["pixel_map"].numpy(), data[i]["metadata"]["bboxes"]))
        rows.append((raw.permute(1, 2, 0).numpy(), rc, rw))

    print(f"CLIP text-conditioned attribution over {len(yurts)} images:")
    print(f"  '{CORRECT}':  energy-in-object {np.mean(ec):.3f}")
    print(f"  '{WRONG}':    energy-in-object {np.mean(ew):.3f}")
    print(f"  correct > wrong on {sum(c > w for c, w in zip(ec, ew))}/{len(yurts)}")

    n = min(4, len(rows))
    fig, ax = plt.subplots(n, 3, figsize=(6.6, 2.2 * n))
    for r in range(n):
        img, rc, rw = rows[r]
        ax[r, 0].imshow(img); ax[r, 0].axis("off")
        if r == 0:
            ax[r, 0].set_title("input", fontsize=9)
        for c, (res, cap) in enumerate([(rc, CORRECT), (rw, WRONG)], start=1):
            m, v = normalize_for_display(res["pixel_map"], percentile=99, smooth_sigma=1.0)
            ax[r, c].imshow(m, cmap="bwr", vmin=-v, vmax=v); ax[r, c].axis("off")
            if r == 0:
                ax[r, c].set_title(f'"{cap}"\nsim={res["similarity"]:.2f}', fontsize=8)
    fig.suptitle("Multi-modal HiLRP: text-conditioned attribution on CLIP", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "clip_multimodal.png"), dpi=150)
    print(f"figure -> {OUT}/clip_multimodal.png")


if __name__ == "__main__":
    main()
