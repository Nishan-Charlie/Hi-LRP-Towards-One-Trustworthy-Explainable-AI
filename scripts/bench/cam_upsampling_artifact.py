"""Demonstrate that the square blocks in CAM maps on hierarchical backbones come
from the coarse terminal feature map (7x7) upsampled 32x to 224x224, and that the
artifact is shared by Grad-CAM and Grad-CAM++ (same terminal map, same upsample).

For Swin-B, PVT-v2, MaxViT-S (all 7x7 terminal) we render, per method, the native
low-resolution grid (nearest-neighbor, so the 7x7 cells are crisp) next to the
bilinear map actually used. The block edges coincide with the 7x7 cell boundaries
for both methods. Saves a vector PDF + PNG for the paper.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import xai_bench.methods, xai_bench.metrics, xai_bench.models, xai_bench.datasets  # noqa: F401
from xai_bench.registry import MODELS, DATASETS
from xai_bench.methods.cam_methods import _to_nchw

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = np.array([0.485, 0.456, 0.406]); STD = np.array([0.229, 0.224, 0.225])
MODELS_ROWS = [
    ("swin_base_patch4_window7_224", "Swin-B"),
    ("pvt_v2_b2", "PVT-v2"),
    ("maxvit_small_tf_224", "MaxViT-S"),
]


def denorm(t):
    img = t.detach().cpu().numpy().transpose(1, 2, 0) * STD + MEAN
    return np.clip(img, 0, 1)


def cam_lowres(mw, x, target, plus_plus):
    """Return the raw CAM at the terminal grid resolution (before upsampling)."""
    model = mw.model
    fmt = mw.cam_format
    x = x.clone().detach().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    feat = model.forward_features(x)
    feat.retain_grad()
    out = model.forward_head(feat)
    out[0, int(target)].backward()
    A = _to_nchw(feat, fmt)
    G = _to_nchw(feat.grad, fmt)
    if not plus_plus:
        w = G.mean(dim=(2, 3), keepdim=True)
    else:
        g2, g3 = G ** 2, G ** 3
        sa = A.sum(dim=(2, 3), keepdim=True)
        al = g2 / (2 * g2 + sa * g3 + 1e-7)
        al = torch.where(G != 0.0, al, torch.zeros_like(al))
        w = (al * F.relu(G)).sum(dim=(2, 3), keepdim=True)
    cam = F.relu((w * A).sum(dim=1, keepdim=True))
    return cam[0, 0].detach().cpu().numpy()


def norm01(a):
    a = a - a.min()
    m = a.max()
    return a / m if m > 0 else a


def upsample(cam, mode):
    t = torch.tensor(cam)[None, None].float()
    if mode == "nearest":
        up = F.interpolate(t, size=(224, 224), mode="nearest")
    else:
        up = F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)
    return norm01(up[0, 0].numpy())


def pick_image(ds, n=60):
    best, area = 0, -1
    for i in range(min(n, len(ds))):
        _, _, meta = ds[i]
        bb = meta.get("bboxes")
        if not bb:
            continue
        x0, y0, x1, y1 = bb[0]
        a = (x1 - x0) * (y1 - y0)
        if 0.2 < a < 0.9 and a > area:
            area, best = a, i
    return best


def overlay(ax, img, heat, title, grid=False):
    ax.imshow(img)
    ax.imshow(heat, cmap="jet", alpha=0.55, vmin=0, vmax=1)
    if grid:
        for k in range(1, 7):
            ax.axhline(k * 224 / 7, color="w", lw=0.4, alpha=0.6)
            ax.axvline(k * 224 / 7, color="w", lw=0.4, alpha=0.6)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    ds = DATASETS.get("imagenets_cached")(
        cache_path="data/ImageNetS/cache_validation_1000.pt", num_samples=60)
    idx = pick_image(ds)
    img_t, target, _ = ds[idx]
    img = denorm(img_t)
    print(f"using image idx {idx}, target {target}")

    cols = ["Input", "Grad-CAM\n(7$\\times$7 grid)", "Grad-CAM\n(bilinear)",
            "Grad-CAM++\n(7$\\times$7 grid)", "Grad-CAM++\n(bilinear)"]
    fig, axes = plt.subplots(len(MODELS_ROWS), 5, figsize=(11.5, 6.9))

    for r, (mname, disp) in enumerate(MODELS_ROWS):
        mw = MODELS.get(mname)()
        mw.model.to(DEVICE).eval()
        x = img_t.unsqueeze(0).to(DEVICE)
        gc = cam_lowres(mw, x, target, plus_plus=False)
        gpp = cam_lowres(mw, x, target, plus_plus=True)
        hres = gc.shape[0]
        maps = [None,
                (upsample(gc, "nearest"), True),
                (upsample(gc, "bilinear"), False),
                (upsample(gpp, "nearest"), True),
                (upsample(gpp, "bilinear"), False)]
        for c in range(5):
            ax = axes[r, c]
            if c == 0:
                ax.imshow(img)
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(cols[0], fontsize=10)
                ax.set_ylabel(f"{disp}\n(terminal {hres}$\\times${hres})",
                              fontsize=10, rotation=90, labelpad=6)
            else:
                heat, grid = maps[c]
                overlay(ax, img, heat, cols[c] if r == 0 else "", grid=grid)
        del mw
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fig.suptitle("CAM square artifacts on hierarchical backbones originate from the "
                 "7$\\times$7 terminal grid upsampled 32$\\times$; both CAM variants share it",
                 fontsize=11, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("pdf", "png"):
        fig.savefig(f"figures/bench/cam_upsampling_artifact.{ext}", dpi=160, bbox_inches="tight")
    print("saved figures/bench/cam_upsampling_artifact.{pdf,png}")


if __name__ == "__main__":
    main()
