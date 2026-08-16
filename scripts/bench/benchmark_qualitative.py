"""Qualitative 'breaking point' figure for the benchmark paper: Grad-CAM for a few
images/ images across the benchmark architectures, showing that CAM localizes well
on the CNN / hierarchical / hybrid backbones but diffuses on the linear-attention
EfficientViT variants (the visual analogue of the Pointing-Game heatmap).

  gen      generate Grad-CAM maps (base env with the registry)
  assemble build figures/bench/bench_qualitative.png/pdf

Runs the registry Grad-CAM (per-model target layers) so it matches the benchmark.
"""
import os
import sys
import glob
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

TMP = os.path.join("results", "qual_bench")
os.makedirs(TMP, exist_ok=True)
os.makedirs("figures", exist_ok=True)

MODELS = [  # (registry key, short label)
    ("resnet50", "ResNet-50"),
    ("swin_base_patch4_window7_224", "Swin-B"),
    ("maxvit_small_tf_224", "MaxViT-S"),
    ("mobilevitv2_100", "MobileViT"),
    ("efficientvit_b1", "EffViT-B1"),
    ("efficientvit_b2", "EffViT-B2"),
]
IMAGES = ["cheetah", "ambulance", "bus"]
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
SIZE = 224


def preprocess(pil):
    from torchvision import transforms
    t = transforms.Compose([transforms.Resize((SIZE, SIZE)), transforms.ToTensor(),
                            transforms.Normalize(mean=list(MEAN), std=list(STD))])
    return t(pil)


def _img_path(name):
    for p in glob.glob("images/*.*"):
        if os.path.splitext(os.path.basename(p))[0] == name:
            return p
    raise FileNotFoundError(name)


def gen():
    from PIL import Image
    from xai_bench.registry import MODELS as REG, METHODS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for mk, _ in MODELS:
        mw = REG.get(mk)(); mw.model.eval().to(device)
        gc = METHODS.get("grad_cam")(model_wrapper=mw, model=mw.model)
        for name in IMAGES:
            x = preprocess(Image.open(_img_path(name)).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                t = int(mw(x).argmax())
            m = gc(x, t)
            m = (m[0] if getattr(m, "ndim", 2) == 3 else m).detach().cpu().numpy()
            np.save(os.path.join(TMP, f"{name}__{mk}.npy"), m)
            rgb = np.clip(x[0].cpu().numpy().transpose(1, 2, 0) * STD + MEAN, 0, 1)
            np.save(os.path.join(TMP, f"{name}__input.npy"), rgb)
        print(f"gen {mk} done", flush=True)


def _heat(mp):
    from scipy.ndimage import gaussian_filter, zoom
    h = np.abs(np.asarray(mp, dtype=np.float64))
    if h.shape[0] != SIZE:
        h = zoom(h, SIZE / h.shape[0], order=1)
    h = gaussian_filter(h, 2.0)
    return np.clip(h / (np.percentile(h, 99) + 1e-12), 0, 1)


def assemble():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = ["input"] + [m[0] for m in MODELS]
    titles = ["Input"] + [m[1] for m in MODELS]
    fig, ax = plt.subplots(len(IMAGES), len(cols), figsize=(1.75 * len(cols), 1.75 * len(IMAGES)))
    ax = np.atleast_2d(ax)
    for r, name in enumerate(IMAGES):
        rgb = np.load(os.path.join(TMP, f"{name}__input.npy"))
        ax[r, 0].imshow(rgb); ax[r, 0].axis("off")
        for c, mk in enumerate(cols[1:], 1):
            gray = rgb.mean(2, keepdims=True).repeat(3, 2) * 0.55 + 0.15
            h = _heat(np.load(os.path.join(TMP, f"{name}__{mk}.npy")))
            ax[r, c].imshow(gray)
            ax[r, c].imshow(h, cmap="turbo", alpha=np.clip(h * 1.1, 0, 1))
            ax[r, c].axis("off")
        if r == 0:
            for c, t in enumerate(titles):
                ax[0, c].set_title(t, fontsize=10)
    fig.subplots_adjust(wspace=0.04, hspace=0.04)
    fig.savefig("figures/bench/bench_qualitative.pdf", bbox_inches="tight")
    fig.savefig("figures/bench/bench_qualitative.png", dpi=160, bbox_inches="tight")
    print("wrote figures/bench/bench_qualitative.pdf")


if __name__ == "__main__":
    {"gen": gen, "assemble": assemble}[sys.argv[1] if len(sys.argv) > 1 else "assemble"]()
