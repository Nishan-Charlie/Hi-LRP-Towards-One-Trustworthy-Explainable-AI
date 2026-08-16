"""Build a publication-quality qualitative heatmap gallery from the images/ folder.

Diverse real images (rows) x attribution methods (columns), shown as clean
heatmap OVERLAYS on the (desaturated) input rather than raw signed maps, in the
style of a TIP/CVPR qualitative figure. HiLRP traces object structure; CAM
baselines blob; gradient baselines scatter.

Run in THREE processes (HiLRP's class-level patches would perturb the baselines'
gradients if run in-process):

  <base-python> scripts/bench/heatmap_gallery.py baselines   # vanilla model
  <mri-python>  scripts/bench/heatmap_gallery.py hilrp        # patched model
  <base-python> scripts/bench/heatmap_gallery.py contact      # contact sheet (curation)
  <base-python> scripts/bench/heatmap_gallery.py assemble a,b,c,d,e   # final figure, chosen images

Model is PVT-v2-b2 (strongest localization); override with QUAL_MODEL.
"""
import os
import sys
import glob
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

MODEL = os.environ.get("QUAL_MODEL", "pvt_v2_b2")
TMP = os.path.join("results", "qual_gallery")
os.makedirs(TMP, exist_ok=True)
os.makedirs("figures", exist_ok=True)

BASELINES = ["grad_cam", "grad_cam_plus_plus", "integrated_gradients"]
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
SIZE = 224


def list_images():
    paths = sorted(glob.glob("images/*.*"))
    return [p for p in paths if p.lower().rsplit(".", 1)[-1] in ("jpg", "jpeg", "png")]


def preprocess(pil):
    from torchvision import transforms
    t = transforms.Compose([
        transforms.Resize((SIZE, SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(MEAN), std=list(STD)),
    ])
    return t(pil)


def _map2d(m):
    m = m[0] if getattr(m, "ndim", 2) == 3 else m
    return m.detach().cpu().numpy() if torch.is_tensor(m) else np.asarray(m)


def run_baselines():
    from PIL import Image
    from xai_bench.registry import MODELS as REG, METHODS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mw = REG.get(MODEL)(); mw.model.eval().to(device)
    for p in list_images():
        name = os.path.splitext(os.path.basename(p))[0]
        x = preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            t = int(mw(x).argmax())
        # save the denormalized input once
        rgb = np.clip(x[0].cpu().numpy().transpose(1, 2, 0) * STD + MEAN, 0, 1)
        np.save(os.path.join(TMP, f"{name}__input.npy"), rgb)
        for meth in BASELINES:
            fn = METHODS.get(meth)(model_wrapper=mw, model=mw.model)
            np.save(os.path.join(TMP, f"{name}__{meth}.npy"), _map2d(fn(x, t)))
        print(f"baselines {name}: pred {t}", flush=True)


def run_hilrp():
    import timm
    from PIL import Image
    from xai_bench.methods.hilrp_method import _dispatch
    attribute = _dispatch(MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(MODEL, pretrained=True).eval().to(device)
    for p in list_images():
        name = os.path.splitext(os.path.basename(p))[0]
        x = preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
        m = attribute(model, x, gamma=0.25)["pixel_map"].numpy()
        np.save(os.path.join(TMP, f"{name}__hilrp.npy"), m)
        print(f"hilrp {name}: done", flush=True)


def _heat(mp):
    """Smoothed, percentile-normalized magnitude in [0,1] for overlay."""
    from scipy.ndimage import gaussian_filter, zoom
    h = np.abs(np.asarray(mp, dtype=np.float64))
    if h.shape[0] != SIZE:
        h = zoom(h, SIZE / h.shape[0], order=1)
    h = gaussian_filter(h, sigma=2.5)
    hi = np.percentile(h, 99.0) + 1e-12
    return np.clip(h / hi, 0, 1)


def _overlay(ax, rgb, heat, cmap="turbo"):
    import matplotlib.pyplot as plt
    gray = rgb.mean(2, keepdims=True).repeat(3, 2) * 0.55 + 0.15
    ax.imshow(gray)
    ax.imshow(heat, cmap=cmap, alpha=np.clip(heat * 1.1, 0, 1))
    ax.axis("off")


def contact():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [os.path.splitext(os.path.basename(p))[0] for p in list_images()]
    names = [n for n in names if os.path.exists(os.path.join(TMP, f"{n}__hilrp.npy"))]
    cols = ["input", "hilrp"] + BASELINES
    titles = ["input", "HiLRP", "Grad-CAM", "Grad-CAM++", "Int.Grad"]
    fig, ax = plt.subplots(len(names), len(cols), figsize=(1.7 * len(cols), 1.7 * len(names)))
    ax = np.atleast_2d(ax)
    for r, n in enumerate(names):
        rgb = np.load(os.path.join(TMP, f"{n}__input.npy"))
        ax[r, 0].imshow(rgb); ax[r, 0].axis("off")
        ax[r, 0].text(-0.1, 0.5, n, rotation=90, va="center", ha="right",
                      transform=ax[r, 0].transAxes, fontsize=8)
        for c, key in enumerate(cols[1:], 1):
            fp = os.path.join(TMP, f"{n}__{key}.npy")
            if os.path.exists(fp):
                _overlay(ax[r, c], rgb, _heat(np.load(fp)))
            else:
                ax[r, c].axis("off")
        if r == 0:
            for c, t in enumerate(titles):
                ax[0, c].set_title(t, fontsize=9)
    fig.tight_layout()
    fig.savefig("results/qual_gallery/_contact_gallery.png", dpi=110, bbox_inches="tight")
    print("wrote results/qual_gallery/_contact_gallery.png")


def assemble(chosen):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.family"] = "Times New Roman"
    import matplotlib.pyplot as plt
    cols = ["input", "hilrp"] + BASELINES
    titles = ["Input", "HiLRP (ours)", "Grad-CAM", "Grad-CAM++", "Integrated Grad"]
    fig, ax = plt.subplots(len(chosen), len(cols), figsize=(2.15 * len(cols), 2.15 * len(chosen)))
    ax = np.atleast_2d(ax)
    for r, n in enumerate(chosen):
        rgb = np.load(os.path.join(TMP, f"{n}__input.npy"))
        ax[r, 0].imshow(rgb); ax[r, 0].axis("off")
        for c, key in enumerate(cols[1:], 1):
            _overlay(ax[r, c], rgb, _heat(np.load(os.path.join(TMP, f"{n}__{key}.npy"))))
        if r == 0:
            for c, t in enumerate(titles):
                ax[0, c].set_title(t, fontsize=17, fontweight="bold")
    fig.subplots_adjust(wspace=0.03, hspace=0.03)
    out = "figures/bench/qualitative_gallery.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "contact"
    if mode == "assemble":
        assemble(sys.argv[2].split(","))
    else:
        {"baselines": run_baselines, "hilrp": run_hilrp, "contact": contact}[mode]()
