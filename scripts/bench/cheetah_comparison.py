"""Rebuild the cross-architecture comparison figure (paper Fig. 3) on a chosen
images/ photo (default images/cheetah.jpg) instead of the ImageNet-S yurt.

HiLRP's class-level LRP patches would perturb the baselines' gradients if run in
the same process, so baselines and HiLRP are generated in SEPARATE runs:

  python scripts/bench/cheetah_comparison.py baselines   # vanilla models
  python scripts/bench/cheetah_comparison.py hilrp        # patched models
  python scripts/bench/cheetah_comparison.py assemble     # build the PNG/PDF

Grid: rows = Swin-B, PVT-v2-b2, EfficientViT-B2, MobileViT-v2; columns = Input,
HiLRP, Grad-CAM, Grad-CAM++, Integrated Grad, SmoothGrad. Same image every row,
so Grad-CAM's collapse on EfficientViT's linear attention is directly comparable.

Override the image with QUAL_IMG (path). Assumes CWD = repo root.
"""
import os
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

IMG = os.environ.get("QUAL_IMG", "images/cheetah.jpg")
TMP = os.path.join("results", "cheetah_tmp")
OUT_DIR = os.path.join("figures", "bench")
os.makedirs(TMP, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = ["swin_base_patch4_window7_224", "pvt_v2_b2", "efficientvit_b2", "mobilevitv2_100"]
LABELS = {"swin_base_patch4_window7_224": "Swin-B", "pvt_v2_b2": "PVT-v2-b2",
          "efficientvit_b2": "EfficientViT-B2", "mobilevitv2_100": "MobileViT-v2"}
BASELINES = ["grad_cam", "grad_cam_plus_plus", "integrated_gradients", "smoothgrad"]
TITLES = ["Input", "HiLRP (ours)", "Grad-CAM", "Grad-CAM++", "Integrated Grad", "SmoothGrad"]
COLLAPSE_ROW = 2  # EfficientViT-B2
SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def _preprocess():
    from PIL import Image
    from torchvision import transforms
    t = transforms.Compose([transforms.Resize((SIZE, SIZE)), transforms.ToTensor(),
                            transforms.Normalize(mean=list(MEAN), std=list(STD))])
    return t(Image.open(IMG).convert("RGB")).unsqueeze(0)


def _map2d(m):
    m = m[0] if getattr(m, "ndim", 2) == 3 else m
    return m.detach().cpu().numpy() if torch.is_tensor(m) else np.asarray(m)


def run_baselines():
    from xai_bench.registry import MODELS as REG, METHODS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    xn = _preprocess()
    rgb = np.clip(xn[0].numpy().transpose(1, 2, 0) * STD + MEAN, 0, 1)
    np.save(os.path.join(TMP, "__input.npy"), rgb)
    for name in MODELS:
        mw = REG.get(name)(); mw.model.eval().to(device)
        x = xn.to(device)
        with torch.no_grad():
            t = int(mw(x).argmax())
        for meth in BASELINES:
            fn = METHODS.get(meth)(model_wrapper=mw, model=mw.model)
            np.save(os.path.join(TMP, f"{name}__{meth}.npy"), _map2d(fn(x, t)))
        print(f"baselines {LABELS[name]}: pred {t} done", flush=True)


def run_hilrp():
    import timm
    # lxt>=2.1 ships lxt/efficient/models/bert.py, which imports a transformers
    # symbol removed in transformers 5.x. HiLRP never uses lxt's default model
    # maps (it passes its own patch_map), and get_default_map is the only thing
    # core.py needs from that package, so we stub it out to avoid the broken
    # transitive import without downgrading transformers.
    import types
    _stub = types.ModuleType("lxt.efficient.models")
    _stub.get_default_map = lambda *a, **k: {}
    sys.modules["lxt.efficient.models"] = _stub

    from xai_bench.methods.hilrp.swin_lxt import attribute_swin, ensure_patched as sp
    from xai_bench.methods.hilrp.pvt_lxt import attribute_pvt, ensure_patched as pp
    from xai_bench.methods.hilrp.efficientvit_lxt import attribute_efficientvit, ensure_patched as ep
    from xai_bench.methods.hilrp.mobilevit_lxt import attribute_mobilevit, ensure_patched as mp
    sp(); pp(); ep(); mp()
    xn = _preprocess()                                      # ImageNet-normalized
    xraw = (xn[0] * torch.tensor(STD).view(3, 1, 1) + torch.tensor(MEAN).view(3, 1, 1)
            ).clamp(0, 1).unsqueeze(0).float()              # raw [0,1] for MobileViT
    jobs = [
        ("swin_base_patch4_window7_224", attribute_swin, xn),
        ("pvt_v2_b2", attribute_pvt, xn),
        ("efficientvit_b2", attribute_efficientvit, xn),
        ("mobilevitv2_100", attribute_mobilevit, xraw),
    ]
    for name, attr, x in jobs:
        model = timm.create_model(name, pretrained=True).eval()
        m = attr(model, x.float(), gamma=0.25)["pixel_map"].numpy()
        np.save(os.path.join(TMP, f"{name}__hilrp.npy"), m)
        print(f"hilrp {LABELS[name]}: done", flush=True)


def _heat(mp):
    from scipy.ndimage import gaussian_filter, zoom
    h = np.abs(np.asarray(mp, dtype=np.float64))
    if h.shape[0] != SIZE:
        h = zoom(h, SIZE / h.shape[0], order=1)
    h = gaussian_filter(h, sigma=2.5)
    return np.clip(h / (np.percentile(h, 99.0) + 1e-12), 0, 1)


def assemble():
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.family"] = "Times New Roman"
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    rgb = np.load(os.path.join(TMP, "__input.npy"))
    cols = ["hilrp"] + BASELINES
    n_r, n_c = len(MODELS), len(cols) + 1
    fig, ax = plt.subplots(n_r, n_c, figsize=(1.95 * n_c, 1.95 * n_r))
    ax = np.atleast_2d(ax)

    def _overlay(a, heat):
        gray = rgb.mean(2, keepdims=True).repeat(3, 2) * 0.55 + 0.15
        a.imshow(gray)
        a.imshow(heat, cmap="turbo", alpha=np.clip(heat * 1.1, 0, 1))

    for r, key in enumerate(MODELS):
        ax[r, 0].imshow(rgb)
        ax[r, 0].set_xticks([]); ax[r, 0].set_yticks([])
        for s in ax[r, 0].spines.values():
            s.set_visible(False)
        ax[r, 0].set_ylabel(LABELS[key], fontsize=13, fontweight="bold", labelpad=8)
        for c, meth in enumerate(cols, start=1):
            p = os.path.join(TMP, f"{key}__{meth}.npy")
            if os.path.exists(p):
                _overlay(ax[r, c], _heat(np.load(p)))
            ax[r, c].axis("off")
    for c, t in enumerate(TITLES):
        ax[0, c].set_title(t, fontsize=14, fontweight="bold", pad=6)

    # Flag the Grad-CAM collapse on linear attention (scores are dataset-level).
    def _flag(cell, color, text):
        a = ax[cell]
        a.add_patch(Rectangle((1, 1), SIZE - 3, SIZE - 3, fill=False,
                              edgecolor=color, linewidth=3.0, zorder=10))
        a.text(0.5, 0.965, text, transform=a.transAxes, ha="center", va="top",
               fontsize=10, color="white", fontweight="bold", zorder=11,
               bbox=dict(boxstyle="round,pad=0.22", facecolor=color, edgecolor="none"))

    _flag((COLLAPSE_ROW, 2), "#d81b1b", "dataset Pointing 0.55")   # Grad-CAM
    _flag((COLLAPSE_ROW, 1), "#1a7f37", "dataset Pointing 0.96")   # HiLRP

    fig.subplots_adjust(wspace=0.04, hspace=0.06, left=0.06)
    out = os.path.join(OUT_DIR, "arch_method_comparison")
    fig.savefig(out + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    print("wrote", out + ".png")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "assemble"
    {"baselines": run_baselines, "hilrp": run_hilrp, "assemble": assemble}[mode]()
