"""Build figures/bench/qualitative_comparison.png: HiLRP vs baselines across models.

Run in THREE separate processes (HiLRP's class-level patches would alter the
baselines' gradients if run in-process):

  python scripts/bench/qualitative_comparison.py baselines   # vanilla models
  python scripts/bench/qualitative_comparison.py hilrp        # patched models
  python scripts/bench/qualitative_comparison.py assemble     # build the PNG

Grid: rows = Swin-B, PVT-v2, EfficientViT-B2, MobileViT-v2; columns = input,
HiLRP, Grad-CAM, Grad-CAM++, Integrated Grad, SmoothGrad. The same image is used
for every row so the EfficientViT collapse of Grad-CAM is directly comparable to
the other architectures.
"""
import os
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

CACHE = "data/ImageNetS/cache_validation_100.pt"
TMP = os.path.join("results", "qual_tmp")
os.makedirs(TMP, exist_ok=True)
os.makedirs("figures", exist_ok=True)

MODELS = ["swin_base_patch4_window7_224", "pvt_v2_b2", "efficientvit_b2", "mobilevitv2_100"]
LABELS = {"swin_base_patch4_window7_224": "Swin-B", "pvt_v2_b2": "PVT-v2",
          "efficientvit_b2": "EfficientViT-B2", "mobilevitv2_100": "MobileViT-v2"}
IMG_IDX = int(os.environ.get("QUAL_IDX", "0"))
BASELINES = ["grad_cam", "grad_cam_plus_plus", "integrated_gradients", "smoothgrad"]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _map2d(m):
    m = m[0] if getattr(m, "ndim", 2) == 3 else m
    return m.detach().cpu().numpy() if torch.is_tensor(m) else np.asarray(m)


def run_baselines():
    from xai_bench.registry import MODELS as REG, METHODS
    data = torch.load(CACHE, map_location="cpu", weights_only=False)
    x = data[IMG_IDX]["image"].unsqueeze(0)
    for name in MODELS:
        mw = REG.get(name)(); mw.model.eval()
        with torch.no_grad():
            t = int(mw(x).argmax())
        for meth in BASELINES:
            fn = METHODS.get(meth)(model_wrapper=mw, model=mw.model)
            m = _map2d(fn(x, t))
            np.save(os.path.join(TMP, f"{name}__{meth}.npy"), m)
        print(f"baselines {LABELS[name]}: pred {t}  done")


def run_hilrp():
    from xai_bench.methods.hilrp.swin_lxt import attribute_swin, ensure_patched as sp
    from xai_bench.methods.hilrp.pvt_lxt import attribute_pvt, ensure_patched as pp
    from xai_bench.methods.hilrp.efficientvit_lxt import attribute_efficientvit, ensure_patched as ep
    from xai_bench.methods.hilrp.mobilevit_lxt import attribute_mobilevit, ensure_patched as mp
    import timm
    sp(); pp(); ep(); mp()
    data = torch.load(CACHE, map_location="cpu", weights_only=False)
    xn = data[IMG_IDX]["image"].unsqueeze(0)                 # ImageNet-normalized
    xraw = (data[IMG_IDX]["image"] * STD + MEAN).clamp(0, 1).unsqueeze(0)
    jobs = [
        ("swin_base_patch4_window7_224", attribute_swin, xn),
        ("pvt_v2_b2", attribute_pvt, xn),
        ("efficientvit_b2", attribute_efficientvit, xn),
        ("mobilevitv2_100", attribute_mobilevit, xraw),      # mobilevit wants raw [0,1]
    ]
    for name, attr, x in jobs:
        model = timm.create_model(name, pretrained=True).eval()
        m = attr(model, x, gamma=0.25)["pixel_map"].numpy()
        np.save(os.path.join(TMP, f"{name}__hilrp.npy"), m)
        print(f"hilrp {LABELS[name]}: done")


def assemble():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from xai_bench.methods.hilrp.viz import normalize_for_display
    data = torch.load(CACHE, map_location="cpu", weights_only=False)
    img = np.clip(data[IMG_IDX]["image"].permute(1, 2, 0).numpy() * STD.squeeze().numpy()
                  + MEAN.squeeze().numpy(), 0, 1)

    cols = ["input", "hilrp"] + BASELINES
    titles = ["input", "HiLRP (ours)", "Grad-CAM", "Grad-CAM++", "Integrated Grad", "SmoothGrad"]
    fig, ax = plt.subplots(len(MODELS), len(cols), figsize=(2.05 * len(cols), 2.05 * len(MODELS)))
    for r, name in enumerate(MODELS):
        ax[r, 0].imshow(img); ax[r, 0].axis("off")
        ax[r, 0].set_ylabel(LABELS[name], fontsize=10)
        ax[r, 0].set_title("input" if r == 0 else "", fontsize=9)
        # re-show ylabel (axis off hides it) via text
        ax[r, 0].text(-0.08, 0.5, LABELS[name], rotation=90, va="center", ha="right",
                      transform=ax[r, 0].transAxes, fontsize=10)
        for c, key in enumerate(cols[1:], start=1):
            p = os.path.join(TMP, f"{name}__{key}.npy")
            if not os.path.exists(p):
                ax[r, c].axis("off"); continue
            m, v = normalize_for_display(np.load(p), percentile=99.0, smooth_sigma=0.6)
            ax[r, c].imshow(m, cmap="bwr", vmin=-v, vmax=v); ax[r, c].axis("off")
            if r == 0:
                ax[r, c].set_title(titles[c], fontsize=9)
    fig.suptitle("HiLRP vs baseline attributions across architectures (same input)", fontsize=11)
    fig.tight_layout()
    out = "figures/bench/qualitative_comparison.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "assemble"
    {"baselines": run_baselines, "hilrp": run_hilrp, "assemble": assemble}[mode]()
