"""One image (images/cat.jpg), every model x every method: the full qualitative
grid. Rows = 8 benchmark backbones, columns = input + 14 attribution methods.

Run in FOUR separate compute processes (class-level patches from hilrp/attnlrp
would alter the baselines' gradients in-process; the flat-ViT AttnLRP path and
the naive hierarchical path must also not share a process), then assemble:

  python scripts/bench/qualitative_grid_cat.py baselines
  python scripts/bench/qualitative_grid_cat.py hilrp
  python scripts/bench/qualitative_grid_cat.py attnlrp_hier
  python scripts/bench/qualitative_grid_cat.py attnlrp_vit
  python scripts/bench/qualitative_grid_cat.py assemble

Every process explains the SAME class per model: the wrapper's top-1 on the cat
image, recorded by the baselines pass in targets.json.
"""
import json
import os
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

IMAGE = os.path.join("images", "cat.jpg")
TMP = os.path.join("results", "qual_grid_cat")
OUT = os.path.join("figures", "bench", "qualitative_grid_cat.png")
os.makedirs(TMP, exist_ok=True)

MODELS = ["resnet50", "vit_base_patch16_224", "swin_base_patch4_window7_224",
          "pvt_v2_b2", "maxvit_small_tf_224", "mobilevitv2_100",
          "efficientvit_b1", "efficientvit_b2"]
ROWLABEL = {"resnet50": "RN-50", "vit_base_patch16_224": "ViT-B/16",
            "swin_base_patch4_window7_224": "Swin-B", "pvt_v2_b2": "PVT-v2",
            "maxvit_small_tf_224": "MaxViT-S", "mobilevitv2_100": "MobileViT-v2",
            "efficientvit_b1": "EffViT-B1", "efficientvit_b2": "EffViT-B2"}

BASELINES = ["saliency", "integrated_gradients", "input_x_gradient", "smoothgrad",
             "vargrad", "gradient_shap", "grad_cam", "grad_cam_plus_plus", "lrp",
             "occlusion", "rise", "lime"]
HIER = [m for m in MODELS if m not in ("resnet50", "vit_base_patch16_224")]

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
SIZE = 224


def load_pil():
    from PIL import Image
    return Image.open(IMAGE).convert("RGB")


def preprocess_imagenet():
    from torchvision import transforms
    t = transforms.Compose([
        transforms.Resize((SIZE, SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(MEAN), std=list(STD)),
    ])
    return t(load_pil())


def preprocess_native(model):
    from torchvision import transforms
    cfg = model.pretrained_cfg
    t = transforms.Compose([
        transforms.Resize((SIZE, SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(cfg["mean"]), std=list(cfg["std"])),
    ])
    return t(load_pil())


def _map2d(m):
    m = m[0] if getattr(m, "ndim", 2) == 3 else m
    return m.detach().cpu().numpy() if torch.is_tensor(m) else np.asarray(m)


def get_wrapper(name):
    from xai_bench.registry import MODELS as REG
    from xai_bench.models.timm_models import create_timm_model
    try:
        return REG.get(name)()
    except Exception:
        return create_timm_model(name)


def _save_target(name, t):
    path = os.path.join(TMP, "targets.json")
    targets = {}
    if os.path.exists(path):
        with open(path) as f:
            targets = json.load(f)
    targets[name] = t
    with open(path, "w") as f:
        json.dump(targets, f)


def _record_failure(name, meth, err):
    """Attempted-but-failed cells (vs structurally undefined) for assemble()."""
    path = os.path.join(TMP, "failures.json")
    fails = {}
    if os.path.exists(path):
        with open(path) as f:
            fails = json.load(f)
    fails[f"{name}__{meth}"] = str(err)[:200]
    with open(path, "w") as f:
        json.dump(fails, f, indent=1)


def run_baselines(only=None):
    """One model per process (pass the model name): a CUDA OOM from the mask
    batches of RISE/LIME on the 8GB card poisons the allocator for everything
    after it, so isolation + a CPU retry keeps every cell recoverable."""
    import gc
    from xai_bench.registry import METHODS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_cpu = preprocess_imagenet()
    rgb = np.clip(x_cpu.numpy().transpose(1, 2, 0) * STD + MEAN, 0, 1)
    np.save(os.path.join(TMP, "input.npy"), rgb)

    for name in ([only] if only else MODELS):
        mw = get_wrapper(name)
        mw.model.eval().to(device)
        x = x_cpu.unsqueeze(0).to(device)
        with torch.no_grad():
            t = int(mw(x).argmax())
        _save_target(name, t)
        methods = BASELINES + (["attention_rollout"] if name == "vit_base_patch16_224" else [])
        for meth in methods:
            out = os.path.join(TMP, f"{name}__{meth}.npy")
            if os.path.exists(out):
                continue
            try:
                fn = METHODS.get(meth)(model_wrapper=mw, model=mw.model)
                np.save(out, _map2d(fn(x, t)))
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    print(f"  FAIL {name}/{meth}: {e}", flush=True)
                    _record_failure(name, meth, e)
                else:
                    print(f"  OOM {name}/{meth}: retrying on CPU", flush=True)
                    try:
                        gc.collect()
                        torch.cuda.empty_cache()
                        mw_cpu = get_wrapper(name)
                        mw_cpu.model.eval().to("cpu")
                        fn = METHODS.get(meth)(model_wrapper=mw_cpu, model=mw_cpu.model)
                        np.save(out, _map2d(fn(x_cpu.unsqueeze(0), t)))
                        del mw_cpu
                    except Exception as e2:
                        print(f"  FAIL {name}/{meth} (cpu): {e2}", flush=True)
                        _record_failure(name, meth, e2)
            except Exception as e:
                print(f"  FAIL {name}/{meth}: {e}", flush=True)
                _record_failure(name, meth, e)
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except RuntimeError:
                pass
        del mw
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except RuntimeError:
            pass
        print(f"baselines {name}: target {t}", flush=True)


def _targets():
    with open(os.path.join(TMP, "targets.json")) as f:
        return json.load(f)


def run_hilrp():
    import timm
    from xai_bench.methods.hilrp_method import _dispatch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    targets = _targets()
    for name in [m for m in MODELS if m != "resnet50"]:
        model = timm.create_model(name, pretrained=True).eval().to(device)
        x = preprocess_native(model).unsqueeze(0).to(device)
        res = _dispatch(name)(model, x, target=targets[name], gamma=0.25)
        np.save(os.path.join(TMP, f"{name}__hilrp.npy"), res["pixel_map"].numpy())
        del model
        torch.cuda.empty_cache()
        print(f"hilrp {name}: done", flush=True)


def run_attnlrp_hier():
    import timm
    from xai_bench.methods.attnlrp_method import _naive_patch_once, _attribute_naive
    _naive_patch_once()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    targets = _targets()
    for name in HIER:
        model = timm.create_model(name, pretrained=True).eval().to(device)
        x = preprocess_native(model).unsqueeze(0).to(device)
        pm = _attribute_naive(model, x, target=targets[name], gamma=0.25)
        np.save(os.path.join(TMP, f"{name}__attnlrp.npy"), pm.numpy())
        del model
        torch.cuda.empty_cache()
        print(f"attnlrp {name}: done", flush=True)


def run_attnlrp_vit():
    import timm
    from xai_bench.methods.hilrp.vit_lxt import attribute_vit
    device = "cuda" if torch.cuda.is_available() else "cpu"
    targets = _targets()
    name = "vit_base_patch16_224"
    model = timm.create_model(name, pretrained=True).eval().to(device)
    x = preprocess_native(model).unsqueeze(0).to(device)
    res = attribute_vit(model, x, target=targets[name], gamma=0.25, attn_mode="attnlrp")
    np.save(os.path.join(TMP, f"{name}__attnlrp.npy"), res["pixel_map"].numpy())
    print(f"attnlrp {name}: done", flush=True)


def _heat(mp):
    from scipy.ndimage import gaussian_filter, zoom
    h = np.abs(np.asarray(mp, dtype=np.float64))
    if h.shape[0] != SIZE:
        h = zoom(h, SIZE / h.shape[0], order=1)
    h = gaussian_filter(h, sigma=2.5)
    hi = np.percentile(h, 99.0) + 1e-12
    return np.clip(h / hi, 0, 1)


def assemble():
    """Benchmark version: no HiLRP column. Failure instances are explicit:
    attempted-but-failed cells (Captum LRP on every transformer) are rendered
    as 'does not run', and the documented localization collapses (Grad-CAM /
    Grad-CAM++ on linear attention, rollout on ViT) get a red border."""
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.family"] = "Times New Roman"
    import matplotlib.pyplot as plt

    cols = ["saliency", "integrated_gradients", "input_x_gradient",
            "smoothgrad", "vargrad", "gradient_shap", "grad_cam",
            "grad_cam_plus_plus", "attention_rollout", "attnlrp",
            "occlusion", "rise", "lime"]
    titles = ["Saliency", "Int.Grad", "Inp$\\times$Grad", "SmoothGrad",
              "VarGrad", "GradSHAP", "Grad-CAM", "Grad-CAM++",
              "Rollout", "AttnLRP", "Occlusion", "RISE", "LIME"]

    # documented collapse instances (red border): Grad-CAM family on linear
    # attention (PG 0.44-0.56 vs the 0.61 random-point prior), rollout on ViT
    COLLAPSE = {("efficientvit_b1", "grad_cam"), ("efficientvit_b2", "grad_cam"),
                ("efficientvit_b1", "grad_cam_plus_plus"),
                ("efficientvit_b2", "grad_cam_plus_plus"),
                ("vit_base_patch16_224", "attention_rollout")}

    fails = {}
    fp_fail = os.path.join(TMP, "failures.json")
    if os.path.exists(fp_fail):
        with open(fp_fail) as f:
            fails = json.load(f)

    rgb = np.load(os.path.join(TMP, "input.npy"))
    gray = rgb.mean(2, keepdims=True).repeat(3, 2) * 0.55 + 0.15

    nr, nc = len(MODELS), len(cols)
    fig, ax = plt.subplots(nr, nc, figsize=(1.35 * nc, 1.42 * nr))
    for r, name in enumerate(MODELS):
        ax[r, 0].text(-0.12, 0.5, ROWLABEL[name], rotation=90, va="center",
                      ha="center", transform=ax[r, 0].transAxes,
                      fontsize=13, fontweight="bold")
        for c, key in enumerate(cols):
            a = ax[r, c]
            a.axis("off")
            fp = os.path.join(TMP, f"{name}__{key}.npy")
            if os.path.exists(fp):
                h = _heat(np.load(fp))
                a.imshow(gray)
                a.imshow(h, cmap="turbo", alpha=np.clip(h * 1.1, 0, 1))
                if (name, key) in COLLAPSE:
                    a.add_patch(plt.Rectangle((0.012, 0.012), 0.976, 0.976,
                                transform=a.transAxes, fill=False,
                                edgecolor="#C1121F", linewidth=3.0))
            elif f"{name}__{key}" in fails:
                a.imshow(np.full((SIZE, SIZE, 3), [0.98, 0.92, 0.92]))
                a.text(0.5, 0.5, "does not\nrun", transform=a.transAxes,
                       ha="center", va="center", fontsize=12,
                       color="#8B1A1A", fontweight="bold")
                a.add_patch(plt.Rectangle((0.012, 0.012), 0.976, 0.976,
                            transform=a.transAxes, fill=False,
                            edgecolor="#C1121F", linewidth=1.2))
            else:
                a.imshow(np.full((SIZE, SIZE, 3), 0.93))
                a.text(0.5, 0.5, "n/a", transform=a.transAxes, ha="center",
                       va="center", fontsize=12, color="0.45")
        if r == 0:
            for c, t in enumerate(titles):
                ax[0, c].set_title(t, fontsize=13, fontweight="bold")
    fig.subplots_adjust(wspace=0.03, hspace=0.03, left=0.02, right=0.995,
                        top=0.96, bottom=0.005)
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "assemble"
    if mode == "baselines":
        run_baselines(sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        {"hilrp": run_hilrp, "attnlrp_hier": run_attnlrp_hier,
         "attnlrp_vit": run_attnlrp_vit, "assemble": assemble}[mode]()
