"""Scaled Pointing-Game evaluation with confidence intervals and paired tests.

Addresses the scale gap: run the win-condition comparison (HiLRP vs Grad-CAM on
the architectures where CAM collapses) at n>=500 with 95% bootstrap CIs and a
paired McNemar test, instead of n=100 with no error bars.

HiLRP (class-patched) and Grad-CAM (vanilla) must run in SEPARATE processes:

  python scripts/bench/scaled_eval.py gradcam efficientvit_b2 1000
  python scripts/bench/scaled_eval.py hilrp   efficientvit_b2 1000
  python scripts/bench/scaled_eval.py stats   efficientvit_b2

Per-image pointing hits are cached to results/scaled/<model>_<method>.npy so
stats can combine them (paired: both methods on the same images).
"""
import os
import sys
import glob
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

OUT = os.path.join("results", "scaled")
os.makedirs(OUT, exist_ok=True)


DATASETS = {
    "imagenets": os.path.join("data", "ImageNetS", "cache_validation_*.pt"),
    "voc": os.path.join("data", "VOC", "cache_voc_*.pt"),
}


def load_cache(dataset="imagenets"):
    caches = glob.glob(DATASETS[dataset])
    path = max(caches, key=lambda p: int(p.rsplit("_", 1)[-1].split(".")[0]))
    return torch.load(path, map_location="cpu", weights_only=False), path


def npy_name(model_name, method, dataset):
    """ImageNet-S keeps the legacy <model>_<method>.npy name; other datasets get
    a <model>_<dataset>_<method>.npy name so results never clobber each other."""
    tag = "" if dataset == "imagenets" else f"_{dataset}"
    return os.path.join(OUT, f"{model_name}{tag}_{method}.npy")


def pointing(pm, bboxes):
    h, w = pm.shape
    py, px = np.unravel_index(np.argmax(pm), pm.shape)
    return any(b[0] * w <= px <= b[2] * w and b[1] * h <= py <= b[3] * h for b in bboxes)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_gradcam(model_name, n, dataset="imagenets"):
    from xai_bench.registry import MODELS, METHODS
    data, path = load_cache(dataset)
    n = min(n, len(data))
    mw = MODELS.get(model_name)(); mw.model.eval().to(DEVICE)
    gc = METHODS.get("grad_cam")(model_wrapper=mw, model=mw.model)
    hits = np.zeros(n, dtype=bool)
    for i in range(n):
        x = data[i]["image"].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            t = int(mw(x).argmax())
        m = gc(x, t); m = (m[0] if m.ndim == 3 else m).detach().cpu().numpy()
        hits[i] = pointing(m, data[i]["metadata"]["bboxes"])
        if (i + 1) % 100 == 0:
            print(f"  gradcam {i+1}/{n}  running {hits[:i+1].mean():.3f}", flush=True)
    np.save(npy_name(model_name, "gradcam", dataset), hits)
    print(f"grad_cam {model_name} [{dataset}]: {hits.mean():.3f} (n={n}, cache={path})")


def run_hilrp(model_name, n, dataset="imagenets"):
    import timm
    import torch.nn.functional as F
    from xai_bench.methods.hilrp_method import _dispatch
    attribute = _dispatch(model_name)
    data, path = load_cache(dataset)
    n = min(n, len(data))
    model = timm.create_model(model_name, pretrained=True).eval().to(DEVICE)
    size = model.default_cfg.get("input_size", (3, 224, 224))[-1]
    cfg = model.pretrained_cfg
    mean = torch.tensor(cfg["mean"]).view(3, 1, 1); std = torch.tensor(cfg["std"]).view(3, 1, 1)
    IM = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1); IS = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    need_raw = any(abs(a - b) > 1e-6 for a, b in zip(cfg["mean"], (0.485, 0.456, 0.406)))
    hits = np.zeros(n, dtype=bool)
    for i in range(n):
        img = data[i]["image"]
        if need_raw:
            img = ((img * IS + IM).clamp(0, 1) - mean) / std
        x = img.unsqueeze(0).to(DEVICE)
        if size != 224:
            x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        r = attribute(model, x, gamma=0.25)
        hits[i] = pointing(r["pixel_map"].numpy(), data[i]["metadata"]["bboxes"])
        if (i + 1) % 100 == 0:
            print(f"  hilrp {i+1}/{n}  running {hits[:i+1].mean():.3f}", flush=True)
    np.save(npy_name(model_name, "hilrp", dataset), hits)
    print(f"hilrp {model_name} [{dataset}]: {hits.mean():.3f} (n={n})")


def stats(model_name, dataset="imagenets"):
    from scipy.stats import binomtest
    h = np.load(npy_name(model_name, "hilrp", dataset))
    g = np.load(npy_name(model_name, "gradcam", dataset))
    n = min(len(h), len(g)); h, g = h[:n], g[:n]

    def ci(x):
        rng = np.random.RandomState(0)
        boot = [x[rng.randint(0, len(x), len(x))].mean() for _ in range(2000)]
        return x.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)

    hm, hlo, hhi = ci(h); gm, glo, ghi = ci(g)
    # paired McNemar: disagreements
    b = int((h & ~g).sum())   # HiLRP hit, GradCAM miss
    c = int((~h & g).sum())   # GradCAM hit, HiLRP miss
    p = binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) > 0 else 1.0
    print(f"=== {model_name} [{dataset}]  (n={n}) ===")
    print(f"  HiLRP    {hm:.3f}  95% CI [{hlo:.3f}, {hhi:.3f}]")
    print(f"  Grad-CAM {gm:.3f}  95% CI [{glo:.3f}, {ghi:.3f}]")
    print(f"  paired McNemar: HiLRP-only={b}, GradCAM-only={c}, p={p:.2e}")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "stats":
        stats(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "imagenets")
    else:
        dataset = sys.argv[4] if len(sys.argv) > 4 else "imagenets"
        {"gradcam": run_gradcam, "hilrp": run_hilrp}[mode](sys.argv[2], int(sys.argv[3]), dataset)
