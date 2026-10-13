"""Deletion / Insertion curves with blur infill, as a METRIC DIAGNOSTIC.

The paper's position (Sec. V-A) is that perturbation-based fidelity cannot
referee modern ViTs, because they are robust to the spatial masking these
metrics rely on. Faithfulness Correlation is shown to be noise. This script
tests the same claim on the other two members of that family, Deletion and
Insertion AUC, so the critique rests on measurement rather than assertion.

It is deliberately NOT a win condition. If HiLRP wins here, good; if the methods
are statistically indistinguishable, that is the point being made.

Protocol choices that matter:
  * blur infill, not black patches. Black patches are off-manifold and produce a
    confounded drop that measures OOD sensitivity rather than attribution
    quality (the critique raised by TIS).
  * patch-level ranking on a 16x16 grid, matched to the token grid, so the
    perturbation unit is one the architecture can actually represent.
  * the curve tracks softmax probability of the ORIGINALLY PREDICTED class, so
    a misclassified image still measures the explanation of the model.

  python scripts/bench/perturbation_curves.py gradcam efficientvit_b2 100
  python scripts/bench/perturbation_curves.py hilrp   efficientvit_b2 100
  python scripts/bench/perturbation_curves.py report  efficientvit_b2
"""
import os
import sys
import warnings

import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.bench.scaled_eval import OUT, DEVICE, load_cache  # noqa: E402

PATCH = 16
N_STEPS = 25
BLUR_SIGMA = 11


def npz_name(model_name, method, dataset):
    tag = "" if dataset == "imagenets" else f"_{dataset}"
    return os.path.join(OUT, f"{model_name}{tag}_{method}_perturb.npz")


def gaussian_blur(x, sigma=BLUR_SIGMA):
    """Heavy Gaussian blur used as the in-distribution replacement signal."""
    k = int(sigma * 2) * 2 + 1
    coords = torch.arange(k, dtype=torch.float32, device=x.device) - k // 2
    g = torch.exp(-(coords ** 2) / (2 * (sigma / 2) ** 2))
    g = (g / g.sum()).view(1, 1, 1, -1)
    c = x.shape[1]
    out = F.conv2d(F.pad(x, (k // 2,) * 2 + (0, 0), mode="reflect"),
                   g.expand(c, 1, 1, k), groups=c)
    out = F.conv2d(F.pad(out, (0, 0) + (k // 2,) * 2, mode="reflect"),
                   g.transpose(2, 3).expand(c, 1, k, 1), groups=c)
    return out


def patch_order(pixel_map, grid):
    """Descending relevance order over a grid x grid patch lattice."""
    m = np.asarray(pixel_map, dtype=np.float64)
    if m.ndim == 3:
        m = m[0]
    t = torch.from_numpy(m)[None, None]
    pooled = F.adaptive_avg_pool2d(t, grid)[0, 0].numpy()
    return np.argsort(-pooled.ravel()), pooled.size


def curves(model, x, pixel_map, target):
    """Deletion and insertion probability curves for one image."""
    grid = x.shape[-1] // PATCH
    order, n_patch = patch_order(pixel_map, grid)
    blurred = gaussian_blur(x)

    steps = np.linspace(0, n_patch, N_STEPS + 1).astype(int)
    del_batch, ins_batch = [], []
    for s in steps:
        mask = torch.zeros(n_patch, device=x.device)
        mask[torch.from_numpy(order[:s].copy()).to(x.device)] = 1.0
        mask = mask.view(1, 1, grid, grid)
        mask = F.interpolate(mask, size=x.shape[-2:], mode="nearest")
        del_batch.append(x * (1 - mask) + blurred * mask)   # remove top-ranked
        ins_batch.append(blurred * (1 - mask) + x * mask)   # restore top-ranked

    with torch.no_grad():
        d = torch.softmax(model(torch.cat(del_batch)), -1)[:, target].cpu().numpy()
        i = torch.softmax(model(torch.cat(ins_batch)), -1)[:, target].cpu().numpy()
    return d, i


def _auc(y):
    return float(np.trapezoid(y, dx=1.0 / (len(y) - 1)))


def _run(model_name, n, dataset, method):
    import timm
    data, path = load_cache(dataset)
    n = min(n, len(data))
    print(f"cache={path}")

    model = timm.create_model(model_name, pretrained=True).eval().to(DEVICE)
    size = model.default_cfg.get("input_size", (3, 224, 224))[-1]
    cfg = model.pretrained_cfg
    mean = torch.tensor(cfg["mean"]).view(3, 1, 1)
    std = torch.tensor(cfg["std"]).view(3, 1, 1)
    IM = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    IS = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    need_raw = any(abs(a - b) > 1e-6 for a, b in zip(cfg["mean"], (0.485, 0.456, 0.406)))

    if method == "hilrp":
        from xai_bench.methods.hilrp_method import _dispatch
        attribute = _dispatch(model_name)
        gc = None
    else:
        from xai_bench.registry import MODELS, METHODS
        mw = MODELS.get(model_name)()
        mw.model.eval().to(DEVICE)
        gc = METHODS.get("grad_cam")(model_wrapper=mw, model=mw.model)

    dele = np.full(n, np.nan)
    ins = np.full(n, np.nan)
    for k in range(n):
        img = data[k]["image"]
        if need_raw:
            img = ((img * IS + IM).clamp(0, 1) - mean) / std
        x = img.unsqueeze(0).to(DEVICE)
        if size != 224:
            x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        with torch.no_grad():
            target = int(model(x).argmax())

        if method == "hilrp":
            pm = attribute(model, x, gamma=0.25)["pixel_map"].numpy()
        else:
            m = gc(x, target)
            pm = (m[0] if m.ndim == 3 else m).detach().cpu().numpy()

        d, i = curves(model, x, pm, target)
        dele[k], ins[k] = _auc(d), _auc(i)
        if (k + 1) % 25 == 0:
            print(f"  {method} {k+1}/{n}  del {np.nanmean(dele[:k+1]):.4f} "
                  f"ins {np.nanmean(ins[:k+1]):.4f}", flush=True)

    np.savez(npz_name(model_name, method, dataset), deletion=dele, insertion=ins)
    print(f"{method} {model_name} [{dataset}] n={n}")
    print(f"  deletion  AUC {np.nanmean(dele):.4f}  (lower is better)")
    print(f"  insertion AUC {np.nanmean(ins):.4f}  (higher is better)")


def report(model_name, dataset="imagenets"):
    from scipy.stats import wilcoxon
    h = np.load(npz_name(model_name, "hilrp", dataset))
    g = np.load(npz_name(model_name, "gradcam", dataset))
    print(f"=== {model_name} [{dataset}] perturbation diagnostic ===")
    for k, better in (("deletion", "lower"), ("insertion", "higher")):
        a, b = h[k], g[k]
        m = ~np.isnan(a) & ~np.isnan(b)
        a, b = a[m], b[m]
        p = wilcoxon(a, b).pvalue
        print(f"  {k:<10s} ({better} better)  HiLRP {a.mean():.4f}   "
              f"Grad-CAM {b.mean():.4f}   diff {a.mean()-b.mean():+.4f}   "
              f"paired p={p:.3f}")
    print("\nRead this as a diagnostic, not a win condition: the claim under "
          "test is whether these metrics separate methods at all.")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "report":
        report(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "imagenets")
    else:
        ds = sys.argv[4] if len(sys.argv) > 4 else "imagenets"
        _run(sys.argv[2], int(sys.argv[3]), ds, mode)
