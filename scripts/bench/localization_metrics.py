"""Threshold-free localization metrics beyond the Pointing Game.

The Pointing Game only asks whether the argmax lands in the object, so it
saturates on ImageNet-S (center-point prior 0.97) and cannot separate the top
tier. This script adds the metrics that do separate it, on the same cache, the
same images, and the same per-image paired protocol as scaled_eval.py:

  energy      fraction of positive relevance mass falling inside the GT region.
              Uniform noise scores the region's AREA, so the honest number is
              the excess over that prior, reported as `energy_excess`.
  ap          average precision of |R| treated as a per-pixel detector of the
              region. Threshold-free, so it does not reward smooth blobs the
              way a fixed-threshold IoU does.
  iou_best    IoU at the best threshold over a sweep. Reported ONLY with the
              caveat that it favors smooth maps over heavy-tailed ones; see
              the note in report().

Ground truth is the union of the cached bounding boxes. The ImageNet-S cache
stores boxes derived from the segmentation masks, not the masks themselves
(cache_imagenets.py), so these are box-level, not mask-level, figures. To get
true mask-level numbers, cache_imagenets.py must be changed to keep the resized
binary mask alongside the boxes, and the cache rebuilt.

Run one method per process, as with scaled_eval.py (HiLRP is class-patched):

  python scripts/bench/localization_metrics.py gradcam efficientvit_b2 1000
  python scripts/bench/localization_metrics.py hilrp   efficientvit_b2 1000
  python scripts/bench/localization_metrics.py report  efficientvit_b2

Per-image values are cached to results/scaled/<model>[_<dataset>]_<method>_loc.npz
so report() can pair them across methods.
"""
import os
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.bench.scaled_eval import OUT, DEVICE, load_cache  # noqa: E402

METRICS = ("energy", "energy_excess", "ap", "iou_best", "area")


def npz_name(model_name, method, dataset):
    tag = "" if dataset == "imagenets" else f"_{dataset}"
    return os.path.join(OUT, f"{model_name}{tag}_{method}_loc.npz")


def region_from_bboxes(bboxes, h, w):
    """Union of the normalized xyxy boxes as a boolean array of shape (h, w)."""
    region = np.zeros((h, w), dtype=bool)
    for x0, y0, x1, y1 in bboxes:
        c0, c1 = int(np.floor(x0 * w)), int(np.ceil(x1 * w))
        r0, r1 = int(np.floor(y0 * h)), int(np.ceil(y1 * h))
        region[max(r0, 0):min(r1, h), max(c0, 0):min(c1, w)] = True
    return region


def average_precision(scores, region):
    """AP of `scores` as a per-pixel detector of `region`, computed exactly.

    Sorts pixels by score and integrates precision over recall, which is the
    threshold-free summary of the precision-recall curve.
    """
    order = np.argsort(-scores.ravel())
    hit = region.ravel()[order]
    n_pos = int(hit.sum())
    if n_pos == 0 or n_pos == hit.size:
        return np.nan
    tp = np.cumsum(hit)
    precision = tp / np.arange(1, hit.size + 1)
    return float((precision * hit).sum() / n_pos)


def best_iou(scores, region, n_thresh=50):
    """Max IoU over a sweep of quantile thresholds on `scores`."""
    qs = np.linspace(0.50, 0.999, n_thresh)
    thresholds = np.quantile(scores, qs)
    best = 0.0
    for t in thresholds:
        pred = scores >= t
        union = np.logical_or(pred, region).sum()
        if union:
            best = max(best, float(np.logical_and(pred, region).sum() / union))
    return best


def score_map(pixel_map, bboxes):
    """All metrics for one attribution map against one image's ground truth."""
    m = np.asarray(pixel_map, dtype=np.float64)
    if m.ndim == 3:
        m = m[0]
    h, w = m.shape
    region = region_from_bboxes(bboxes, h, w)
    area = float(region.mean())

    pos = np.clip(m, 0, None)
    total = pos.sum()
    energy = float(pos[region].sum() / total) if total > 0 else np.nan

    mag = np.abs(m)
    return {
        "energy": energy,
        "energy_excess": energy - area,
        "ap": average_precision(mag, region),
        "iou_best": best_iou(mag, region),
        "area": area,
    }


def _collect(maps_iter, n, model_name, method, dataset):
    acc = {k: np.full(n, np.nan) for k in METRICS}
    for i, (pixel_map, bboxes) in enumerate(maps_iter):
        if not bboxes:
            continue
        for k, v in score_map(pixel_map, bboxes).items():
            acc[k][i] = v
        if (i + 1) % 100 == 0:
            done = acc["energy_excess"][:i + 1]
            print(f"  {method} {i+1}/{n}  running energy_excess "
                  f"{np.nanmean(done):+.3f}", flush=True)
    np.savez(npz_name(model_name, method, dataset), **acc)
    print(f"{method} {model_name} [{dataset}]  n={n}")
    for k in METRICS:
        print(f"  {k:<14s} {np.nanmean(acc[k]):+.4f}")


def run_gradcam(model_name, n, dataset="imagenets"):
    from xai_bench.registry import MODELS, METHODS
    data, path = load_cache(dataset)
    n = min(n, len(data))
    print(f"cache={path}")
    mw = MODELS.get(model_name)()
    mw.model.eval().to(DEVICE)
    gc = METHODS.get("grad_cam")(model_wrapper=mw, model=mw.model)

    def gen():
        for i in range(n):
            x = data[i]["image"].unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                t = int(mw(x).argmax())
            m = gc(x, t)
            m = (m[0] if m.ndim == 3 else m).detach().cpu().numpy()
            yield m, data[i]["metadata"]["bboxes"]

    _collect(gen(), n, model_name, "gradcam", dataset)


def run_hilrp(model_name, n, dataset="imagenets"):
    import timm
    import torch.nn.functional as F
    from xai_bench.methods.hilrp_method import _dispatch
    attribute = _dispatch(model_name)
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
    need_raw = any(abs(a - b) > 1e-6
                   for a, b in zip(cfg["mean"], (0.485, 0.456, 0.406)))

    def gen():
        for i in range(n):
            img = data[i]["image"]
            if need_raw:
                img = ((img * IS + IM).clamp(0, 1) - mean) / std
            x = img.unsqueeze(0).to(DEVICE)
            if size != 224:
                x = F.interpolate(x, size=(size, size), mode="bilinear",
                                  align_corners=False)
            r = attribute(model, x, gamma=0.25)
            yield r["pixel_map"].numpy(), data[i]["metadata"]["bboxes"]

    _collect(gen(), n, model_name, "hilrp", dataset)


def report(model_name, dataset="imagenets"):
    from scipy.stats import wilcoxon
    h = np.load(npz_name(model_name, "hilrp", dataset))
    g = np.load(npz_name(model_name, "gradcam", dataset))

    def ci(x):
        x = x[~np.isnan(x)]
        rng = np.random.RandomState(0)
        boot = [x[rng.randint(0, len(x), len(x))].mean() for _ in range(2000)]
        return x.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)

    print(f"=== {model_name} [{dataset}] ===")
    print(f"{'metric':<14s} {'HiLRP':>22s} {'Grad-CAM':>22s} {'paired p':>10s}")
    for k in METRICS:
        a, b = h[k], g[k]
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        am, alo, ahi = ci(a)
        bm, blo, bhi = ci(b)
        both = ~np.isnan(a) & ~np.isnan(b)
        try:
            p = wilcoxon(a[both], b[both]).pvalue
        except ValueError:          # all differences zero
            p = 1.0
        print(f"{k:<14s} {am:+.3f} [{alo:+.3f},{ahi:+.3f}] "
              f"{bm:+.3f} [{blo:+.3f},{bhi:+.3f}] {p:10.2e}")
    print("\nnote: `energy_excess` is the headline number (energy minus the "
          "region's area, which is what a uniform map scores).")
    print("note: `iou_best` favors smooth low-frequency maps over heavy-tailed "
          "ones at any fixed threshold; quote `ap` instead if only one "
          "threshold-free number is wanted.")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "report":
        report(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "imagenets")
    else:
        ds = sys.argv[4] if len(sys.argv) > 4 else "imagenets"
        {"gradcam": run_gradcam, "hilrp": run_hilrp}[mode](
            sys.argv[2], int(sys.argv[3]), ds)
