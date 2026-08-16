"""Convergence of the sampled-Shapley reference, to justify m = 64 permutations.

The paper uses agreement with sampled permutation Shapley as its primary
yardstick at m = 64. That number needs evidence, not assertion. This script runs
the sampler once at m = M_MAX and reads the running estimate off the PREFIX of
the same permutation sequence, so the whole curve costs one run rather than one
run per m.

Two curves are reported:
  self-consistency  Spearman between the estimate at m and the estimate at
                    M_MAX, i.e. how settled the reference itself is.
  agreement         Spearman between HiLRP's per-segment scores and the
                    reference at m, i.e. whether the reported number has
                    stopped moving.

Protocol matches scripts/hilrp/shapley_pilot.py: SLIC segments, Gaussian-blur
infill (not black patches), target-logit value function. The HiLRP maps are
computed in a second pass, after the reference, because HiLRP's patches are
process-global.

Run:  <mri-diffuser python> scripts/hilrp/shapley_convergence.py [n_images] [m_max]
Out:  results/shapley_convergence/{curve.csv, shapley_convergence.png}
"""
import os
import sys
import warnings

import numpy as np
import torch
import torchvision.transforms.functional as TF

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import timm
from scipy.stats import spearmanr
from skimage.segmentation import slic

OUT = os.path.join("results", "shapley_convergence")
os.makedirs(OUT, exist_ok=True)

CACHE = os.path.join("data", "ImageNetS", "cache_validation_1000.pt")
MODEL = "swin_tiny_patch4_window7_224"
N_SEGMENTS = 24
BLUR_KERNEL = 51
BATCH = 32
GAMMA = 0.25
M_GRID = [4, 8, 16, 32, 64, 128]

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_cache(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def make_regions(img_norm):
    raw = (img_norm * STD + MEAN).clamp(0, 1).permute(1, 2, 0).numpy()
    return slic(raw, n_segments=N_SEGMENTS, compactness=20, start_label=0)


def shapley_marginals(model, img_norm, labels, target, n_perms, device):
    """Per-permutation marginal contributions, shape [n_perms, K].

    Returned unaveraged so that a prefix mean gives the estimate at any m.
    """
    K = labels.max() + 1
    x = img_norm.to(device)
    blurred = TF.gaussian_blur(x, BLUR_KERNEL).to(device)
    masks = torch.stack([torch.from_numpy(labels == k) for k in range(K)]).to(device)

    vals = np.zeros((n_perms, K), dtype=np.float64)
    rng = np.random.RandomState(0)
    for p in range(n_perms):
        perm = rng.permutation(K)
        imgs = torch.empty(K + 1, *x.shape, device=device)
        cur = blurred.clone()
        imgs[0] = cur
        for i, k in enumerate(perm):
            cur = torch.where(masks[k].unsqueeze(0), x, cur)
            imgs[i + 1] = cur
        logits = []
        with torch.no_grad():
            for b in range(0, K + 1, BATCH):
                logits.append(model(imgs[b:b + BATCH])[:, target])
        vals[p, perm] = np.diff(torch.cat(logits).double().cpu().numpy())
    return vals


def region_sums(pixel_map, labels):
    K = labels.max() + 1
    return np.array([pixel_map[labels == k].sum() for k in range(K)])


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    m_max = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    grid = [m for m in M_GRID if m <= m_max]
    data = load_cache(CACHE)
    n = min(n, len(data))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- pass 1: the reference, on vanilla classes -------------------------
    print(f"=== pass 1: Shapley reference, {n} images, m_max={m_max} ===")
    vanilla = timm.create_model(MODEL, pretrained=True).to(device).eval()
    per_image = []
    for i in range(n):
        img = data[i]["image"]
        labels = make_regions(img)
        x = img.unsqueeze(0).to(device)
        with torch.no_grad():
            target = int(vanilla(x)[0].argmax())
        vals = shapley_marginals(vanilla, img, labels, target, m_max, device)
        per_image.append(dict(idx=i, labels=labels, vals=vals, target=target))
        print(f"  image {i + 1}/{n}", flush=True)
    del vanilla
    torch.cuda.empty_cache()

    # ---- pass 2: HiLRP -----------------------------------------------------
    print("=== pass 2: HiLRP ===")
    from xai_bench.methods.hilrp.swin_lxt import attribute_swin
    model = timm.create_model(MODEL, pretrained=True).to(device).eval()
    for rec in per_image:
        x = data[rec["idx"]]["image"].unsqueeze(0).to(device)
        pm = attribute_swin(model, x, target=rec["target"], gamma=GAMMA)["pixel_map"].numpy()
        rec["hilrp"] = region_sums(pm, rec["labels"])
    del model
    torch.cuda.empty_cache()

    # ---- curves ------------------------------------------------------------
    rows = []
    for m in grid:
        selfc, agree = [], []
        for rec in per_image:
            est = rec["vals"][:m].mean(0)
            ref = rec["vals"].mean(0)
            if est.std() > 1e-12 and ref.std() > 1e-12:
                selfc.append(spearmanr(est, ref).correlation)
            if est.std() > 1e-12 and rec["hilrp"].std() > 1e-12:
                agree.append(spearmanr(rec["hilrp"], est).correlation)
        rows.append((m, float(np.nanmean(selfc)), float(np.nanstd(selfc)),
                     float(np.nanmean(agree)), float(np.nanstd(agree) / np.sqrt(max(len(agree), 1)))))

    print("\n" + "=" * 62)
    print(f"{'m':>5s} {'self-consistency':>20s} {'HiLRP agreement':>20s}")
    print("=" * 62)
    for m, sc, scs, ag, ags in rows:
        print(f"{m:>5d} {sc:>15.3f}+-{scs:.3f} {ag:>15.3f}+-{ags:.3f}")
    print("=" * 62)

    with open(os.path.join(OUT, "curve.csv"), "w") as f:
        f.write("m,self_consistency,self_consistency_std,agreement,agreement_sem\n")
        for m, sc, scs, ag, ags in rows:
            f.write(f"{m},{sc:.4f},{scs:.4f},{ag:.4f},{ags:.4f}\n")

    make_figure(rows, n)
    print(f"\nwrote {OUT}/curve.csv, shapley_convergence.png")


def make_figure(rows, n):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))                   # <repo>/scripts
    from figstyle import apply_paper_style, save, COL_W, BLUE, RED
    apply_paper_style()
    import matplotlib.pyplot as plt

    ms = [r[0] for r in rows]
    sc = [r[1] for r in rows]
    ag = [r[3] for r in rows]
    ags = [r[4] for r in rows]

    # Column width, so the 7-8 pt type here is 7-8 pt on the printed page.
    fig, ax = plt.subplots(1, 2, figsize=(COL_W, 1.95))
    ax[0].plot(ms, sc, "o-", color=BLUE, ms=3, lw=1.2)
    ax[0].set_ylabel(r"Spearman vs. $m=%d$" % ms[-1])
    ax[0].set_title("Reference Self-Consistency")

    ax[1].errorbar(ms, ag, yerr=ags, fmt="o-", color=RED, capsize=2, ms=3, lw=1.2)
    ax[1].set_ylabel("Spearman")
    ax[1].set_title("HiLRP Agreement")

    for a in ax:
        a.set_xscale("log", base=2)
        a.set_xticks(ms); a.set_xticklabels(ms)
        a.set_xlabel("Permutations $m$")
        a.axvline(64, color="k", ls=":", lw=0.8)
        a.grid(alpha=0.3, lw=0.5)

    fig.tight_layout(pad=0.3)
    # suptitle after tight_layout, lifted clear of the panel titles; bbox_inches
    # "tight" grows the canvas to include it.
    fig.suptitle(f"Sampled-Shapley Convergence (Swin-T, {n} images, "
                 f"{N_SEGMENTS} SLIC segments)", y=1.06)
    fig.savefig(os.path.join(OUT, "shapley_convergence.png"), dpi=400,
                bbox_inches="tight")
    save(fig, "figures/hilrp/shapley_convergence")


def replot():
    """Rebuild the figure from results/shapley_convergence/curve.csv.

    The curve is expensive (one full sampled-Shapley run) and already cached, so
    restyling the figure must not require recomputing it.
    """
    import csv
    path = os.path.join(OUT, "curve.csv")
    with open(path) as f:
        rows = [(int(r["m"]), float(r["self_consistency"]),
                 float(r["self_consistency_std"]), float(r["agreement"]),
                 float(r["agreement_sem"])) for r in csv.DictReader(f)]
    n = int(os.environ.get("SHAPLEY_N", 20))    # n images the cached curve used
    make_figure(rows, n)


if __name__ == "__main__":
    if "--replot" in sys.argv:
        replot()
    else:
        main()
