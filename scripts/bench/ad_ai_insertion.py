"""Average Drop / Average Increase / Deletion / Insertion, with a random control.

These are the perturbation metrics the CAM literature reports, and the paper
argues elsewhere that the deletion family cannot rank attributions on modern
ViTs. Reporting them without that argument would be incomplete; asserting the
argument without reporting them would look like avoiding the metrics we lose.
So we report all four AND the control that shows what they measure.

Definitions, following the CAM literature:
  AD  mean over images of max(0, p - p') / p, where p is the target softmax
      probability on the full image and p' the probability when only the top
      20% most-attributed pixels are kept (the rest zeroed). Lower is better.
  AI  fraction of images where p' > p. Higher is better.
  Del AUC of the target probability as pixels are removed most-important-first.
      Lower is better.
  Ins AUC of the target probability as pixels are added most-important-first.
      Higher is better.

The control is a RANDOM map that consults neither model nor image, generated at
a coarse and a fine resolution and upsampled. Yuan et al. (TIP 2025) report that
such a map beats real methods on Deletion for CNNs; this script tests the same
claim on ViTs, on our own backbones.

Two-pass protocol: all vanilla methods run before HiLRP installs its global
class patches.

Run:  <mri-diffuser python> scripts/bench/ad_ai_insertion.py [n_images]
Out:  results/ad_ai/{per_image.csv, summary.csv}
"""
import os
import sys
import warnings

import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import timm

OUT = os.path.join("results", "ad_ai")
os.makedirs(OUT, exist_ok=True)

CACHE = os.path.join("data", "ImageNetS", "cache_validation_1000.pt")
GAMMA = 0.25
CONV_GAMMA = {"efficientvit": 0.05, "mobilevit": 0.1}
MODELS_ = ["swin_base_patch4_window7_224", "pvt_v2_b2", "efficientvit_b2"]

KEEP_FRAC = 0.20          # AD/AI keep the top 20% of pixels
STEP_FRAC = 0.036         # deletion/insertion step, ~3.6% of pixels
SEED = 0

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_cache(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def adapt_input(img, model):
    mean = torch.tensor(model.default_cfg["mean"]).view(3, 1, 1)
    std = torch.tensor(model.default_cfg["std"]).view(3, 1, 1)
    raw = (img * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)
    return (raw - mean) / std


def conv_gamma_for(name):
    for p, g in CONV_GAMMA.items():
        if name.startswith(p):
            return g
    return None


def get_attributor(name):
    if name.startswith("swin"):
        from xai_bench.methods.hilrp.swin_lxt import attribute_swin
        return attribute_swin
    if name.startswith("pvt"):
        from xai_bench.methods.hilrp.pvt_lxt import attribute_pvt
        return attribute_pvt
    if name.startswith("efficientvit"):
        from xai_bench.methods.hilrp.efficientvit_lxt import attribute_efficientvit
        return attribute_efficientvit
    raise ValueError(name)


@torch.no_grad()
def probs_for(model, batch, target, bs=32):
    """Target-class softmax probability for a batch of perturbed images."""
    out = []
    for i in range(0, batch.shape[0], bs):
        logits = model(batch[i:i + bs])
        out.append(F.softmax(logits, dim=-1)[:, target])
    return torch.cat(out)


def curves(model, x, order, target, zero):
    """Deletion and insertion probability curves.

    `order` is a flat pixel ranking, most important first. `zero` is the
    baseline image (the model's own zero point after normalization).
    """
    n_pix = order.numel()
    step = max(1, int(round(STEP_FRAC * n_pix)))
    cuts = list(range(0, n_pix + 1, step))
    if cuts[-1] != n_pix:
        cuts.append(n_pix)

    dele, ins = [], []
    for k in cuts:
        md = torch.ones(n_pix, device=x.device)
        md[order[:k]] = 0.0                       # remove the top-k
        dele.append(md.view(1, 1, *x.shape[-2:]))
        mi = torch.zeros(n_pix, device=x.device)
        mi[order[:k]] = 1.0                       # insert the top-k
        ins.append(mi.view(1, 1, *x.shape[-2:]))

    dmask = torch.cat(dele); imask = torch.cat(ins)
    dbatch = x * dmask + zero * (1 - dmask)
    ibatch = x * imask + zero * (1 - imask)
    pd = probs_for(model, dbatch, target).cpu().numpy()
    pi = probs_for(model, ibatch, target).cpu().numpy()
    xs = np.array(cuts) / n_pix
    return float(np.trapz(pd, xs)), float(np.trapz(pi, xs))


def evaluate_map(model, x, rel, target, p_full, zero):
    """AD, AI, Deletion AUC, Insertion AUC for one attribution map."""
    flat = torch.as_tensor(rel, device=x.device, dtype=torch.float32).flatten()
    order = torch.argsort(flat, descending=True)

    k = int(round(KEEP_FRAC * flat.numel()))
    keep = torch.zeros_like(flat)
    keep[order[:k]] = 1.0
    kept = x * keep.view(1, 1, *x.shape[-2:]) + zero * (1 - keep.view(1, 1, *x.shape[-2:]))
    p_keep = float(probs_for(model, kept, target)[0])

    ad = max(0.0, p_full - p_keep) / max(p_full, 1e-12)
    ai = 1.0 if p_keep > p_full else 0.0
    dele, ins = curves(model, x, order, target, zero)
    return ad, ai, dele, ins


def random_maps(shape, rng):
    """Model-free and image-free controls at two granularities."""
    h, w = shape
    out = {}
    for res, label in ((h, "random-fine"), (14, "random-coarse")):
        m = torch.from_numpy(rng.random((res, res)).astype(np.float32))
        if res != h:
            m = F.interpolate(m[None, None], size=(h, w), mode="bilinear",
                              align_corners=False)[0, 0]
        out[label] = m
    return out


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    data = load_cache(CACHE)
    n = min(n, len(data))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []

    # ---- pass 1: Grad-CAM and the random controls (no HiLRP patches yet) ----
    print("=== pass 1: vanilla (Grad-CAM + random controls) ===")
    from xai_bench.models.timm_models import create_timm_model
    from xai_bench.registry import METHODS
    import xai_bench.methods  # noqa: F401

    for name in MODELS_:
        print(f"  {name}")
        wrapper = create_timm_model(name)
        net = wrapper.model.to(dev).eval()
        cam = METHODS.get("grad_cam")(model_wrapper=wrapper, model=net)
        rng = np.random.default_rng(SEED)
        for i in range(n):
            x = data[i]["image"].unsqueeze(0).to(dev)
            zero = torch.zeros_like(x)
            with torch.no_grad():
                target = int(net(x)[0].argmax())
                p_full = float(F.softmax(net(x)[0], -1)[target])
            maps = {"grad_cam": cam(x, target).squeeze().detach().cpu()}
            maps.update(random_maps(x.shape[-2:], rng))
            for meth, m in maps.items():
                ad, ai, d, ins = evaluate_map(net, x, m, target, p_full, zero)
                rows.append(dict(model=name, method=meth, idx=i,
                                 ad=ad, ai=ai, deletion=d, insertion=ins))
            if (i + 1) % 25 == 0:
                print(f"    {i + 1} images")
        del cam, net, wrapper
        torch.cuda.empty_cache()

    # ---- pass 2: HiLRP -----------------------------------------------------
    print("\n=== pass 2: HiLRP ===")
    for name in MODELS_:
        print(f"  {name}")
        model = timm.create_model(name, pretrained=True).to(dev).eval()
        attribute = get_attributor(name)
        kw = dict(gamma=GAMMA)
        cg = conv_gamma_for(name)
        if cg is not None:
            kw["conv_gamma"] = cg
        for i in range(n):
            x = adapt_input(data[i]["image"], model).unsqueeze(0).to(dev)
            zero = torch.zeros_like(x)
            with torch.no_grad():
                target = int(model(x)[0].argmax())
                p_full = float(F.softmax(model(x)[0], -1)[target])
            rel = attribute(model, x, target=target, **kw)["pixel_map"]
            ad, ai, d, ins = evaluate_map(model, x, rel, target, p_full, zero)
            rows.append(dict(model=name, method="hilrp", idx=i,
                             ad=ad, ai=ai, deletion=d, insertion=ins))
            if (i + 1) % 25 == 0:
                print(f"    {i + 1} images")
        del model
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------- output
    fields = ["model", "method", "idx", "ad", "ai", "deletion", "insertion"]
    with open(os.path.join(OUT, "per_image.csv"), "w") as f:
        f.write(",".join(fields) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in fields) + "\n")

    order = ["hilrp", "grad_cam", "random-coarse", "random-fine"]
    print("\n" + "=" * 78)
    print(f"{'model':<30s} {'method':<15s} {'AD':>7s} {'AI':>7s} {'Del':>8s} {'Ins':>8s}")
    print("=" * 78)
    with open(os.path.join(OUT, "summary.csv"), "w") as f:
        f.write("model,method,n,ad,ai,deletion,insertion\n")
        for m in MODELS_:
            for meth in order:
                sel = [r for r in rows if r["model"] == m and r["method"] == meth]
                if not sel:
                    continue
                g = lambda k: float(np.mean([r[k] for r in sel]))
                print(f"{m:<30s} {meth:<15s} {g('ad')*100:>7.1f} {g('ai')*100:>7.1f} "
                      f"{g('deletion'):>8.4f} {g('insertion'):>8.4f}")
                f.write(f"{m},{meth},{len(sel)},{g('ad'):.4f},{g('ai'):.4f},"
                        f"{g('deletion'):.4f},{g('insertion'):.4f}\n")
    print("=" * 78)
    print(f"\nwrote {OUT}/per_image.csv, summary.csv")


if __name__ == "__main__":
    main()
