"""Dense-mask localization: Energy-Based Pointing Game (EBPG) and strict peak-in-mask PG.

Replaces the bounding-box Pointing Game with the pixel-perfect ImageNet-S silhouette.
  EBPG  = sum(|attr| * mask) / sum(|attr|)   (fraction of attribution mass on the object)
  PGd   = 1[ argmax|attr| in mask ]          (strict peak-in-silhouette pointing game)
Random-map EBPG baseline = mean mask area = 0.18.

CAM uses each backbone's terminal feature map, except the isotropic ViT, whose head
pools only the class token, so we read from the last block's input norm (as in the MS
reproduction). Gradient methods come from the registry. n=1000, single seed.
"""
import warnings
warnings.filterwarnings("ignore")
import json
import numpy as np
import torch
import torch.nn.functional as F

import xai_bench.methods, xai_bench.metrics, xai_bench.models, xai_bench.datasets  # noqa: F401
from xai_bench.registry import MODELS, METHODS
from xai_bench.methods.cam_methods import _to_nchw

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N = 1000
MODEL_ORDER = ["resnet50", "vit_base_patch16_224", "swin_base_patch4_window7_224",
               "pvt_v2_b2", "maxvit_small_tf_224", "mobilevitv2_100",
               "efficientvit_b1", "efficientvit_b2"]
GRAD_METHODS = ["saliency", "input_x_gradient", "smoothgrad", "vargrad"]
CAM_METHODS = ["grad_cam", "grad_cam_plus_plus"]


def cam_map(mw, x, target, plus_plus, vit):
    """CAM at input resolution. vit=True reads from blocks[-1].norm1 (spatial tokens)."""
    model = mw.model
    x = x.clone().detach().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    if vit:
        store = {}
        h = model.blocks[-1].norm1.register_forward_hook(lambda m, i, o: store.__setitem__("A", o))
        out = model(x)
        A_ = store["A"]; A_.retain_grad()
        model.zero_grad(set_to_none=True)
        out[0, int(target)].backward()
        h.remove()
        B, Nt, C = A_.shape
        n = Nt - 1; Hs = Ws = int(round(n ** 0.5))
        A = A_[:, 1:1 + Hs * Ws, :].transpose(1, 2).reshape(B, C, Hs, Ws)
        G = A_.grad[:, 1:1 + Hs * Ws, :].transpose(1, 2).reshape(B, C, Hs, Ws)
    else:
        feat = model.forward_features(x); feat.retain_grad()
        out = model.forward_head(feat)
        out[0, int(target)].backward()
        A = _to_nchw(feat, mw.cam_format); G = _to_nchw(feat.grad, mw.cam_format)
    if not plus_plus:
        w = G.mean(dim=(2, 3), keepdim=True)
    else:
        g2, g3 = G ** 2, G ** 3
        sa = A.sum(dim=(2, 3), keepdim=True)
        al = g2 / (2 * g2 + sa * g3 + 1e-7)
        al = torch.where(G != 0.0, al, torch.zeros_like(al))
        w = (al * F.relu(G)).sum(dim=(2, 3), keepdim=True)
    cam = F.relu((w * A).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
    return cam[0, 0].detach()


def scores(attr, mask):
    a = attr.abs().float()
    while a.dim() > 2:            # collapse channel/batch dims to [H, W]
        a = a.sum(0)
    tot = a.sum()
    if tot <= 0 or not torch.isfinite(tot):
        return None, None
    ebpg = float((a * mask).sum() / tot)
    idx = int(a.view(-1).argmax())
    pgd = float(mask.view(-1)[idx].item())
    return ebpg, pgd


def main():
    # Trusted, self-produced local cache; try safe load first, fall back if it holds
    # non-tensor record objects.
    try:
        imgs = torch.load("data/ImageNetS/cache_validation_1000.pt", weights_only=True)
    except Exception:
        imgs = torch.load("data/ImageNetS/cache_validation_1000.pt", weights_only=False)
    masks = torch.load("data/ImageNetS/cache_masks_1000.pt", weights_only=True)
    n = min(N, len(imgs))
    results = {}

    for model_name in MODEL_ORDER:
        mw = MODELS.get(model_name)()
        mw.model.to(DEVICE).eval()
        is_vit = model_name.startswith("vit")
        grad_fns = {m: METHODS.get(m)(model_wrapper=mw, model=mw.model) for m in GRAD_METHODS}
        acc = {m: {"ebpg": [], "pgd": []} for m in GRAD_METHODS + CAM_METHODS}

        for i in range(n):
            rec = imgs[i]
            img, target = rec["image"], int(rec["label"])
            mrec = masks[i]["mask"]
            if mrec is None:
                continue
            x = img.unsqueeze(0).to(DEVICE)
            mask = mrec.to(DEVICE)
            if mask.dim() == 3:              # RGB-encoded mask -> collapse channels
                mask = mask.amax(dim=-1)
            mask = (mask > 0).float()
            for m in GRAD_METHODS:
                attr = grad_fns[m](x, target=target)
                a = attr[0] if isinstance(attr, torch.Tensor) and attr.dim() == 3 else attr
                e, p = scores(a.detach(), mask)
                if e is not None:
                    acc[m]["ebpg"].append(e); acc[m]["pgd"].append(p)
            for m in CAM_METHODS:
                a = cam_map(mw, x, target, plus_plus=(m == "grad_cam_plus_plus"), vit=is_vit)
                e, p = scores(a, mask)
                if e is not None:
                    acc[m]["ebpg"].append(e); acc[m]["pgd"].append(p)

        for m in acc:
            e = np.array(acc[m]["ebpg"]); p = np.array(acc[m]["pgd"])
            results[f"{model_name}|{m}"] = dict(
                n=len(e), ebpg=float(e.mean()) if len(e) else float("nan"),
                pgd=float(p.mean()) if len(p) else float("nan"))
            print(f"{model_name:30s} {m:18s} n={len(e):4d} EBPG={results[f'{model_name}|{m}']['ebpg']:.3f} "
                  f"PGdense={results[f'{model_name}|{m}']['pgd']:.3f}", flush=True)
        json.dump(results, open("results/ebpg_results.json", "w"), indent=2)
        del mw
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("DONE -> results/ebpg_results.json")


if __name__ == "__main__":
    main()
