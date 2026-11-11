"""Shared helpers for the Neurocomputing revision experiments (NEUCOM-D-26-16590).

Every attribution here is computed exactly as in the benchmark registry
(xai_bench/methods/*), except that the final ``abs()`` applied by
``CaptumMethodWrapper`` is NOT applied, so the signed map is kept and the three
post-processing protocols (abs / positive-only / signed) can be scored from the
same attribution. ``to_registry_map`` reproduces the registry output exactly.
"""
import os
import sys
import zlib
import warnings

warnings.filterwarnings("ignore")
try:  # run below the user's own work on this shared laptop
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except Exception:
    pass
ROOT =os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import torch
import torch.nn.functional as F

import xai_bench.methods, xai_bench.metrics, xai_bench.models, xai_bench.datasets  # noqa: F401,E402
from xai_bench.registry import MODELS, METHODS  # noqa: E402
from xai_bench.models.timm_models import create_timm_model, _infer_arch  # noqa: E402
from xai_bench.methods.cam_methods import _to_nchw  # noqa: E402
from xai_bench.metrics.localization import _create_bbox_mask  # noqa: E402
from xai_bench.utils.seed import set_seed  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = os.path.join("results", "revision_neucom")
os.makedirs(OUT, exist_ok=True)

MAIN_MODELS = ["resnet50", "vit_base_patch16_224", "swin_base_patch4_window7_224",
               "pvt_v2_b2", "maxvit_small_tf_224", "mobilevitv2_100",
               "efficientvit_b1", "efficientvit_b2"]
SHORT = {"resnet50": "RN-50", "vit_base_patch16_224": "ViT-B/16",
         "swin_base_patch4_window7_224": "Swin-B", "pvt_v2_b2": "PVT-v2",
         "maxvit_small_tf_224": "MaxViT-S", "mobilevitv2_100": "MobileViT",
         "efficientvit_b1": "EffViT-B1", "efficientvit_b2": "EffViT-B2"}
# The main five-metric grid used the first 100 images for RN-50 and ViT-B/16 and the
# first 50 for the other six backbones (see results/*/config.json).
MAIN_GRID_N = {m: (100 if m in ("resnet50", "vit_base_patch16_224") else 50) for m in MAIN_MODELS}

SIGNED_METHODS = {"integrated_gradients", "input_x_gradient", "gradient_shap",
                  "occlusion", "lime", "attnlrp"}
LIME_PPE = int(os.environ.get("LIME_PPE", "25"))  # LIME forward passes per batch


def evaluation_seed(seed, model_name, method_name, image_idx):
    """Same derivation as xai_bench.runner.evaluation_seed."""
    return zlib.crc32(f"{seed}|{model_name}|{method_name}|{image_idx}".encode())


def load_data(n=1000):
    # Self-produced local cache; safe load first, as in scripts/bench/ebpg_eval.py.
    try:
        imgs = torch.load("data/ImageNetS/cache_validation_1000_v2.pt", weights_only=True)
    except Exception:
        imgs = torch.load("data/ImageNetS/cache_validation_1000_v2.pt", weights_only=False)
    masks = torch.load("data/ImageNetS/cache_masks_1000.pt", weights_only=True)
    out = []
    for i in range(min(n, len(imgs))):
        rec = imgs[i]
        m = masks[i]["mask"] if i in masks else None
        if m is not None:
            m = m.float()
            if m.dim() == 3:
                m = m.amax(dim=-1)
            m = (m > 0).float()
        assert masks[i]["label"] == rec["label"], f"mask/image misaligned at {i}"
        out.append(dict(idx=i, image=rec["image"], label=int(rec["label"]),
                        bboxes=rec["metadata"].get("bboxes", []), mask=m))
    return out


def load_model(name):
    if name in MODELS:
        mw = MODELS.get(name)()
    else:
        mw = create_timm_model(name, timm_name=name)
    mw.model.to(DEVICE).eval()
    for p in mw.model.parameters():
        p.requires_grad_(False)
    return mw


def n_params(mw):
    return sum(p.numel() for p in mw.model.parameters())


# ---------------------------------------------------------------- attributions
def _captum_signed(model, cls, x, target, **kw):
    attr = cls(model).attribute(x, target=target, **kw)
    return attr.sum(dim=1)[0].detach()   # [H, W], signed (registry then applies abs)


def cam_map(mw, x, target, plus_plus, alpha_floor=None, return_parts=False):
    """Grad-CAM / Grad-CAM++ exactly as the benchmark (closed-form alpha of
    Chattopadhay et al., Eq. 19, as in pytorch-grad-cam). Isotropic ViTs read the
    last block's input norm (spatial tokens). ``alpha_floor`` clamps |2 + S*g|
    away from zero (stabilised variant used only for the mechanism ablation)."""
    model = mw.model
    x = x.clone().detach().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    if mw.cam_format in ("tokens_cls", "tokens_nocls"):
        inner = model.model if hasattr(model, "in_scale") else model
        store = {}
        h = inner.blocks[-1].norm1.register_forward_hook(lambda m, i, o: store.__setitem__("A", o))
        with torch.enable_grad():
            out = model(x)
            A_ = store["A"]
            A_.retain_grad()
            out[0, int(target)].backward()
        h.remove()
        B, Nt, C = A_.shape
        start = 1 if mw.cam_format == "tokens_cls" else 0
        n = Nt - start
        Hs = Ws = int(round(n ** 0.5))
        A = A_[:, start:start + Hs * Ws, :].transpose(1, 2).reshape(B, C, Hs, Ws).detach()
        G = A_.grad[:, start:start + Hs * Ws, :].transpose(1, 2).reshape(B, C, Hs, Ws).detach()
    else:
        with torch.enable_grad():
            feat = model.forward_features(x)
            feat.retain_grad()
            out = model.forward_head(feat)
            out[0, int(target)].backward()
        A = _to_nchw(feat, mw.cam_format).detach()
        G = _to_nchw(feat.grad, mw.cam_format).detach()
    S = A.sum(dim=(2, 3), keepdim=True)
    if not plus_plus:
        w = G.mean(dim=(2, 3), keepdim=True)
    else:
        g2, g3 = G ** 2, G ** 3
        denom = 2 * g2 + S * g3
        if alpha_floor is not None:
            # denom = g^2 (2 + S g); keep the sign of (2 + S g) but bound its magnitude
            core = 2 + S * G
            core = torch.where(core.abs() < alpha_floor,
                               torch.where(core >= 0, torch.full_like(core, alpha_floor),
                                           torch.full_like(core, -alpha_floor)), core)
            denom = g2 * core
        al = g2 / (denom + 1e-7)
        al = torch.where(G != 0.0, al, torch.zeros_like(al))
        w = (al * F.relu(G)).sum(dim=(2, 3), keepdim=True)
    cam = F.relu((w * A).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
    if return_parts:
        return cam[0, 0].detach(), dict(A=A, G=G, S=S)
    return cam[0, 0].detach()


def signed_attribution(method, mw, x, target, registry_fn=None):
    """Return the 2-D map BEFORE the registry's final abs(): signed for IG, IxG,
    GradientSHAP, Occlusion, LIME and AttnLRP; non-negative by construction for the
    others (Saliency |grad|, SmoothGrad of |grad|, VarGrad, CAM, Rollout, RISE)."""
    from captum.attr import IntegratedGradients, InputXGradient, GradientShap, Occlusion, Lime
    from xai_bench.methods.captum_methods import _grid_feature_mask
    model = mw.model
    t = int(target)
    if method == "integrated_gradients":
        return _captum_signed(model, IntegratedGradients, x, t, n_steps=50, internal_batch_size=4)
    if method == "input_x_gradient":
        return _captum_signed(model, InputXGradient, x, t)
    if method == "gradient_shap":
        return _captum_signed(model, GradientShap, x, t, baselines=torch.zeros_like(x), n_samples=20)
    if method == "occlusion":
        C, H, W = x.shape[-3:]
        # perturbations_per_eval only batches the identical forward passes
        return _captum_signed(model, Occlusion, x, t, sliding_window_shapes=(C, 15, 15),
                              strides=(C, 8, 8), perturbations_per_eval=32)
    if method == "lime":
        # perturbations_per_eval batches the forward passes only; the samples are drawn
        # in the same order, so the map equals the one-at-a-time evaluation
        return _captum_signed(model, Lime, x, t, n_samples=100, feature_mask=_grid_feature_mask(x, cell=16),
                              perturbations_per_eval=LIME_PPE)
    if method == "grad_cam":
        return cam_map(mw, x, t, plus_plus=False)
    if method == "grad_cam_plus_plus":
        return cam_map(mw, x, t, plus_plus=True)
    # remaining methods: the registry output itself (non-negative, or AttnLRP signed)
    a = registry_fn(x, target=t)
    a = a.detach()
    while a.dim() > 2:
        a = a[0]
    return a.to(x.device)


def protocol_maps(a, method):
    """abs (benchmark protocol), pos (positive evidence only), signed (raw)."""
    a = a.float()
    if method in SIGNED_METHODS:
        return {"abs": a.abs(), "pos": a.clamp(min=0), "signed": a}
    return {"abs": a.abs(), "pos": a.abs(), "signed": a.abs()}


# ---------------------------------------------------------------- localisation
def bbox_pg(a, bboxes):
    """Quantus PointingGame semantics: hit if ANY max-valued pixel lies in the box."""
    if not bboxes:
        return np.nan
    mask = _create_bbox_mask(bboxes, tuple(a.shape)).to(a.device)
    if not mask.any():
        return np.nan
    mx = a == a.max()
    return float((mx & mask).any())


def ebpg(e, mask):
    """Energy-based pointing game on the dense silhouette for a non-negative map;
    for a signed map returns the NET evidence share sum(a*mask)/sum(|a|) in [-1, 1]."""
    tot = e.abs().sum()
    if tot <= 0 or not torch.isfinite(tot):
        return np.nan
    return float((e * mask).sum() / tot)


def dense_pg(a, mask):
    if not torch.isfinite(a).all() or a.max() == a.min():
        return np.nan
    return float(mask.view(-1)[int(a.view(-1).argmax())].item())


def is_degenerate(a):
    return bool((not torch.isfinite(a).all()) or float(a.max() - a.min()) == 0.0)
