"""Proper ViT-B/16 Grad-CAM / Grad-CAM++ Max-Sensitivity + Pointing Game.

The generic forward_features Grad-CAM is degenerate on isotropic ViT: timm's
head pools only the CLS token, so the terminal spatial tokens receive zero
gradient and the CAM is all-zero. This script uses the standard ViT Grad-CAM
target layer, the input LayerNorm of the last transformer block (blocks[-1].norm1),
where the last attention still routes the CLS-logit gradient to every spatial
token, so the CAM is well defined. Same alpha-weight formulas as the generic
implementation; same Quantus metrics and MS protocol (nr_samples=3, lower_bound=0.2)
as the main run, so the ViT-B row is directly comparable.
"""
import warnings
warnings.filterwarnings("ignore")
import json
import numpy as np
import torch
import torch.nn.functional as F

import xai_bench.methods, xai_bench.metrics, xai_bench.models, xai_bench.datasets  # noqa: F401
from xai_bench.registry import METHODS, METRICS, MODELS, DATASETS

N_IMAGES = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_vit_cam(model, plus_plus):
    target_layer = model.blocks[-1].norm1
    store = {}

    def fwd_hook(_m, _i, o):
        store["A"] = o

    target_layer.register_forward_hook(fwd_hook)

    def method(inputs, target):
        inputs = inputs.clone().detach().to(DEVICE)
        model.zero_grad(set_to_none=True)
        out = model(inputs)
        A = store["A"]
        A.retain_grad()
        model.zero_grad(set_to_none=True)
        t = int(target)
        out[0, t].backward()
        G = A.grad
        B, N, C = A.shape
        n = N - 1
        Hs = Ws = int(round(n ** 0.5))
        A_sp = A[:, 1:1 + Hs * Ws, :].transpose(1, 2).reshape(B, C, Hs, Ws)
        G_sp = G[:, 1:1 + Hs * Ws, :].transpose(1, 2).reshape(B, C, Hs, Ws)
        if not plus_plus:
            w = G_sp.mean(dim=(2, 3), keepdim=True)
        else:
            g2, g3 = G_sp ** 2, G_sp ** 3
            sum_act = A_sp.sum(dim=(2, 3), keepdim=True)
            alpha = g2 / (2 * g2 + sum_act * g3 + 1e-7)
            alpha = torch.where(G_sp != 0.0, alpha, torch.zeros_like(alpha))
            w = (alpha * F.relu(G_sp)).sum(dim=(2, 3), keepdim=True)
        cam = F.relu((w * A_sp).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=inputs.shape[-2:], mode="bilinear", align_corners=False)
        return cam.squeeze(1)

    return method


def run(plus_plus):
    mw = MODELS.get("vit_base_patch16_224")()
    mw.model.to(DEVICE).eval()
    model = mw.model
    method = make_vit_cam(model, plus_plus)
    ds = DATASETS.get("imagenets_cached")(
        cache_path="data/ImageNetS/cache_validation_1000.pt", num_samples=N_IMAGES)

    ms_metric = METRICS.get("max_sensitivity")
    pg_metric = METRICS.get("pointing_game")
    ms_vals, pg_vals, nonzero = [], [], 0

    for i in range(len(ds)):
        img, target, meta = ds[i]
        img = img.to(DEVICE)
        attr = method(img.unsqueeze(0), target=target).detach()
        if float(attr.abs().max()) > 0:
            nonzero += 1

        def explain_func(model, inputs, targets, **kwargs):
            inputs_t = torch.tensor(inputs, device=DEVICE, dtype=torch.float32)
            tt = targets[0].item() if hasattr(targets, "__len__") else targets
            return method(inputs_t, target=tt).detach().cpu().numpy()

        ms = ms_metric(attr=attr[0], metadata=meta, model=model, image=img, target=target,
                       explain_func=explain_func,
                       metric_kwargs={"nr_samples": 3, "lower_bound": 0.2, "return_aggregate": False})
        pg = pg_metric(attr=attr[0], metadata=meta, model=model, image=img, target=target,
                       explain_func=explain_func)
        if ms is not None and not (isinstance(ms, float) and np.isnan(ms)):
            ms_vals.append(float(ms))
        if pg is not None and not (isinstance(pg, float) and np.isnan(pg)):
            pg_vals.append(float(pg))

    tag = "grad_cam_plus_plus" if plus_plus else "grad_cam"
    ms_vals = np.array(ms_vals)
    pg_vals = np.array(pg_vals)
    res = dict(method=tag, n_nonzero=nonzero, n_ms=len(ms_vals),
               ms_mean=float(ms_vals.mean()), ms_std=float(ms_vals.std()),
               ms_median=float(np.median(ms_vals)), ms_max=float(ms_vals.max()),
               pg_mean=float(pg_vals.mean()))
    print(f"[{tag}] nonzero maps {nonzero}/{len(ds)} | n_MS={len(ms_vals)} "
          f"MS mean={res['ms_mean']:.3f} std={res['ms_std']:.3f} median={res['ms_median']:.3f} "
          f"max={res['ms_max']:.2f} | PG={res['pg_mean']:.3f}")
    return res


if __name__ == "__main__":
    out = [run(False), run(True)]
    json.dump(out, open("results/vit_gradcam_ms.json", "w"), indent=2)
    print("saved results/vit_gradcam_ms.json")
