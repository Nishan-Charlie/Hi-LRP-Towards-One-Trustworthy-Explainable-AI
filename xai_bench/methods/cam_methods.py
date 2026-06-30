import torch
import torch.nn.functional as F

from xai_bench.registry import METHODS


def _to_nchw(feat: torch.Tensor, cam_format: str) -> torch.Tensor:
    """Reshape a backbone's ``forward_features`` output to [B, C, H, W]."""
    if cam_format == "nchw":
        return feat
    if cam_format == "nhwc":                          # Swin: [B, H, W, C]
        return feat.permute(0, 3, 1, 2).contiguous()
    if cam_format in ("tokens_cls", "tokens_nocls"):  # classic ViT: [B, N, C]
        B, N, C = feat.shape
        start = 1 if cam_format == "tokens_cls" else 0
        n = N - start
        H = W = int(round(n ** 0.5))
        tokens = feat[:, start:start + H * W, :]
        return tokens.transpose(1, 2).reshape(B, C, H, W)
    raise ValueError(f"unknown cam_format '{cam_format}'")


def _generic_gradcam(model_wrapper, inputs: torch.Tensor, target: int,
                     plus_plus: bool = False) -> torch.Tensor:
    """Architecture-agnostic Grad-CAM / Grad-CAM++.

    Uses timm's ``forward_features`` + ``forward_head`` so a single code path
    covers classic ViTs (token sequence), CNNs and MaxViT/MobileViT/EfficientViT
    (channels-first maps) and Swin (channels-last maps) without per-model hooks.
    """
    model = model_wrapper.model
    cam_format = getattr(model_wrapper, "cam_format", "nchw")

    inputs = inputs.clone().detach().requires_grad_(True)
    model.zero_grad(set_to_none=True)

    feat = model.forward_features(inputs)
    feat.retain_grad()
    out = model.forward_head(feat)
    out[0, target].backward()

    A = _to_nchw(feat, cam_format)
    G = _to_nchw(feat.grad, cam_format)

    if not plus_plus:
        weights = G.mean(dim=(2, 3), keepdim=True)
    else:
        g2, g3 = G ** 2, G ** 3
        sum_act = A.sum(dim=(2, 3), keepdim=True)
        eps = 1e-7
        alpha = g2 / (2 * g2 + sum_act * g3 + eps)
        alpha = torch.where(G != 0.0, alpha, torch.zeros_like(alpha))
        weights = (alpha * F.relu(G)).sum(dim=(2, 3), keepdim=True)

    cam = F.relu((weights * A).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=inputs.shape[-2:], mode="bilinear", align_corners=False)
    return cam.squeeze(1)


@METHODS.register("grad_cam")
def get_gradcam(model_wrapper, **kwargs):
    def method(inputs, target):
        return _generic_gradcam(model_wrapper, inputs, target, plus_plus=False)
    return method


@METHODS.register("grad_cam_plus_plus")
def get_gradcam_plus_plus(model_wrapper, **kwargs):
    def method(inputs, target):
        return _generic_gradcam(model_wrapper, inputs, target, plus_plus=True)
    return method
