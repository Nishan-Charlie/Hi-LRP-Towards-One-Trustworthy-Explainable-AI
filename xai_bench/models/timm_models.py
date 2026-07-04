import torch
import torch.nn as nn
import timm
from typing import Dict, Optional, Tuple

from xai_bench.registry import MODELS

# The datasets bake ImageNet normalization into every cached/streamed image.
# Not every timm model expects those stats: the mobilevit family expects raw
# [0,1] inputs (pretrained_cfg mean 0, std 1). Feeding it ImageNet-normalized
# tensors is out-of-distribution (top-1 collapses to ~0), which contaminated the
# original mobilevitv2 benchmark rows. The adapter below converts between the
# two stats with one exact per-channel affine, INSIDE the model, so every call
# path (forward, forward_features for CAM, captum, quantus perturbations) sees
# the correct input space while the bench keeps one uniform image cache.
_BENCH_MEAN = (0.485, 0.456, 0.406)
_BENCH_STD = (0.229, 0.224, 0.225)


class InputStatsAdapter(nn.Module):
    """Wraps a timm model whose expected input stats differ from the bench's.

    x_model = ((x * bench_std + bench_mean) - model_mean) / model_std
            = x * (bench_std / model_std) + (bench_mean - model_mean) / model_std
    """

    def __init__(self, model: nn.Module, model_mean, model_std):
        super().__init__()
        self.model = model
        bm = torch.tensor(_BENCH_MEAN).view(1, 3, 1, 1)
        bs = torch.tensor(_BENCH_STD).view(1, 3, 1, 1)
        mm = torch.tensor(model_mean).view(1, 3, 1, 1)
        ms = torch.tensor(model_std).view(1, 3, 1, 1)
        self.register_buffer("in_scale", bs / ms)
        self.register_buffer("in_shift", (bm - mm) / ms)

    def _adapt(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.in_scale + self.in_shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(self._adapt(x))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward_features(self._adapt(x))

    def forward_head(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model.forward_head(x, **kwargs)

    def __getattr__(self, name):
        # delegate anything else (stages, blocks, layer4, pretrained_cfg, ...)
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("model"), name)


def _needs_stats_adapter(model: nn.Module) -> Optional[Tuple[tuple, tuple]]:
    cfg = getattr(model, "pretrained_cfg", None) or getattr(model, "default_cfg", {})
    mean = tuple(cfg.get("mean", _BENCH_MEAN))
    std = tuple(cfg.get("std", _BENCH_STD))
    if all(abs(a - b) < 1e-6 for a, b in zip(mean, _BENCH_MEAN)) and \
       all(abs(a - b) < 1e-6 for a, b in zip(std, _BENCH_STD)):
        return None
    return mean, std


# CAM feature formats produced by a backbone's ``forward_features``:
#   tokens_cls   : [B, N, C] flat token sequence with a leading class token (classic ViT/DeiT/BEiT)
#   tokens_nocls : [B, N, C] flat token sequence, no class token
#   nchw         : [B, C, H, W] channels-first spatial map (CNNs, MaxViT, MobileViT, EfficientViT)
#   nhwc         : [B, H, W, C] channels-last spatial map (Swin v1)
CAM_FORMATS = ("tokens_cls", "tokens_nocls", "nchw", "nhwc")


class ModelWrapper(nn.Module):
    """A thin wrapper that standardizes a backbone for attribution methods.

    Two attributes drive method routing:
      * ``cam_format``        -- how ``forward_features`` lays out its output, so the
                                 generic Grad-CAM can reshape it to [B, C, H, W].
      * ``supports_attention``-- whether attention-native methods (rollout) apply.
                                 Only true for classic ViTs with a global-attention
                                 CLS token; false for windowed/grid/linear attention
                                 (Swin, MaxViT, MobileViT, EfficientViT) and CNNs.
    """

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        cam_format: str = "nchw",
        supports_attention: bool = False,
    ):
        super().__init__()
        self.model = model
        self.model_name = model_name
        self.cam_format = cam_format
        self.supports_attention = supports_attention
        # Back-compat alias: "is this a transformer with a token sequence?"
        self.is_vit = cam_format in ("tokens_cls", "tokens_nocls")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def has_feature_api(self) -> bool:
        """The generic Grad-CAM uses forward_features + forward_head, which every
        timm backbone in this suite implements."""
        return hasattr(self.model, "forward_features") and hasattr(self.model, "forward_head")

    def get_target_layer(self) -> Optional[nn.Module]:
        """Legacy single-layer accessor (kept for any Captum LayerGradCam path).
        The generic Grad-CAM no longer needs it."""
        if self.is_vit:
            if hasattr(self.model, "blocks"):
                return self.model.blocks[-1].norm1
            if hasattr(self.model, "norm"):
                return self.model.norm
            return None
        if hasattr(self.model, "layer4"):       # ResNets
            return self.model.layer4[-1]
        if hasattr(self.model, "stages"):        # ConvNeXt / hierarchical
            return self.model.stages[-1]
        return None

    def get_attention_maps(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        raise NotImplementedError("Attention maps retrieval requires specific hooks.")


def _infer_arch(name: str) -> Tuple[str, bool]:
    """Fallback (cam_format, supports_attention) inferred from a timm model name."""
    n = name.lower()
    if any(k in n for k in ("maxvit", "mobilevit", "efficientvit", "levit")):
        return "nchw", False                 # conv-hybrid / efficient attention
    if n.startswith("swinv2_cr") or n.startswith("pvt"):
        return "nchw", False
    if n.startswith("swin"):
        return "nhwc", False                 # Swin v1 emits channels-last
    if any(k in n for k in ("vit", "deit", "beit", "eva", "cait")):
        return "tokens_cls", True            # classic global-attention ViT
    return "nchw", False                      # CNNs


def create_timm_model(
    model_name: str,
    pretrained: bool = True,
    timm_name: Optional[str] = None,
    cam_format: Optional[str] = None,
    supports_attention: Optional[bool] = None,
    **kwargs,
) -> ModelWrapper:
    name = timm_name or model_name
    model = timm.create_model(name, pretrained=pretrained, **kwargs)
    stats = _needs_stats_adapter(model)
    if stats is not None:
        model = InputStatsAdapter(model, *stats)
    if cam_format is None or supports_attention is None:
        inferred_fmt, inferred_attn = _infer_arch(name)
        cam_format = cam_format or inferred_fmt
        supports_attention = inferred_attn if supports_attention is None else supports_attention
    return ModelWrapper(
        model, model_name, cam_format=cam_format, supports_attention=supports_attention
    )


# ── Classic backbones used by the controlled cross-architecture results ───────
@MODELS.register("vit_base_patch16_224")
def vit_base_patch16_224(**kwargs):
    return create_timm_model(
        "vit_base_patch16_224", cam_format="tokens_cls", supports_attention=True, **kwargs
    )

@MODELS.register("deit_base_patch16_224")
def deit_base_patch16_224(**kwargs):
    return create_timm_model(
        "deit_base_patch16_224", cam_format="tokens_cls", supports_attention=True, **kwargs
    )

@MODELS.register("resnet50")
def resnet50(**kwargs):
    return create_timm_model("resnet50", cam_format="nchw", supports_attention=False, **kwargs)

@MODELS.register("convnext_base")
def convnext_base(**kwargs):
    return create_timm_model("convnext_base", cam_format="nchw", supports_attention=False, **kwargs)


# ── Efficient / hierarchical / hybrid Vision Transformer suite (10 backbones) ─
# These use windowed (Swin), multi-axis (MaxViT), convolution-hybrid (MobileViT)
# or linear/cascaded (EfficientViT) attention -- there is no global CLS-token
# attention, so attention-native methods do not apply (supports_attention=False).
# Grad-CAM works via the generic forward_features path using the cam_format below.
# All build at 224x224 with a public ImageNet-1k head. Value = (timm tag, cam_format).
EFFICIENT_VIT_SUITE: Dict[str, Tuple[str, str]] = {
    # Swin -- shifted-window attention, channels-last feature map
    "swin_tiny_patch4_window7_224":  ("swin_tiny_patch4_window7_224",  "nhwc"),
    "swin_small_patch4_window7_224": ("swin_small_patch4_window7_224", "nhwc"),
    "swin_base_patch4_window7_224":  ("swin_base_patch4_window7_224",  "nhwc"),
    # MaxViT -- multi-axis (block + grid) attention, channels-first
    "maxvit_tiny_tf_224":            ("maxvit_tiny_tf_224",            "nchw"),
    "maxvit_small_tf_224":           ("maxvit_small_tf_224",           "nchw"),
    # MobileViT -- convolution-transformer hybrid, channels-first
    "mobilevit_s":                   ("mobilevit_s",                   "nchw"),
    "mobilevit_xs":                  ("mobilevit_xs",                  "nchw"),
    "mobilevitv2_100":               ("mobilevitv2_100",               "nchw"),
    # EfficientViT -- multi-scale linear attention (MIT family)
    "efficientvit_b1":               ("efficientvit_b1",               "nchw"),
    "efficientvit_b2":               ("efficientvit_b2",               "nchw"),
    # PVT v2 -- spatial-reduction attention (HiLRP third-corollary leg)
    "pvt_v2_b2":                     ("pvt_v2_b2",                     "nchw"),
}


def _make_efficient_factory(timm_name: str, cam_format: str):
    def _factory(**kwargs):
        return create_timm_model(
            timm_name,
            timm_name=timm_name,
            cam_format=cam_format,
            supports_attention=False,
            **kwargs,
        )
    return _factory


for _key, (_tag, _fmt) in EFFICIENT_VIT_SUITE.items():
    if _key not in MODELS:
        MODELS.add(_key, _make_efficient_factory(_tag, _fmt))
