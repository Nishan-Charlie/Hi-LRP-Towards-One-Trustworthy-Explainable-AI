"""Model wrapper shared by every backbone in the zoo.

A WrappedModel exposes a single contract the attribution methods rely on:
    model(x) -> class logits [B, num_classes]
plus the metadata Grad-CAM and Attention Rollout need (target layer, the
[tokens]->[grid] reshape, patch-grid size, CLS handling).

Heads
-----
- ``builtin``      : the timm model already has a trained classifier (e.g. the
                     ``*_ft_in1k`` / supervised checkpoints). Used directly.
- ``linear_probe`` : self-supervised backbones (DINOv2, MAE, BEiT3 pretrain)
                     expose features only; a linear probe must be fitted to get
                     class logits. Registered but flagged ``ready=False`` until a
                     probe checkpoint is supplied.
- ``clip_zeroshot``: CLIP/EVA-CLIP backbones produce logits from text prompts.
                     Registered but flagged ``ready=False`` for v1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch
import torch.nn as nn


@dataclass
class ModelSpec:
    name: str                         # benchmark display name, e.g. "ViT-B/16-sup"
    timm_name: str                    # timm identifier
    family: str = "vit"               # vit|deit|beit|convit|cait|swin|maxvit|cnn
    head: str = "builtin"             # builtin|linear_probe|clip_zeroshot
    ready: bool = True                # False => skipped by the runner with a note
    # Grad-CAM metadata
    target_layer: Callable[[nn.Module], nn.Module] = None  # model -> layer
    reshape: str = "vit"              # "vit" | "swin" | "none"  (token->grid)
    patch_h: int = 14
    patch_w: int = 14
    n_skip: int = 1                   # tokens to drop before the patch grid (CLS [+reg])
    has_cls: bool = True
    do_rollout: bool = True           # attention rollout supported
    input_size: int = 224
    notes: str = ""


class WrappedModel(nn.Module):
    def __init__(self, module: nn.Module, spec: ModelSpec, device: str = "cpu"):
        super().__init__()
        self.module = module.to(device).eval()
        self.spec = spec
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)

    @property
    def num_params_m(self) -> float:
        return sum(p.numel() for p in self.module.parameters()) / 1e6

    def target_layers(self) -> List[nn.Module]:
        if self.spec.target_layer is None:
            raise ValueError(f"{self.spec.name}: no target_layer defined for Grad-CAM")
        layer = self.spec.target_layer(self.module)
        return layer if isinstance(layer, list) else [layer]

    def reshape_transform(self) -> Optional[Callable]:
        """Return a fn mapping a hooked activation -> [B, C, h, w], or None."""
        s = self.spec
        if s.reshape == "none":
            return None
        if s.reshape == "swin":
            def _swin(t: torch.Tensor) -> torch.Tensor:
                B, N, C = t.shape
                return t.reshape(B, s.patch_h, s.patch_w, C).permute(0, 3, 1, 2)
            return _swin

        # default: ViT-style token sequence [B, N(+skip), C]
        def _vit(t: torch.Tensor) -> torch.Tensor:
            r = t[:, s.n_skip:s.n_skip + s.patch_h * s.patch_w, :]
            B, N, C = r.shape
            return r.reshape(B, s.patch_h, s.patch_w, C).permute(0, 3, 1, 2)
        return _vit
