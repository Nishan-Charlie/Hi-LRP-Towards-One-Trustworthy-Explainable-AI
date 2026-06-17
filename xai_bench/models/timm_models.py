import torch
import torch.nn as nn
import timm
from typing import Any, Dict, Optional, Tuple

from xai_bench.registry import MODELS


class ModelWrapper(nn.Module):
    """A thin wrapper around a model that standardizes access to its internals
    for attribution methods (like Grad-CAM target layers).
    """

    def __init__(self, model: nn.Module, model_name: str, is_vit: bool = False):
        super().__init__()
        self.model = model
        self.model_name = model_name
        self.is_vit = is_vit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_target_layer(self) -> Optional[nn.Module]:
        """Returns the default target layer for CAM-based methods."""
        if self.is_vit:
            # Usually the norm layer before the head, or the last block
            if hasattr(self.model, 'blocks'):
                return self.model.blocks[-1].norm1 # Common in timm ViTs
            elif hasattr(self.model, 'norm'):
                return self.model.norm
            return None
        else:
            # CNNs: last convolutional layer or feature map
            if hasattr(self.model, 'forward_features'):
                # Heuristic: the layer just before the classifier
                if hasattr(self.model, 'layer4'): # ResNets
                    return self.model.layer4[-1]
                elif hasattr(self.model, 'stages'): # ConvNeXt
                    return self.model.stages[-1]
                return None
            return None
            
    def get_attention_maps(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Returns attention maps if available (for Rollout)."""
        # Note: In timm, you usually need to create the model with `output_modules` or
        # use hooks. For simplicity, we might hook into it during rollout, or 
        # rely on specific timm methods if they expose attention.
        raise NotImplementedError("Attention maps retrieval requires specific hooks.")


def create_timm_model(model_name: str, pretrained: bool = True, **kwargs) -> ModelWrapper:
    model = timm.create_model(model_name, pretrained=pretrained, **kwargs)
    is_vit = "vit" in model_name or "deit" in model_name or "beit" in model_name or "eva" in model_name
    return ModelWrapper(model, model_name, is_vit=is_vit)


# Register some common backbones
@MODELS.register("vit_base_patch16_224")
def vit_base_patch16_224(**kwargs):
    return create_timm_model("vit_base_patch16_224", **kwargs)

@MODELS.register("resnet50")
def resnet50(**kwargs):
    return create_timm_model("resnet50", **kwargs)

@MODELS.register("convnext_base")
def convnext_base(**kwargs):
    return create_timm_model("convnext_base", **kwargs)

@MODELS.register("deit_base_patch16_224")
def deit_base_patch16_224(**kwargs):
    return create_timm_model("deit_base_patch16_224", **kwargs)
