import torch
import torch.nn.functional as F
from captum.attr import LayerGradCam

from xai_bench.registry import METHODS


class CaptumLayerCamWrapper:
    def __init__(self, model: torch.nn.Module, method_cls, target_layer, is_vit=False):
        self.model = model
        self.target_layer = target_layer
        self.is_vit = is_vit
        self.method = method_cls(model, target_layer)

    def __call__(self, inputs: torch.Tensor, target: int, **kwargs) -> torch.Tensor:
        # returns [B, 1, ...] (spatial dimensions)
        attr = self.method.attribute(inputs, target=target, relu_attributions=True, **kwargs)
        
        if self.is_vit:
            # ViT attributes might be [B, 1, N] where N = 1 + H*W (cls token + patches)
            # or [B, N, D] depending on how captum hooks it.
            # Captum LayerGradCam usually returns the same shape as the hooked layer output.
            # For timm ViT, output is [B, N, D]. Captum reduces over channels (D).
            # So attr is [B, N].
            if attr.dim() == 2:
                # Assuming shape is [B, 197] for 14x14 patches + 1 cls token
                B, N = attr.shape
                num_patches = N - 1
                H = W = int(num_patches ** 0.5)
                # discard cls token, reshape
                attr = attr[:, 1:].reshape(B, 1, H, W)
            elif attr.dim() == 3 and attr.shape[1] == 1:
                # [B, 1, N]
                B, _, N = attr.shape
                num_patches = N - 1
                H = W = int(num_patches ** 0.5)
                attr = attr[:, :, 1:].reshape(B, 1, H, W)
                
        # Upsample to input resolution
        input_size = inputs.shape[-2:]
        attr = F.interpolate(attr, size=input_size, mode='bilinear', align_corners=False)
        return attr.squeeze(1) # [B, H, W]


@METHODS.register("grad_cam")
def get_gradcam(model_wrapper, **kwargs):
    def method(inputs, target):
        layer = model_wrapper.get_target_layer()
        if layer is None:
            raise ValueError(f"No target layer found for {model_wrapper.model_name}")
        wrapper = CaptumLayerCamWrapper(model_wrapper.model, LayerGradCam, layer, is_vit=model_wrapper.is_vit)
        return wrapper(inputs, target)
    return method
