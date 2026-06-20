import torch
import torch.nn.functional as F
# pyrefly: ignore [missing-import]
from captum.attr import LayerGradCam

from xai_bench.registry import METHODS


class ViTCustomGradCam:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activation = None
        self.gradient = None
        
        def fw_hook(module, input, output):
            self.activation = output
        def bw_hook(module, grad_in, grad_out):
            self.gradient = grad_out[0]
            
        self.target_layer.register_forward_hook(fw_hook)
        self.target_layer.register_full_backward_hook(bw_hook)
        
    def __call__(self, inputs, target, **kwargs):
        inputs = inputs.clone().detach().requires_grad_(True)
        self.model.zero_grad()
        out = self.model(inputs)
        loss = out[0, target]
        loss.backward()
        
        # ViT activation: [B, 197, 768]
        # Pool gradients over the token dimension to get feature importance
        weights = self.gradient.mean(dim=1, keepdim=True) # [B, 1, 768]
        cam = (weights * self.activation).sum(dim=2) # [B, 197]
        cam = F.relu(cam)
        
        # Reshape to 2D
        B, N = cam.shape
        num_patches = N - 1
        H = W = int(num_patches ** 0.5)
        # discard cls token, reshape
        cam = cam[:, 1:].reshape(B, 1, H, W)
        
        input_size = inputs.shape[-2:]
        cam = F.interpolate(cam, size=input_size, mode='bilinear', align_corners=False)
        return cam.squeeze(1)


class ViTCustomGradCamPlusPlus:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activation = None
        self.gradient = None
        
        def fw_hook(module, input, output):
            self.activation = output
        def bw_hook(module, grad_in, grad_out):
            self.gradient = grad_out[0]
            
        self.target_layer.register_forward_hook(fw_hook)
        self.target_layer.register_full_backward_hook(bw_hook)
        
    def __call__(self, inputs, target, **kwargs):
        inputs = inputs.clone().detach().requires_grad_(True)
        self.model.zero_grad()
        out = self.model(inputs)
        loss = out[0, target]
        loss.backward()
        
        # ViT activation: [B, 197, 768]
        # Grad-CAM++ weighting approximation
        grad_2 = self.gradient ** 2
        grad_3 = self.gradient ** 3
        
        sum_activations = self.activation.sum(dim=1, keepdim=True) # sum over spatial/tokens
        eps = 1e-7
        alpha = grad_2 / (2 * grad_2 + sum_activations * grad_3 + eps)
        alpha = torch.where(self.gradient != 0.0, alpha, torch.zeros_like(alpha))
        
        weights = (alpha * F.relu(self.gradient)).sum(dim=1, keepdim=True) # [B, 1, 768]
        cam = (weights * self.activation).sum(dim=2) # [B, 197]
        cam = F.relu(cam)
        
        # Reshape to 2D
        B, N = cam.shape
        num_patches = N - 1
        H = W = int(num_patches ** 0.5)
        cam = cam[:, 1:].reshape(B, 1, H, W)
        
        input_size = inputs.shape[-2:]
        cam = F.interpolate(cam, size=input_size, mode='bilinear', align_corners=False)
        return cam.squeeze(1)


class CNNCustomGradCamPlusPlus:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activation = None
        self.gradient = None
        
        def fw_hook(module, input, output):
            self.activation = output
        def bw_hook(module, grad_in, grad_out):
            self.gradient = grad_out[0]
            
        self.target_layer.register_forward_hook(fw_hook)
        self.target_layer.register_full_backward_hook(bw_hook)
        
    def __call__(self, inputs, target, **kwargs):
        inputs = inputs.clone().detach().requires_grad_(True)
        self.model.zero_grad()
        out = self.model(inputs)
        loss = out[0, target]
        loss.backward()
        
        # CNN activation: [B, C, H, W]
        grad_2 = self.gradient ** 2
        grad_3 = self.gradient ** 3
        
        sum_activations = self.activation.sum(dim=(2, 3), keepdim=True)
        eps = 1e-7
        alpha = grad_2 / (2 * grad_2 + sum_activations * grad_3 + eps)
        alpha = torch.where(self.gradient != 0.0, alpha, torch.zeros_like(alpha))
        
        weights = (alpha * F.relu(self.gradient)).sum(dim=(2, 3), keepdim=True)
        cam = (weights * self.activation).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        input_size = inputs.shape[-2:]
        cam = F.interpolate(cam, size=input_size, mode='bilinear', align_corners=False)
        return cam.squeeze(1)


class CaptumLayerCamWrapper:
    def __init__(self, model: torch.nn.Module, method_cls, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.method = method_cls(model, target_layer)

    def __call__(self, inputs: torch.Tensor, target: int, **kwargs) -> torch.Tensor:
        attr = self.method.attribute(inputs, target=target, relu_attributions=True, **kwargs)
        input_size = inputs.shape[-2:]
        attr = F.interpolate(attr, size=input_size, mode='bilinear', align_corners=False)
        return attr.squeeze(1)


@METHODS.register("grad_cam")
def get_gradcam(model_wrapper, **kwargs):
    def method(inputs, target):
        layer = model_wrapper.get_target_layer()
        if layer is None:
            raise ValueError(f"No target layer found for {model_wrapper.model_name}")
            
        if model_wrapper.is_vit:
            wrapper = ViTCustomGradCam(model_wrapper.model, layer)
        else:
            wrapper = CaptumLayerCamWrapper(model_wrapper.model, LayerGradCam, layer)
            
        return wrapper(inputs, target)
    return method


@METHODS.register("grad_cam_plus_plus")
def get_gradcam_plus_plus(model_wrapper, **kwargs):
    def method(inputs, target):
        layer = model_wrapper.get_target_layer()
        if layer is None:
            raise ValueError(f"No target layer found for {model_wrapper.model_name}")
            
        if model_wrapper.is_vit:
            wrapper = ViTCustomGradCamPlusPlus(model_wrapper.model, layer)
        else:
            wrapper = CNNCustomGradCamPlusPlus(model_wrapper.model, layer)
            
        return wrapper(inputs, target)
    return method
