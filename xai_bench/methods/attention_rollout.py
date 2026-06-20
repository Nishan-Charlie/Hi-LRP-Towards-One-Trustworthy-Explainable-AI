import torch
import torch.nn.functional as F
import numpy as np

from xai_bench.registry import METHODS

def rollout(attentions, discard_ratio, head_fusion):
    """Computes Attention Rollout.
    attentions: list of [B, H, N, N] attention matrices from each layer.
    """
    B, num_heads, N, _ = attentions[0].shape
    result = torch.eye(N, device=attentions[0].device).unsqueeze(0).expand(B, N, N).clone()

    for attention in attentions:
        if head_fusion == "mean":
            attention_heads_fused = attention.mean(dim=1)
        elif head_fusion == "max":
            attention_heads_fused = attention.max(dim=1)[0]
        elif head_fusion == "min":
            attention_heads_fused = attention.min(dim=1)[0]
        else:
            raise ValueError("Attention head fusion type Not supported")

        # Drop the lowest attentions, but don't drop the class token
        flat = attention_heads_fused.view(B, -1)
        _, indices = flat.topk(int(flat.size(-1) * (1 - discard_ratio)), -1, False)
        # Note: a proper rollout masks out the bottom discard_ratio of attention weights
        # Here we do a simplified version:
        attention_heads_fused += torch.eye(N, device=attention.device).unsqueeze(0)
        attention_heads_fused = attention_heads_fused / attention_heads_fused.sum(dim=-1, keepdim=True)
        
        result = torch.matmul(attention_heads_fused, result)

    return result

class AttentionRolloutWrapper:
    def __init__(self, model_wrapper, discard_ratio=0.9, head_fusion="mean"):
        self.model_wrapper = model_wrapper
        self.discard_ratio = discard_ratio
        self.head_fusion = head_fusion

    def __call__(self, inputs: torch.Tensor, target: int, **kwargs) -> torch.Tensor:
        if not self.model_wrapper.is_vit:
            raise ValueError("Attention Rollout is only applicable to ViT models.")
        
        # We need a forward hook to grab the attention maps.
        attentions = []
        # Disable fused_attn so we can hook the attention weights
        for m in self.model_wrapper.model.modules():
            if hasattr(m, 'fused_attn'):
                m.fused_attn = False

        hooks = []
        
        def hook_fn(module, input, output):
            # For timm ViT, attn_drop takes the post-softmax attention weights as input.
            # input is a tuple (tensor,)
            attentions.append(input[0])
            
        # Register hooks to all attn_drop layers
        for name, module in self.model_wrapper.model.named_modules():
            if 'attn_drop' in name:
                hooks.append(module.register_forward_hook(hook_fn))
                
        # Forward pass
        _ = self.model_wrapper(inputs)
        
        for hook in hooks:
            hook.remove()
            
        if len(attentions) == 0:
            raise RuntimeError("Could not hook attention weights. Timm model structure might have changed.")

        # attentions is a list of [B, num_heads, N, N]
        rollout_matrix = rollout(attentions, self.discard_ratio, self.head_fusion)
        
        # We want the attention from the CLS token (index 0) to all patches (index 1:)
        # rollout_matrix is [B, N, N]
        cls_attention = rollout_matrix[:, 0, 1:] # [B, N-1]
        
        B, num_patches = cls_attention.shape
        H = W = int(num_patches ** 0.5)
        
        mask = cls_attention.reshape(B, 1, H, W)
        input_size = inputs.shape[-2:]
        mask = F.interpolate(mask, size=input_size, mode='bilinear', align_corners=False)
        return mask.squeeze(1)


@METHODS.register("attention_rollout")
def get_attention_rollout(model_wrapper, **kwargs):
    def method(inputs, target):
        wrapper = AttentionRolloutWrapper(model_wrapper)
        return wrapper(inputs, target)
    return method


class AttentionGradientWrapper:
    """Attention weights multiplied by their gradients (proxy for Chefer LRP)."""
    def __init__(self, model_wrapper):
        self.model_wrapper = model_wrapper

    def __call__(self, inputs: torch.Tensor, target: int, **kwargs) -> torch.Tensor:
        inputs = inputs.clone().detach().requires_grad_(True)
        if not self.model_wrapper.is_vit:
            raise ValueError("Attention Gradient is only applicable to ViT models.")
            
        for m in self.model_wrapper.model.modules():
            if hasattr(m, 'fused_attn'):
                m.fused_attn = False
                
        attentions = []
        gradients = []
        hooks = []
        
        def fw_hook(module, input, output):
            attentions.append(input[0])
            
        def bw_hook(module, grad_in, grad_out):
            gradients.append(grad_in[0])
            
        for name, module in self.model_wrapper.model.named_modules():
            if 'attn_drop' in name:
                hooks.append(module.register_forward_hook(fw_hook))
                hooks.append(module.register_full_backward_hook(bw_hook))
                
        self.model_wrapper.model.zero_grad()
        out = self.model_wrapper(inputs)
        loss = out[0, target]
        loss.backward()
        
        for hook in hooks:
            hook.remove()
            
        # reverse gradients to match forward pass order
        gradients = gradients[::-1]
        
        # Calculate attention * gradient
        # attentions: [B, num_heads, N, N]
        B, num_heads, N, _ = attentions[0].shape
        result = torch.eye(N, device=inputs.device).unsqueeze(0).expand(B, N, N).clone()
        
        for attn, grad in zip(attentions, gradients):
            # grad is w.r.t input to attn_drop, which is the attention map
            # Weight attention by positive gradients (similar to LRP rules)
            weighted_attn = attn * F.relu(grad)
            weighted_attn = weighted_attn.mean(dim=1) # mean over heads
            
            # Rollout
            weighted_attn += torch.eye(N, device=inputs.device).unsqueeze(0)
            weighted_attn = weighted_attn / (weighted_attn.sum(dim=-1, keepdim=True) + 1e-7)
            result = torch.matmul(weighted_attn, result)
            
        cls_attention = result[:, 0, 1:]
        B, num_patches = cls_attention.shape
        H = W = int(num_patches ** 0.5)
        
        mask = cls_attention.reshape(B, 1, H, W)
        input_size = inputs.shape[-2:]
        mask = F.interpolate(mask, size=input_size, mode='bilinear', align_corners=False)
        return mask.squeeze(1)


@METHODS.register("attention_gradient")
def get_attention_gradient(model_wrapper, **kwargs):
    def method(inputs, target):
        wrapper = AttentionGradientWrapper(model_wrapper)
        return wrapper(inputs, target)
    return method
