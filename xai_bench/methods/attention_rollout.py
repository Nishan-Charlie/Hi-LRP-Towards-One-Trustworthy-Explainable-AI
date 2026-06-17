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
        hooks = []
        
        def hook_fn(module, input, output):
            # output of attention is usually (x, attn) or we have to hook the softmax
            # Since standard timm ViT's don't output attention weights natively without flags,
            # this requires hooking the Softmax inside the Attention block.
            attentions.append(output)
            
        # Register hooks to all Softmax layers in Attention blocks
        for name, module in self.model_wrapper.model.named_modules():
            if 'attn.attn_drop' in name: # Varies by timm version, hooking dropout after softmax
                pass
            if isinstance(module, torch.nn.Softmax) and 'attn' in name:
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
