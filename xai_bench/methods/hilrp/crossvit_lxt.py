"""HiLRP attribution for CrossViT: a Vision Transformer with real CROSS-ATTENTION
layers. This upgrades the cross-attention primitive from proven-in-toy to
realized-on-a-pretrained-model, on an actual ViT doing ImageNet classification
(so the standard Pointing evaluation applies).

CrossViT has two token-scale branches with self-attention, fused by
CrossAttentionBlocks in which the CLS token of one branch attends to the tokens
of the other (query from stream A, key/value from stream B). Both the branch
self-attention (timm vision_transformer.Attention) and the cross-attention
(timm crossvit.CrossAttention) are covered by the same CP rule: detach the
softmax attention, route relevance through the value path. Everything else
(LayerNorm/GELU identity, Linear/Conv gamma, the two patch embeds, token
concat/split) is handled by the shared primitives.

Input is 240x240; callers should resize accordingly.
"""
import torch
import torch.nn as nn

from lxt.efficient.patches import patch_method
from zennit.composites import LayerMapComposite
from zennit.rules import Gamma

from timm.models.crossvit import CrossAttention
import xai_bench.methods.hilrp.vit_lxt as vit_lxt


def cp_cross_attention_forward(self, x):
    """timm crossvit.CrossAttention.forward with CP-LRP: softmax attention
    detached, relevance flows through the value path (the other branch's tokens).
    Forward numerics identical to timm."""
    B, N, C = x.shape
    h, d = self.num_heads, C // self.num_heads
    q = self.wq(x[:, 0:1, ...]).reshape(B, 1, h, d).permute(0, 2, 1, 3)
    k = self.wk(x).reshape(B, N, h, d).permute(0, 2, 1, 3)
    v = self.wv(x).reshape(B, N, h, d).permute(0, 2, 1, 3)
    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    x = (attn.detach() @ v).transpose(1, 2).reshape(B, 1, C)   # CP-LRP
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


_PATCHED = False


def ensure_patched(verbose=False):
    global _PATCHED
    if not _PATCHED:
        vit_lxt.ATTN_MODE = "cp"          # branch self-attention uses CP here
        vit_lxt.ensure_patched(verbose)   # GELU, LayerNorm, vision_transformer.Attention, zennit
        patch_method(cp_cross_attention_forward, CrossAttention)
        _PATCHED = True


def make_composite(gamma=0.25, conv_gamma=None):
    cg = gamma if conv_gamma is None else conv_gamma
    return LayerMapComposite([(nn.Conv2d, Gamma(cg)), (nn.Linear, Gamma(gamma))])


def attribute_crossvit(model, x, target=None, gamma=0.25, conv_gamma=None):
    """Single-backward-pass HiLRP attribution for timm CrossViT (240x240 input)."""
    ensure_patched()
    model.eval()
    x = x.clone().detach().requires_grad_(True)
    composite = make_composite(gamma, conv_gamma)
    with composite.context(model) as mod:
        logits = mod(x)
        if target is None:
            target = int(logits[0].argmax())
        mod.zero_grad(set_to_none=True)
        logits[0, target].backward()
    logit = logits[0, target].item()
    denom = logit if abs(logit) > 1e-12 else 1.0
    pixel_map = (x * x.grad)[0].sum(0).detach().cpu()
    return dict(target=target, logit=logit, pixel_map=pixel_map,
                stage_sums=[("pixels", pixel_map.sum().item() / denom)])
