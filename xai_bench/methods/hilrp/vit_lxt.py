"""HiLRP attribution for plain timm ViTs (DINO, DeiT, AugReg, ...) via the LXT
efficient backend, plus the SSL-scalar entry point.

The port is the simplest of the family: no hierarchy, so Lemma 2 has nothing to
route except the patch-embed conv (Gamma, fourth corollary). Patches: GELU and
LayerNorm identity rules (including the timm LayerNorm subclass, see the
norm-subclass trap in pvt_lxt), and CP attention (softmax matrix detached).

SSL-scalar attribution (the label-free branch): instead of a class logit, the
backward starts from any differentiable scalar of the frozen features. Here:
DINO-style view similarity, s = cos(cls(x), cls(x_ref).detach()), which asks
"what evidence in x supports its similarity to the reference view". No
classifier head, no labels anywhere. Conservation bookkeeping is identical:
seed 1 at the scalar, relevance sums reported relative to s.
"""
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from lxt.efficient.core import monkey_patch
from lxt.efficient.patches import patch_method, non_linear_forward, layer_norm_forward
from lxt.efficient.zennit_patches import monkey_patch_zennit
from zennit.composites import LayerMapComposite
from zennit.rules import Gamma

from lxt.efficient.rules import divide_gradient

from timm.models import vision_transformer as _vit_mod
from timm.models.vision_transformer import Attention
from timm.layers.norm import LayerNorm as TimmLayerNorm


# ----------------------------------------------------------- attention patch
# 'cp' (default): softmax attention detached, relevance through values only.
# 'attnlrp': full AttnLRP semantics in Gradient x Input: the plain softmax vjp
# IS the softmax Taylor rule (Prop 3.1), divide_gradient implements the uniform
# bilinear rule (Eq. 7). Used for the lineage comparison against AttnLRP proper.
ATTN_MODE = "cp"


def cp_vit_attention_forward(self, x, attn_mask=None, is_causal=False):
    """timm vision_transformer.Attention.forward (explicit path) under LRP
    rules; see ATTN_MODE above. Forward numerics identical to timm."""
    assert attn_mask is None and not is_causal, "classification path only"
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)

    q = q * self.scale
    attn = q @ k.transpose(-2, -1)
    if ATTN_MODE == "attnlrp":
        attn = divide_gradient(attn, 2)       # uniform rule on q@k^T
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    if ATTN_MODE == "cp":
        x = attn.detach() @ v                 # CP-LRP: attention as constant weights
    else:
        x = divide_gradient(attn @ v, 2)      # uniform rule on attn@v; softmax vjp = Taylor

    x = x.transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim)
    x = self.norm(x)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


VIT_PATCH_MAP = {
    nn.GELU: partial(patch_method, non_linear_forward, keep_original=True),
    nn.LayerNorm: partial(patch_method, layer_norm_forward),
    TimmLayerNorm: partial(patch_method, layer_norm_forward),   # subclass trap guard
    Attention: partial(patch_method, cp_vit_attention_forward),
}

_PATCHED = False


def ensure_patched(verbose=False):
    global _PATCHED
    if not _PATCHED:
        monkey_patch(_vit_mod, patch_map=VIT_PATCH_MAP, verbose=verbose)
        monkey_patch_zennit(verbose=verbose)
        _PATCHED = True


def make_composite(gamma=0.25, conv_gamma=None):
    conv_gamma = gamma if conv_gamma is None else conv_gamma
    return LayerMapComposite([
        (nn.Conv2d, Gamma(conv_gamma)),
        (nn.Linear, Gamma(gamma)),
    ])


def _run_backward(model, x, scalar_fn, gamma, conv_gamma):
    """Shared driver: forward under the composite, backward from scalar_fn's
    output, return (scalar_value, aux, pixel_map, token_map)."""
    ensure_patched()
    model.eval()
    x = x.clone().detach().requires_grad_(True)
    composite = make_composite(gamma, conv_gamma)
    with composite.context(model) as mod:
        scalar, aux = scalar_fn(mod, x)
        mod.zero_grad(set_to_none=True)
        scalar.backward()
    pixel_map = (x * x.grad)[0].sum(0).detach().cpu()
    return scalar.item(), aux, pixel_map


def attribute_vit(model, x, target=None, gamma=0.25, conv_gamma=None, attn_mode="cp"):
    """Class-logit attribution for a plain timm ViT."""
    global ATTN_MODE
    ATTN_MODE = attn_mode
    holder = {}

    def scalar_fn(mod, xin):
        logits = mod(xin)
        t = int(logits[0].argmax()) if target is None else target
        holder["target"] = t
        return logits[0, t], None

    val, _, pixel_map = _run_backward(model, x, scalar_fn, gamma, conv_gamma)
    denom = val if abs(val) > 1e-12 else 1.0
    return dict(target=holder["target"], logit=val, pixel_map=pixel_map,
                stage_sums=[("pixels", pixel_map.sum().item() / denom)],
                input_map=pixel_map)


def attribute_ssl_similarity(model, x, x_ref, gamma=0.25, conv_gamma=None):
    """Label-free attribution: what evidence in x supports cos-similarity of its
    CLS embedding to the (frozen, detached) embedding of x_ref."""
    with torch.no_grad():
        z_ref = model.forward_features(x_ref)[:, 0]
        z_ref = F.normalize(z_ref, dim=-1)

    def scalar_fn(mod, xin):
        z = mod.forward_features(xin)[:, 0]
        # identity rule for the normalization: cosine is scale-invariant, so an
        # in-graph norm makes the relevance sum structurally zero (the same
        # invariance mechanism as an unpatched LayerNorm). Detaching the norm
        # treats it as a constant and lets relevance conserve to the scalar.
        z = z / z.norm(dim=-1, keepdim=True).detach()
        return (z * z_ref).sum(), None

    sim, _, pixel_map = _run_backward(model, x, scalar_fn, gamma, conv_gamma)
    denom = sim if abs(sim) > 1e-12 else 1.0
    return dict(similarity=sim, pixel_map=pixel_map,
                stage_sums=[("pixels", pixel_map.sum().item() / denom)])
