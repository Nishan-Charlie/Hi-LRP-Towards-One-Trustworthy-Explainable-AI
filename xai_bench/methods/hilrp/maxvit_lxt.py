"""HiLRP attribution for timm MaxViT (multi-axis: block + grid attention) via the
LXT efficient backend. Completes the benchmark: MaxViT alternates convolutional
MBConv stages with block-local and grid-global attention, exercising the full
union of mechanisms (conv stem, windowed attention, and a second permutation
axis, the dilated grid).

The multi-axis structure needs no new rule: block and grid partitioning are both
permutations of the token grid (Lemma 2), so autograd routes relevance through
them exactly. Architecture-specific patches:

  * nn.SiLU        -> identity rule (incl. the SiLU bundled inside BatchNormAct2d,
                     which is a real nn.SiLU submodule)
  * LayerNorm / timm LayerNorm / LayerNorm2d -> identity rule (all three, via the
    norm-subclass guard; LayerNorm2d normalizes over channels in NCHW)
  * AttentionCl    -> CP-LRP (softmax attention detached)
  * Conv2d/Linear  -> gamma; conv+BatchNormAct2d merged by the canonizer
"""
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from lxt.efficient.core import monkey_patch
from lxt.efficient.patches import patch_method, non_linear_forward, layer_norm_forward
from lxt.efficient.zennit_patches import monkey_patch_zennit
from zennit.canonizers import NamedMergeBatchNorm
from zennit.composites import LayerMapComposite
from zennit.rules import Gamma

from timm.models import maxxvit as _mx_mod
from timm.models.maxxvit import AttentionCl
from timm.layers.norm import LayerNorm as TimmLayerNorm, LayerNorm2d
from timm.layers.norm_act import BatchNormAct2d


def ln2d_identity_forward(self, x):
    """LayerNorm2d with the identity rule: normalize over channels (NCHW) with
    the std detached, affine kept."""
    x = x.permute(0, 2, 3, 1)
    mean = x.mean(-1, keepdim=True)
    var = ((x - mean) ** 2).mean(-1, keepdim=True)
    x = (x - mean) / (var + self.eps).sqrt().detach()
    x = x * self.weight + self.bias
    return x.permute(0, 3, 1, 2)


def cp_attention_cl_forward(self, x, shared_rel_pos=None):
    """timm maxxvit.AttentionCl.forward (explicit path) with CP-LRP."""
    B = x.shape[0]
    restore_shape = x.shape[:-1]
    if self.head_first:
        q, k, v = self.qkv(x).view(B, -1, self.num_heads, self.dim_head * 3).transpose(1, 2).chunk(3, dim=3)
    else:
        q, k, v = self.qkv(x).reshape(B, -1, 3, self.num_heads, self.dim_head).transpose(1, 3).unbind(2)
    q = q * self.scale
    attn = q @ k.transpose(-2, -1)
    if self.rel_pos is not None:
        attn = self.rel_pos(attn, shared_rel_pos=shared_rel_pos)
    elif shared_rel_pos is not None:
        attn = attn + shared_rel_pos
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    x = attn.detach() @ v                     # CP-LRP
    x = x.transpose(1, 2).reshape(restore_shape + (-1,))
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


MAXVIT_PATCH_MAP = {
    nn.SiLU: partial(patch_method, non_linear_forward, keep_original=True),
    nn.LayerNorm: partial(patch_method, layer_norm_forward),
    TimmLayerNorm: partial(patch_method, layer_norm_forward),
    LayerNorm2d: partial(patch_method, ln2d_identity_forward),
    AttentionCl: partial(patch_method, cp_attention_cl_forward),
}

_PATCHED = False


def ensure_patched(verbose=False):
    global _PATCHED
    if not _PATCHED:
        monkey_patch(_mx_mod, patch_map=MAXVIT_PATCH_MAP, verbose=verbose)
        monkey_patch_zennit(verbose=verbose)
        _PATCHED = True


def _bn_pairs(model):
    pairs = []
    for name, mod in model.named_modules():
        conv = getattr(mod, "conv", None)
        for bn_attr in ("bn", "norm"):
            bn = getattr(mod, bn_attr, None)
            if isinstance(conv, nn.Conv2d) and isinstance(bn, (nn.BatchNorm2d, BatchNormAct2d)):
                pairs.append(([f"{name}.conv"], f"{name}.{bn_attr}"))
                break
    return pairs


def make_composite(model, gamma=0.25, conv_gamma=None):
    conv_gamma = gamma if conv_gamma is None else conv_gamma
    canon = [NamedMergeBatchNorm(_bn_pairs(model))] if _bn_pairs(model) else []
    return LayerMapComposite(
        [(nn.Conv2d, Gamma(conv_gamma)), (nn.Linear, Gamma(gamma))],
        canonizers=canon,
    )


def attribute_maxvit(model, x, target=None, gamma=0.25, conv_gamma=None):
    ensure_patched()
    model.eval()
    captures, hooks = [], []

    def grab(name):
        def hook(m, a, o):
            t = o[0] if isinstance(o, tuple) else o
            t.retain_grad(); captures.append((name, t))
        return hook

    hooks.append(model.stem.register_forward_hook(grab("stem")))
    for i, stage in enumerate(model.stages):
        hooks.append(stage.register_forward_hook(grab(f"stage{i}")))

    x = x.clone().detach().requires_grad_(True)
    composite = make_composite(model, gamma, conv_gamma)
    try:
        with composite.context(model) as mod:
            logits = mod(x)
            if target is None:
                target = int(logits[0].argmax())
            mod.zero_grad(set_to_none=True)
            logits[0, target].backward()
    finally:
        for h in hooks:
            h.remove()

    logit = logits[0, target].item()
    denom = logit if abs(logit) > 1e-12 else 1.0
    stage_sums = [(n, (t * t.grad)[0].sum().item() / denom) for n, t in captures if t.grad is not None]
    pixel_map = (x * x.grad)[0].sum(0).detach().cpu()
    return dict(target=target, logit=logit, pixel_map=pixel_map, stage_sums=stage_sums)
