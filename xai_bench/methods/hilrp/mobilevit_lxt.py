"""HiLRP attribution for timm MobileViTv2 (conv-transformer hybrid) via the LXT
efficient backend. This is the hybrid leg of Gate 3: Grad-CAM collapses on this
model (pointing game 0.49 in our bench), and no conservation-based method has
defined rules for it.

Why this port is small: the abstraction (CLAUDE.md) holds. Conv stages are
linear maps (zennit Gamma covers Conv2d, BatchNorm at eval is a per-channel
affine that autograd handles with bounded bias absorption). The MobileViTv2
block's unfold/fold are permutations (Lemma 2, transparent). Only three rules
are architecture-specific:

  * nn.SiLU      -> identity rule (same treatment as GELU)
  * GroupNorm1   -> identity rule via detached std (the LayerNorm analog;
                    num_groups == 1 normalizes over (C, H, W) jointly)
  * LinearSelfAttention -> CP-LRP analog for separable attention: the context
    vector (built from query-softmax scores and keys) is the mixing weight,
    so it is detached; relevance flows through the value path only. The
    elementwise value * context product then has one constant factor, so no
    gradient division is needed. F.relu on values conserves exactly in
    Gradient x Input (relu(v) * grad = R_out).

Patches are class-level and process-global. Conservation framing per CLAUDE.md:
token-level guaranteed through the proven operations, pixel-level approximate
under Gamma (report the measured deviation, never claim exact pixel
conservation).
"""
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from lxt.efficient.core import monkey_patch
from lxt.efficient.patches import patch_method, non_linear_forward
from lxt.efficient.rules import divide_gradient
from lxt.efficient.zennit_patches import monkey_patch_zennit
from zennit.canonizers import NamedMergeBatchNorm
from zennit.composites import LayerMapComposite
from zennit.rules import Gamma

from timm.models import mobilevit as _mv_mod
from timm.models.mobilevit import LinearSelfAttention
from timm.layers.norm import GroupNorm1
from timm.layers import norm as _norm_mod


# GroupNorm1 (num_groups==1) takes the mean over the whole (C,H,W) tensor, so
# keeping it live routes every output pixel's gradient through the global mean and
# was suspected of smearing object relevance across the background. A/B TESTED
# (scripts/diagnose_vit_mobilevit.py, n=50): detaching the mean actually LOWERS
# Pointing (0.88 -> 0.82) with no conservation gain, so mean-live is kept. The
# smear affects the map's appearance, not the peak; MobileViT's honest limitation
# is robustness (max-sensitivity), inherent to the deep separable-conv gradient.
DETACH_MEAN = False


# ------------------------------------------------------------- norm patch
def group_norm1_forward(self, x):
    """Identity rule for GroupNorm with num_groups == 1: normalize with the
    std detached, affine kept. Same principle as the LayerNorm patch. With
    DETACH_MEAN, the global mean is detached too (see note above)."""
    dims = tuple(range(1, x.dim()))
    mean = x.mean(dim=dims, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=dims, keepdim=True)
    std = (var + self.eps).sqrt()
    y = (x - (mean.detach() if DETACH_MEAN else mean)) / std.detach()
    if self.weight is not None:
        shape = (1, -1) + (1,) * (x.dim() - 2)
        y = y * self.weight.view(shape) + self.bias.view(shape)
    return y


# -------------------------------------------------------- attention patch
def cp_linear_self_attention_forward(self, x, x_prev=None):
    """timm LinearSelfAttention.forward (self-attention path) with the CP-LRP
    analog: only the softmax context_scores are detached (the attention-weight
    constant). The key path stays live, because in separable attention the
    cross-token mixing lives in context_vector = sum_N(key * scores): detaching
    all of it would leave the transformer with zero spatial relevance routing
    (measured: pointing game drops below the Grad-CAM floor). The elementwise
    relu(value) * context product is bilinear with both factors live, so the
    uniform rule halves the relevance (divide_gradient), matching LXT's mul
    handling. Forward numerics identical to timm."""
    assert x_prev is None, "HiLRP MobileViT patch covers the self-attention path"
    qkv = self.qkv_proj(x)
    query, key, value = qkv.split([1, self.embed_dim, self.embed_dim], dim=1)

    context_scores = F.softmax(query, dim=-1)
    context_scores = self.attn_drop(context_scores)

    context_vector = (key * context_scores.detach()).sum(dim=-1, keepdim=True)

    out = F.relu(value) * context_vector.expand_as(value)
    out = divide_gradient(out, 2)                    # uniform rule for the bilinear mul
    out = self.out_proj(out)
    out = self.out_drop(out)
    return out


MOBILEVIT_PATCH_MAP = {
    nn.SiLU: partial(patch_method, non_linear_forward, keep_original=True),
    GroupNorm1: partial(patch_method, group_norm1_forward),
    LinearSelfAttention: partial(patch_method, cp_linear_self_attention_forward),
}

_PATCHED = False


def ensure_patched(verbose=False):
    global _PATCHED
    if not _PATCHED:
        monkey_patch(_mv_mod, patch_map=MOBILEVIT_PATCH_MAP, verbose=verbose)
        monkey_patch(_norm_mod, patch_map={GroupNorm1: MOBILEVIT_PATCH_MAP[GroupNorm1]},
                     verbose=verbose)
        monkey_patch_zennit(verbose=verbose)
        _PATCHED = True


def bn_merge_canonizer(model):
    """Merge every (conv, bn) pair inside timm ConvNormAct blocks. Without the
    merge, the BatchNorm affine sits unhooked between Gamma-hooked convs and its
    signed scales destroy the gamma positivity bias (measured: relevance sums
    flip to about -3 by stage3 and stay negative to the input). This is the
    standard zennit canonization for conv+BN networks."""
    pairs = []
    for name, mod in model.named_modules():
        conv = getattr(mod, "conv", None)
        if not isinstance(conv, nn.Conv2d):
            continue
        for bn_attr in ("bn", "norm"):      # timm ConvNormAct variants differ
            bn = getattr(mod, bn_attr, None)
            if isinstance(bn, nn.BatchNorm2d):
                pairs.append(([f"{name}.conv"], f"{name}.{bn_attr}"))
                break
    return NamedMergeBatchNorm(pairs)


def make_composite(model, gamma=0.25, conv_gamma=None):
    conv_gamma = gamma if conv_gamma is None else conv_gamma
    return LayerMapComposite([
        (nn.Conv2d, Gamma(conv_gamma)),
        (nn.Linear, Gamma(gamma)),
    ], canonizers=[bn_merge_canonizer(model)])


def attribute_mobilevit(model, x, target=None, gamma=0.25, conv_gamma=None):
    """Single-backward-pass HiLRP attribution for timm MobileViTv2.

    Returns dict with target, logit, pixel_map [H, W], stage_sums (conservation
    trace at each byobnet stage output, sum(R)/logit).
    """
    ensure_patched()
    model.eval()

    captures, hooks = [], []

    def grab_out(name):
        def hook(module, args, out):
            out.retain_grad()
            captures.append((name, out))
        return hook

    hooks.append(model.stem.register_forward_hook(grab_out("stem")))
    for i, stage in enumerate(model.stages):
        hooks.append(stage.register_forward_hook(grab_out(f"stage{i}")))

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

    stage_sums = [(n, (t * t.grad)[0].sum().item() / denom) for n, t in captures]
    pixel_map = (x * x.grad)[0].sum(0).detach().cpu()

    return dict(target=target, logit=logit, pixel_map=pixel_map, stage_sums=stage_sums)
