"""HiLRP attribution for timm EfficientViT (MIT family, multi-scale linear
attention) via the LXT efficient backend. Second hybrid leg of Gate 3: this
model's baseline rows are CLEAN (it expects ImageNet stats), and Grad-CAM
scores 0.55 pointing on efficientvit_b2, so it now carries the hybrid-gap
comparison while the mobilevitv2 baselines are re-run.

Port size (the abstraction holds again): conv stages are Gamma-hooked linear
maps with BN merged by the shared canonizer; ReLU conserves natively in
Gradient x Input (relu(x) * grad = R_out); residual adds and reshapes route
themselves. Only three rules are architecture-specific:

  * nn.Hardswish -> identity rule (same treatment as GELU/SiLU)
  * nn.LayerNorm -> identity rule (one instance, in the head)
  * LiteMLA._attn -> CP-LRP analog for ReLU linear attention: with
    out_i = sum_j (q_i . k_j) v_j / sum_j (q_i . k_j), the mixing weights are
    the q-k kernel products, so q and k are detached; the normalizer column
    (built from the constant ones-pad of v) is then also constant, and
    relevance flows through the value path only. Forward numerics unchanged.

Conservation framing per CLAUDE.md: token-level at transformer stages is the
guaranteed quantity; pixel-level through the deep BN-merged conv stem is a
controlled deviation to be reported, not claimed.
"""
from functools import partial

import torch
import torch.nn as nn

from lxt.efficient.core import monkey_patch
from lxt.efficient.patches import patch_method, non_linear_forward, layer_norm_forward
from lxt.efficient.zennit_patches import monkey_patch_zennit
from zennit.composites import LayerMapComposite
from zennit.rules import Gamma

from timm.models import efficientvit_mit as _ev_mod
from timm.models.efficientvit_mit import LiteMLA

from .mobilevit_lxt import bn_merge_canonizer


# -------------------------------------------------------- attention patch
def cp_lite_mla_attn(self, q, k, v):
    """LiteMLA._attn with the CP-LRP analog: q and k detached (the implicit
    attention weights and the normalizer become constants), relevance flows
    through the value path. Math identical to timm's implementation."""
    dtype = v.dtype
    q, k, v = q.detach().float(), k.detach().float(), v.float()
    kv = k.transpose(-1, -2) @ v
    out = q @ kv
    out = out[..., :-1] / (out[..., -1:].detach() + self.eps)
    return out.to(dtype)


EFFICIENTVIT_PATCH_MAP = {
    nn.Hardswish: partial(patch_method, non_linear_forward, keep_original=True),
    nn.LayerNorm: partial(patch_method, layer_norm_forward),
    LiteMLA: partial(patch_method, cp_lite_mla_attn, method_name="_attn"),
}

_PATCHED = False


def ensure_patched(verbose=False):
    global _PATCHED
    if not _PATCHED:
        monkey_patch(_ev_mod, patch_map=EFFICIENTVIT_PATCH_MAP, verbose=verbose)
        monkey_patch_zennit(verbose=verbose)
        _PATCHED = True


def make_composite(model, gamma=0.25, conv_gamma=None):
    # EfficientViT's deep conv stack + wide head-conv (e.g. 384->2304) make the
    # global gamma=0.25 over-concentrate on positive contributions and drive the
    # input-level relevance to zero (degenerate, sum ~0.003). A lower conv_gamma
    # restores conservation (pixel sum ~0.35) with no loss of localization; gamma
    # below ~0.03 swings the other way into epsilon-cancellation instability.
    # This is the split-gamma strategy used for the convolution-hybrid family.
    conv_gamma = 0.05 if conv_gamma is None else conv_gamma
    return LayerMapComposite([
        (nn.Conv2d, Gamma(conv_gamma)),
        (nn.Linear, Gamma(gamma)),
    ], canonizers=[bn_merge_canonizer(model)])


def attribute_efficientvit(model, x, target=None, gamma=0.25, conv_gamma=None):
    """Single-backward-pass HiLRP attribution for timm EfficientViT (MIT).

    Returns dict with target, logit, pixel_map [H, W], stage_sums (conservation
    trace at stem and each stage output, sum(R)/logit).
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
