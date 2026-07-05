"""HiLRP attribution for timm PVT v2 (spatial-reduction attention) via the LXT
efficient backend. This is the THIRD COROLLARY of the unifying proposition made
concrete: the K/V spatial reduction is a strided conv on the token grid, i.e.
W . phi(concat(neighborhood)), so it needs no new rule at all: the zennit Gamma
hook on the sr Conv2d and the identity-rule LayerNorm behind it ARE the
corollary. With CP attention, relevance enters through the value path at
REDUCED resolution and routes back to full-resolution tokens through the sr
conv: cross-resolution conserved flow on a real pretrained model.

Architecture-specific patches (same recipe as the other ports):

  * nn.GELU      -> identity rule
  * nn.LayerNorm -> identity rule (block norms, post-sr norm, patch-embed norm)
  * pvt_v2.Attention -> explicit softmax path with the attention matrix
    detached (CP-LRP); forward numerics identical to timm's non-fused path

Everything else routes itself in Gradient x Input: residual adds, reshapes,
the overlapping patch-embed conv (a linear neighborhood map, Lemma 1 does not
require a partition), the depthwise conv inside the MLP (Conv2d, Gamma).

Note: the `pool` branch (linear-attention "li" variants) is also handled by the
same patched forward; the pooled path is average pooling (linear) + sr conv.
"""
from functools import partial

import torch
import torch.nn as nn

from lxt.efficient.core import monkey_patch
from lxt.efficient.patches import patch_method, non_linear_forward, layer_norm_forward
from lxt.efficient.zennit_patches import monkey_patch_zennit
from zennit.composites import LayerMapComposite
from zennit.rules import Gamma

from timm.models import pvt_v2 as _pvt_mod
from timm.models.pvt_v2 import Attention
from timm.layers.norm import LayerNorm as TimmLayerNorm


# ----------------------------------------------------------- attention patch
def cp_pvt_attention_forward(self, x, feat_size):
    """timm pvt_v2.Attention.forward (explicit path) with CP-LRP: the softmax
    attention matrix is detached, relevance flows through the value path and
    from there through the sr conv back to full-resolution tokens."""
    B, N, C = x.shape
    H, W = feat_size
    q = self.q(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

    if self.pool is not None:
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.sr(self.pool(x)).reshape(B, C, -1).permute(0, 2, 1)
        x = self.norm(x)
        x = self.act(x)
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    else:
        if self.sr is not None:
            x = x.permute(0, 2, 1).reshape(B, C, H, W)
            x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    k, v = kv.unbind(0)

    q = q * self.scale
    attn = q @ k.transpose(-2, -1)
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    x = attn.detach() @ v                    # CP-LRP: attention as constant weights

    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


PVT_PATCH_MAP = {
    nn.GELU: partial(patch_method, non_linear_forward, keep_original=True),
    nn.LayerNorm: partial(patch_method, layer_norm_forward),
    # CRITICAL: timm's PVT norms are timm.layers.norm.LayerNorm, a SUBCLASS whose
    # own forward ignores the nn.LayerNorm class patch. Unpatched LN runs with the
    # full gradient, and LayerNorm's scale/shift invariance forces per-token
    # x . grad == 0 exactly: this silently zeroed every deep-cut relevance sum
    # ("the conservation cliff") while leaving maps plausible. Patch the subclass.
    TimmLayerNorm: partial(patch_method, layer_norm_forward),
    Attention: partial(patch_method, cp_pvt_attention_forward),
}


def audit_norm_patches(model, verbose=True):
    """Guard against the class-subclass patch trap: list every normalization
    module whose executing (class) forward is not an LRP-patched function."""
    unpatched = []
    for name, m in model.named_modules():
        # input-statistic norms only: eval-mode BatchNorm uses running stats and
        # is a plain affine (no invariance issue, no patch needed)
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm2d)):
            fwd = type(m).forward.__name__
            if fwd not in ("layer_norm_forward", "group_norm1_forward",
                           "ln2d_identity_forward"):
                unpatched.append((name, type(m).__module__ + "." + type(m).__name__, fwd))
    if verbose and unpatched:
        print("WARNING: unpatched norm modules (class forward not an LRP rule):")
        for n, t, f in unpatched:
            print(f"  {n}: {t} (forward={f})")
    return unpatched

_PATCHED = False


def ensure_patched(verbose=False):
    global _PATCHED
    if not _PATCHED:
        monkey_patch(_pvt_mod, patch_map=PVT_PATCH_MAP, verbose=verbose)
        monkey_patch_zennit(verbose=verbose)
        _PATCHED = True


def make_composite(gamma=0.25, conv_gamma=None):
    conv_gamma = gamma if conv_gamma is None else conv_gamma
    return LayerMapComposite([
        (nn.Conv2d, Gamma(conv_gamma)),
        (nn.Linear, Gamma(gamma)),
    ])


def attribute_pvt(model, x, target=None, gamma=0.25, conv_gamma=None):
    """Single-backward-pass HiLRP attribution for timm PVT v2.

    Returns dict with target, logit, pixel_map [H, W], stage_sums (conservation
    trace at patch embed and each stage output, sum(R)/logit).
    """
    ensure_patched()
    model.eval()

    captures, hooks = [], []

    def grab_out(name):
        def hook(module, args, out):
            t = out[0] if isinstance(out, tuple) else out
            t.retain_grad()
            captures.append((name, t))
        return hook

    hooks.append(model.patch_embed.register_forward_hook(grab_out("embed")))
    for i, stage in enumerate(model.stages):
        hooks.append(stage.register_forward_hook(grab_out(f"stage{i}")))

    x = x.clone().detach().requires_grad_(True)
    composite = make_composite(gamma, conv_gamma)

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

    stage_sums = []
    for n, t in captures:
        if t.grad is not None:
            stage_sums.append((n, (t * t.grad)[0].sum().item() / denom))
    pixel_map = (x * x.grad)[0].sum(0).detach().cpu()

    return dict(target=target, logit=logit, pixel_map=pixel_map, stage_sums=stage_sums)
