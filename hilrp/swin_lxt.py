"""Gate-2-real: HiLRP attribution for pretrained timm Swin via LXT's efficient
(Gradient x Input) backend + zennit gamma composites.

Why this backend and not pure epsilon
-------------------------------------
A first implementation on `lxt.explicit` with epsilon rules everywhere was
forward-exact (max|diff| ~3e-7) and conserved at the head (sum(R) at the final
norm ~= 1.01), but relevance amplified with sign flips descending the stages on
real images (|sum| up to ~10^3 at the input tokens): the known epsilon-LRP
cancellation blow-up on deep vision models. LXT's own vision recipe addresses
exactly this -- "AttnLRP outside the attention & CP-LRP inside the attention is
easier to tune for gamma" (lxt.efficient.models.vit_torch) -- i.e. gamma rules
on Linear/Conv2d via zennit, identity-rule LayerNorm/GELU, and CP-LRP (detached
attention matrix) inside the attention. We mirror that recipe for timm Swin.

What is patched vs. what routes itself
--------------------------------------
In the Gradient x Input convention (relevance R(t) = t * t.grad), the residual
adds, cyclic shifts, window partitions, patch-merge regroups, padding/slicing,
constant scaling, and mean pooling are all handled *exactly* by plain autograd
(HiLRP Lemma 2: permutations and 1-in-1-out linear ops conserve; proportional
split through + and mean falls out of x * grad). Only three things need rules:

  * nn.LayerNorm  -> identity rule (stop-gradient through the std)     [LXT patch]
  * nn.GELU       -> identity rule                                     [LXT patch]
  * WindowAttention -> CP-LRP: attention matrix detached, relevance
    flows through the value path only (the SW-MSA mask consequence
    still holds trivially: blocked pairs have attention ~0, so their
    value relevance is ~0)                                             [ours]
  * nn.Linear / nn.Conv2d -> gamma rule via zennit composite           [zennit]

The gamma-on-Conv2d covers the strided patch-embed conv -- the fourth corollary
of the HiLRP proposition -- so maps extend to pixel level for free.

NOTE: patches are class-level and process-global (nn.GELU, nn.LayerNorm, timm
WindowAttention). Use a dedicated process for attribution runs.

Conservation is not faithfulness: seeds are `logits[0, target].backward()`
(seed 1), so total relevance = the logit value; the per-stage sums in
`stage_sums` are reported normalized by the logit and should be ~1.0 up to
bias/gamma absorption. Whether maps land on objects / are class-sensitive is
the Gate-2-real script's job (scripts/gate2_real_swin.py).
"""
from functools import partial

import torch
import torch.nn as nn

from lxt.efficient.core import monkey_patch
from lxt.efficient.patches import patch_method, non_linear_forward, layer_norm_forward
from lxt.efficient.rules import divide_gradient
from lxt.efficient.zennit_patches import monkey_patch_zennit
from zennit.composites import LayerMapComposite
from zennit.rules import Gamma

from timm.models import swin_transformer as _swin_mod
from timm.models.swin_transformer import WindowAttention


# ----------------------------------------------------------- attention patch
# module-level switch: 'cp' (attention matrix detached, LXT's vision default)
# or 'attnlrp' (full AttnLRP: in Gradient x Input, the plain softmax vjp IS the
# softmax Taylor rule of Prop 3.1, and divide_gradient on the two matmul outputs
# implements the uniform bilinear rule of Eq. 7).
ATTN_MODE = "cp"


def hilrp_window_attention_forward(self, x, mask=None):
    """timm WindowAttention.forward (explicit path) under LRP rules; forward
    numerics identical to timm's non-fused path. See ATTN_MODE above."""
    B_, N, C = x.shape
    qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    q = q * self.scale
    attn = q @ k.transpose(-2, -1)
    if ATTN_MODE == "attnlrp":
        attn = divide_gradient(attn, 2)      # uniform rule on q@k^T
    attn = attn + self._get_rel_pos_bias()
    if mask is not None:
        num_win = mask.shape[0]
        attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, self.num_heads, N, N)
    attn = self.softmax(attn)
    attn = self.attn_drop(attn)

    if ATTN_MODE == "cp":
        x = attn.detach() @ v                # CP-LRP: attention as constant weights
    else:
        x = divide_gradient(attn @ v, 2)     # uniform rule on attn@v; softmax vjp = Taylor rule

    x = x.transpose(1, 2).reshape(B_, N, -1)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


# patch map in LXT style (class-level; see lxt.efficient.models.vit_torch)
SWIN_PATCH_MAP = {
    nn.GELU: partial(patch_method, non_linear_forward, keep_original=True),
    nn.LayerNorm: partial(patch_method, layer_norm_forward),
    WindowAttention: partial(patch_method, hilrp_window_attention_forward),
}

_PATCHED = False


def ensure_patched(verbose=False):
    """Apply the class-level patches once (idempotent)."""
    global _PATCHED
    if not _PATCHED:
        monkey_patch(_swin_mod, patch_map=SWIN_PATCH_MAP, verbose=verbose)
        monkey_patch_zennit(verbose=verbose)
        _PATCHED = True


def make_composite(gamma=0.25, conv_gamma=None):
    """zennit composite: gamma rule on every Linear and Conv2d (incl. qkv, proj,
    MLP, patch-merge reduction, head, and the strided patch-embed conv)."""
    conv_gamma = gamma if conv_gamma is None else conv_gamma
    return LayerMapComposite([
        (nn.Conv2d, Gamma(conv_gamma)),
        (nn.Linear, Gamma(gamma)),
    ])


# ---------------------------------------------------------------- attribution
def attribute_swin(model, x, target=None, gamma=0.25, conv_gamma=None, attn_mode="cp"):
    """Single-backward-pass HiLRP attribution for a timm SwinTransformer.

    Args:
        model: timm SwinTransformer in eval mode (unmodified; patching is
               class-level and applied automatically).
        x: input batch [1, 3, H, W], normalized as the model expects.
        target: class index (default: argmax).
        gamma / conv_gamma: zennit Gamma parameters for Linear / Conv2d.

    Returns dict with:
        target, logit  chosen class and its value
        pixel_map      [H, W] relevance at input pixels (through patch-embed conv)
        stage_maps     list of (name, [h, w]) token relevance, coarse-to-fine:
                       tokens 56x56, stage inputs 28/14/7, final norm 7x7
        stage_sums     list of (name, sum(R)/logit) -- ~1.0 up to bias/gamma
                       absorption; the conservation trace
    """
    global ATTN_MODE
    ensure_patched()
    ATTN_MODE = attn_mode
    model.eval()

    captures = []
    hooks = []

    def grab_out(name):
        def hook(module, args, out):
            out.retain_grad()
            captures.append((name, out))
        return hook

    hooks.append(model.patch_embed.register_forward_hook(grab_out("tokens_56")))
    for i, stage in enumerate(model.layers):
        if i > 0:
            hooks.append(stage.downsample.register_forward_hook(grab_out(f"stage{i}_in_{56 // 2 ** i}")))
    hooks.append(model.norm.register_forward_hook(grab_out("final_7")))

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

    stage_maps, stage_sums = [], []
    for name, t in captures:
        R = (t * t.grad)[0]                          # Gradient x Input relevance
        stage_sums.append((name, R.sum().item() / denom))
        stage_maps.append((name, R.sum(-1).detach().cpu()))

    pixel_map = (x * x.grad)[0].sum(0).detach().cpu()

    return dict(
        target=target,
        logit=logit,
        pixel_map=pixel_map,
        stage_maps=stage_maps,
        stage_sums=stage_sums,
        input_map=stage_maps[0][1],
    )
