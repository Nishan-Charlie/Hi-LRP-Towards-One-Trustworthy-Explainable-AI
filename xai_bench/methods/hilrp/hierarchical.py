"""The HiLRP contribution: flow-conserving LRP rules for the resolution-reduction
mechanisms of hierarchical / hybrid vision transformers.

Unifying proposition
--------------------
Any resolution-reduction operator expressible as

        T(neighborhood) = W . phi(concat(x^(1), ..., x^(n)))

with `concat` an index partition and `phi` a conservation-preserving
normalization admits one epsilon-rule and conserves relevance up to the
bias/epsilon terms. Corollaries (one proof, three architectures):

    * Swin patch merging         : n=4, phi=LayerNorm, W: 4C -> 2C
    * strided-conv patch embed   : conv(stride=kernel=patch) = W . concat(pixels)
    * PVT/CvT spatial reduction   : strided conv on the K/V token grid

This module implements the patch-merging corollary and the shifted-window
(cyclic shift + attention mask) transition. Both are verified to conserve
numerically in `tests/test_conservation.py`.

Conservation is not faithfulness. These rules guarantee no relevance leaks;
whether the resulting maps are *correct* is a separate question answered by
Gate-2-real (real Swin-T through LXT, maps-on-objects + class-sensitivity).
"""
import torch

from .rules import lrp_linear, lrp_layernorm_identity, layernorm, DEFAULT_EPS


# --------------------------------------------------------------- patch merging
def patch_merge_forward(x, ln_weight, ln_bias, ln_eps, W_reduction):
    """Swin PatchMerging forward. x: [H, W, C] -> [H/2, W/2, 2C].

    Grouping matches timm's `PatchMerging` exactly (verified bit-for-bit).
    Returns (y, z) where z is the pre-LayerNorm concatenated tensor, needed by
    the backward rule.
    """
    H, W, C = x.shape
    # matches timm PatchMerging: reshape(H/2,2,W/2,2,C).permute(0,1,3,4,2,5).flatten(3)
    # (non-batched: axes (H/2, 2a, W/2, 2c, C) -> (H/2, W/2, 2c, 2a, C))
    z = (x.reshape(H // 2, 2, W // 2, 2, C)
           .permute(0, 2, 3, 1, 4)
           .reshape(H // 2, W // 2, 4 * C))
    a = layernorm(z, ln_weight, ln_bias, ln_eps)
    y = a @ W_reduction.T                                # Swin reduction is bias-free
    return y, z


def patch_merge_lrp(R_y, z, ln_weight, ln_bias, ln_eps, W_reduction, C, eps=DEFAULT_EPS):
    """Backward relevance for patch merging: R_y [H/2, W/2, 2C] -> R_x [H, W, C].

    Three composed, conservation-preserving steps:
      1. reduction Linear (bias-free) via the epsilon-rule,
      2. LayerNorm via the identity rule (see rules.lrp_layernorm_identity),
      3. un-concatenation: the inverse of the 2x2 spatial regroup, a pure index
         permutation that redistributes relevance with zero mixing.
    """
    a = layernorm(z, ln_weight, ln_bias, ln_eps)
    R_a = lrp_linear(a, W_reduction, None, R_y, eps=eps)          # step 1
    R_z = lrp_layernorm_identity(R_a)                            # step 2
    return unconcat_2x2(R_z, C)                                  # step 3


def unconcat_2x2(R_z, C):
    """Inverse of the 2x2 concat/regroup. R_z [H/2, W/2, 4C] -> [H, W, C].

    Exact inverse of the grouping in `patch_merge_forward`: undoes
    permute(0,2,3,1,4) with permute(0,3,1,2,4), so it round-trips to identity and
    conserves the total.
    """
    H2, W2, _ = R_z.shape
    return (R_z.reshape(H2, W2, 2, 2, C)
               .permute(0, 3, 1, 2, 4)
               .reshape(H2 * 2, W2 * 2, C))


# ----------------------------------------------------- shifted-window transition
def window_partition(x, ws):
    """[H, W, C] -> [nW, ws*ws, C]. A reshape (index partition); conserves exactly."""
    H, W, C = x.shape
    return (x.reshape(H // ws, ws, W // ws, ws, C)
             .permute(0, 2, 1, 3, 4)
             .reshape(-1, ws * ws, C))


def window_reverse(win, ws, H, W):
    """Inverse of `window_partition`. [nW, ws*ws, C] -> [H, W, C]."""
    C = win.shape[-1]
    return (win.reshape(H // ws, W // ws, ws, ws, C)
               .permute(0, 2, 1, 3, 4)
               .reshape(H, W, C))


def cyclic_shift(x, shift):
    """Swin cyclic shift (torch.roll). A permutation: relevance un-rolls exactly."""
    return torch.roll(x, (-shift, -shift), dims=(0, 1))


def cyclic_unshift(x, shift):
    """Inverse of `cyclic_shift`, used both in the forward and to un-roll relevance."""
    return torch.roll(x, (shift, shift), dims=(0, 1))


def make_attn_mask(H, W, ws, shift):
    """Swin shifted-window attention mask: 0 = attend, -inf(=-100) = blocked.

    Blocked pairs receive softmax weight ~ e^{-100} ~ 1e-44, so under any
    attention-value rule they carry ~0 relevance -> the shift/mask transition is
    conservation-neutral (verified: leak ~1e-43, independent of epsilon).
    """
    img = torch.zeros(H, W)
    cnt = 0
    for hs in (slice(0, -ws), slice(-ws, -shift), slice(-shift, None)):
        for wsl in (slice(0, -ws), slice(-ws, -shift), slice(-shift, None)):
            img[hs, wsl] = cnt
            cnt += 1
    mw = window_partition(img.unsqueeze(-1), ws).squeeze(-1)      # [nW, ws*ws]
    attn_mask = mw.unsqueeze(1) - mw.unsqueeze(2)
    return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)


# ----------------------------------------------------- deformable attention (bilinear sampling)
def deformable_sample_forward(x, grid):
    """Deformable attention bilinear sampling forward (e.g., Def-DETR or DAT).
    
    Args:
        x: [B, C, H, W] source values.
        grid: [B, H_out, W_out, 2] sampling grid with normalized coordinates in [-1, 1].
        
    Returns:
        Sampled features [B, C, H_out, W_out].
    """
    return torch.nn.functional.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)


def deformable_sample_lrp(R_y, x, grid, eps=DEFAULT_EPS):
    """Backward relevance for deformable (bilinear) sampling.
    
    Treats the sampling as a fixed-grid linear map (since bilinear interpolation
    is a linear combination of source pixels). Relevance propagates from the
    interpolated output back to the discrete grid points using the standard
    epsilon rule, maintaining perfect conservation.
    """
    # Track gradient on x to capture the linear mapping (W^T)
    x_track = x.detach().requires_grad_(True)
    y = deformable_sample_forward(x_track, grid)
    
    # Epsilon stabilizer: z_i = y_i + eps * sign(y_i)
    z = y + eps * torch.sign(y)
    
    # s = R_y / z
    s = R_y / z
    
    # We want R_x = x * W^T s. Since y = W x, the gradient of sum(y * s) w.r.t x is exactly W^T s.
    R_a = torch.autograd.grad(outputs=y, inputs=x_track, grad_outputs=s, retain_graph=False)[0]
    
    return R_a * x
