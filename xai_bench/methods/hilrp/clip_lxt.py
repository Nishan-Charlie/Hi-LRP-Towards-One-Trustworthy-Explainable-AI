"""HiLRP attribution for a multi-modal model: CLIP image-text similarity.

This is the cross-modal / multi-attention leg of the generalization claim, and
the direct Chefer ICCV'21 differentiator (attribution across modalities). CLIP's
image tower is a standard ViT built from nn.MultiheadAttention, LayerNorm, and
GELU, so HiLRP applies with the same four primitives. The scalar is not a class
logit but the image-text alignment

    s = cos( f_img(x), f_text(caption) )

with the text embedding frozen. HiLRP then answers "what image evidence supports
this text description", a label-free, text-conditioned explanation that no
classifier-head method can produce. The cosine normalization on the image side
is scale-invariant, so we detach its denominator (as in the SSL-scalar branch).

Uses open_clip. Patches are class-level (nn.MultiheadAttention CP, LayerNorm and
GELU identity), so run in a dedicated process.
"""
import torch
import torch.nn as nn

from lxt.efficient.patches import (
    patch_method, non_linear_forward, layer_norm_forward,
    cp_multi_head_attention_forward,
)
from lxt.efficient.zennit_patches import monkey_patch_zennit
from zennit.composites import LayerMapComposite
from zennit.rules import Gamma

_PATCHED = False


def ensure_patched(verbose=False):
    """Patch the torch building blocks CLIP's image tower is made of.
    MultiheadAttention -> CP (stop grad through softmax), LayerNorm/GELU ->
    identity. LayerScale is a per-channel affine and conserves under autograd."""
    global _PATCHED
    if not _PATCHED:
        patch_method(non_linear_forward, nn.GELU, keep_original=True)
        patch_method(layer_norm_forward, nn.LayerNorm)
        patch_method(cp_multi_head_attention_forward, nn.MultiheadAttention, keep_original=True)
        monkey_patch_zennit(verbose=verbose)
        _PATCHED = True


def make_composite(gamma=0.25):
    return LayerMapComposite([(nn.Conv2d, Gamma(gamma)), (nn.Linear, Gamma(gamma))])


def attribute_clip_image_text(model, image, text_embed, gamma=0.25):
    """Attribute the CLIP image-text similarity to input pixels.

    Args:
        model: an open_clip model (already on the right device).
        image: [1, 3, H, W], CLIP-normalized.
        text_embed: precomputed, L2-normalized, detached text embedding [D].
        gamma: zennit Gamma for Linear/Conv.

    Returns dict: similarity, pixel_map [H, W].
    """
    ensure_patched()
    model.eval()
    x = image.clone().detach().requires_grad_(True)
    composite = make_composite(gamma)
    with composite.context(model.visual) as vis:
        img = vis(x)                                   # projected image embedding [1, D]
        img = img / img.norm(dim=-1, keepdim=True).detach()   # scale-invariant -> detach denom
        s = (img[0] * text_embed).sum()
        model.zero_grad(set_to_none=True)
        s.backward()
    pixel_map = (x * x.grad)[0].sum(0).detach().cpu()
    return dict(similarity=s.item(), pixel_map=pixel_map)
