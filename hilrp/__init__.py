"""HiLRP -- hierarchical layer-wise relevance propagation for resolution-reducing
vision transformers (Swin patch merging, shifted windows, PVT spatial reduction).

The contribution lives in `hierarchical.py`. `rules.py` holds inherited LRP
primitives and `toy_swin.py` is a test-only reference model. Conservation is
guaranteed by the tests in `tests/`, with dtype-tied numeric ceilings; the tests
ARE the "conservation-guaranteed" claim, so they are kept as sharp as the proof.

Internal name only: `HiLRP` is unclaimed in the literature (cf. AttnLRP,
LRP-QViT, CA-LIG) but signals "hierarchical" rather than "resolution-reduction
rules" -- revisit before it goes in a paper.
"""
from .hierarchical import (
    patch_merge_forward, patch_merge_lrp, unconcat_2x2,
    window_partition, window_reverse, cyclic_shift, cyclic_unshift, make_attn_mask,
)

__all__ = [
    "patch_merge_forward", "patch_merge_lrp", "unconcat_2x2",
    "window_partition", "window_reverse", "cyclic_shift", "cyclic_unshift",
    "make_attn_mask",
]
