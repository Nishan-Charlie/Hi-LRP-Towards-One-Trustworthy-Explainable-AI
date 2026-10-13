"""HiLRP: relevance propagation for hierarchical and hybrid Vision Transformers.

    import hilrp
    res = hilrp.attribute(model, x)        # any supported timm backbone
    res["pixel_map"]                       # [H, W] signed relevance
    res["stage_sums"]                      # conservation trace, input to head

Layout
------
``api``                    ``attribute`` and the per-family dispatch
``*_lxt``                  one port per architecture family, on the LXT backend
``hierarchical``           THE CONTRIBUTION: rules for patch merging, shifted
                           windows and the un-concat regroup
``attention_primitives``   the four primitives, float64 reference implementation
``rules``                  inherited LRP primitives, dependency-free
``viz``                    ``normalize_for_display``; use it for all rendering
``tests``, ``tests_backend``   reference-rule and production-port suites

Importing ``hilrp`` needs only torch. The ports additionally need ``lxt`` and
``zennit`` and are imported on first use.
"""

__version__ = "0.2.0"

# Must precede any LXT import; see hilrp/_compat.py.
from . import _compat as _compat  # noqa: F401

from .api import attribute, attributor_for, patched_ports, supported_architectures
from .hierarchical import (
    patch_merge_forward, patch_merge_lrp, unconcat_2x2,
    window_partition, window_reverse, cyclic_shift, cyclic_unshift, make_attn_mask,
)
from .viz import normalize_for_display, show_relevance

__all__ = [
    "attribute", "attributor_for", "supported_architectures", "patched_ports",
    "normalize_for_display", "show_relevance",
    "patch_merge_forward", "patch_merge_lrp", "unconcat_2x2",
    "window_partition", "window_reverse", "cyclic_shift", "cyclic_unshift",
    "make_attn_mask",
]
