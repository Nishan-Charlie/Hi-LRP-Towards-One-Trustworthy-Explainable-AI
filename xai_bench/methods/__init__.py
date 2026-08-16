"""Attribution methods registry and wrappers.

This module provides standardized wrappers around various attribution methods
(Captum, CAM, native) that conform to a common `__call__(inputs, target)` API.

Imports are tolerant: `captum_methods` and `perturbation_methods` need captum,
which the HiLRP sub-package does not. A missing optional dependency records a
registration error (surfaced by `Registry.get`) instead of making
`xai_bench.methods.hilrp` unimportable.
"""

from xai_bench.registry import import_optional_submodules

import_optional_submodules(__name__, (
    "captum_methods",
    "cam_methods",
    "attention_rollout",
    "perturbation_methods",
    "hilrp_method",    # NOTE: hilrp requires a DEDICATED run (class-level patches)
    "attnlrp_method",  # AttnLRP baseline (Achtibat 2024): ATTN_MODE='attnlrp', flat ViTs only
))
