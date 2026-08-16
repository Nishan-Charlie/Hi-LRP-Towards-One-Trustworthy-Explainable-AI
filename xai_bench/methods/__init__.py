"""Attribution methods registry and wrappers.

This module provides standardized wrappers around various attribution methods
(Captum, CAM, native) that conform to a common `__call__(inputs, target)` API.
"""

from . import captum_methods
from . import cam_methods
from . import attention_rollout
from . import perturbation_methods
from . import hilrp_method   # NOTE: hilrp requires a DEDICATED run (class-level patches)
from . import attnlrp_method  # AttnLRP baseline (Achtibat 2024): ATTN_MODE='attnlrp', flat ViTs only
