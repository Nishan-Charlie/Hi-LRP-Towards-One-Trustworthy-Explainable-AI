"""Attribution methods registry and wrappers.

This module provides standardized wrappers around various attribution methods
(Captum, CAM, native) that conform to a common `__call__(inputs, target)` API.
"""

from . import captum_methods
from . import cam_methods
from . import attention_rollout
