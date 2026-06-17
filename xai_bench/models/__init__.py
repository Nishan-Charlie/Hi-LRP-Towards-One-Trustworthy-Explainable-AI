"""Model registry and wrappers.

This module provides wrappers around timm models that expose common interfaces
needed by attribution methods (e.g., getting the target layer for Grad-CAM).
"""

from . import timm_models
