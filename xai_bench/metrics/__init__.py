"""Metrics for evaluating attribution method faithfulness, localization, and sparseness.

Every module here imports quantus. Imports are tolerant so that a bare
torch/timm install can still use `xai_bench.methods.hilrp`; a missing quantus
records a registration error (surfaced by `Registry.get`) rather than making the
whole package unimportable.
"""

from xai_bench.registry import import_optional_submodules

import_optional_submodules(__name__, (
    "localization",
    "faithfulness",
    "sparseness",
    "robustness",
    "randomisation",
    "axiomatic",
))
