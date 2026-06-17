"""xai_bench — a controlled benchmark of attribution methods on ViT foundation models.

See RESEARCH_PLAN.md for the study design. The package is organized as four
registries — models, methods, datasets, metrics — plus a runner that sweeps the
(model x method x dataset x metric) matrix and writes tidy result rows.
"""

__version__ = "0.1.0"

from .registry import Registry, MODELS, METHODS, DATASETS, METRICS  # noqa: F401

# Importing the sub-packages triggers registration of their built-in entries.
from . import models as _models  # noqa: F401
from . import methods as _methods  # noqa: F401
from . import datasets as _datasets  # noqa: F401
from . import metrics as _metrics  # noqa: F401
