"""xai_bench — a controlled benchmark of attribution methods on ViT foundation models.

See docs/RESEARCH_PLAN.md for the study design. The package is organized as four
registries — models, methods, datasets, metrics — plus a runner that sweeps the
(model x method x dataset x metric) matrix and writes tidy result rows.

Importing the sub-packages is what registers their built-in entries, so it stays
eager. It is also TOLERANT: the benchmark sub-packages pull heavy optional
dependencies (captum for methods, quantus for metrics), and a missing one used to
make the whole package unimportable — including `hilrp`, which
needs neither. HiLRP's rules are deliberately self-contained, so a bare
`torch`/`timm` install must be enough to import and test them.

Failures are recorded, never swallowed: a sub-package that does not import leaves
a hole in its registry, and `Registry.get` reports the underlying ImportError on a
lookup miss (see registry.REGISTRATION_ERRORS).
"""

import importlib

__version__ = "0.2.0"

# Importing hilrp installs the transformers shim LXT needs (hilrp/_compat.py);
# it must precede any LXT import.
import hilrp as _hilrp  # noqa: F401

from .registry import (  # noqa: F401
    Registry, MODELS, METHODS, DATASETS, METRICS, REGISTRATION_ERRORS,
)

# Order matters only for readability; each sub-package registers independently.
_SUBPACKAGES = ("models", "methods", "datasets", "metrics")

for _name in _SUBPACKAGES:
    try:
        importlib.import_module(f".{_name}", __name__)
    except ImportError as _exc:  # missing optional dependency, not a bug
        REGISTRATION_ERRORS[_name] = _exc

del _name, importlib


def registration_report() -> str:
    """Human-readable summary of what registered and what did not."""
    lines = [
        f"{reg.kind}s: {len(reg.names())} registered"
        for reg in (MODELS, METHODS, DATASETS, METRICS)
    ]
    for module, exc in sorted(REGISTRATION_ERRORS.items()):
        lines.append(f"MISSING {module}: {exc}")
    return "\n".join(lines)
