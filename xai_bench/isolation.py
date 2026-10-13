"""Process isolation for attribution methods that patch classes.

HiLRP and the AttnLRP baseline run on the LXT efficient backend, which rewrites
``forward`` on CLASSES (nn.LayerNorm, nn.GELU, the timm attention blocks), not on
instances. A patch is process-global and is never undone. Forward numerics stay
identical, but every other method's GRADIENTS in the same interpreter then flow
through the patched rules, so a baseline computed after a patch is silently a
different method and its scores are not comparable.

That used to be a convention in docstrings. It is enforced here, in two places:

  * ``check_run``     static, when a RunConfig is built: a patching method must be
                      the only entry in ``methods``.
  * ``check_process`` dynamic, when a run starts: a run that needs unpatched
                      gradients refuses to start in an interpreter where a patch
                      is already live (a second runner call, a notebook cell, a
                      script that called a port directly).

Deliberately dependency-free (no torch, no LXT) so config validation stays cheap
and importable in the minimal test environment.
"""

from __future__ import annotations

import sys
from typing import Iterable, List

# Registry names of the methods whose setup applies class-level patches.
PATCHING_METHODS = frozenset({"hilrp", "attnlrp"})

# Backbones on which `attnlrp` runs the published flat-ViT recipe. Everything
# else gets the naive generic recipe (see methods/attnlrp_method.py).
FLAT_VIT_PREFIXES = ("vit", "deit", "beit")

_ATTNLRP_MODULE = "xai_bench.methods.attnlrp_method"


def is_flat_vit(model_name: str) -> bool:
    return model_name.lower().startswith(FLAT_VIT_PREFIXES)


def _patching(methods: Iterable[str]) -> List[str]:
    return sorted({m.lower() for m in methods} & PATCHING_METHODS)


def check_run(methods: Iterable[str], models: Iterable[str]) -> None:
    """Reject a (methods, models) sweep that cannot be honest in one process."""
    methods, models = list(methods), list(models)
    patching = _patching(methods)

    if patching and len({m.lower() for m in methods}) > 1:
        raise ValueError(
            f"methods {methods} cannot share one process: {patching} applies "
            "class-level forward patches (LayerNorm, GELU, attention) that alter "
            "the gradients of every other method. Give each of "
            f"{patching} its own config with a single-entry `methods:` list "
            "(see configs/hilrp_grid.yaml) and run the rest separately."
        )

    if patching == ["attnlrp"]:
        flat = [m for m in models if is_flat_vit(m)]
        other = [m for m in models if not is_flat_vit(m)]
        if flat and other:
            raise ValueError(
                f"attnlrp cannot sweep flat ViTs {flat} and other backbones "
                f"{other} in one process: the flat-ViT recipe patches the timm "
                "norm subclasses, which silently changes the naive baseline run "
                "on the others. Split the models across two configs (see "
                "configs/attnlrp_grid_vit.yaml and configs/attnlrp_grid_hier.yaml)."
            )


def live_patches() -> List[str]:
    """Modules whose class-level patches are live in this interpreter.

    Reads the ports' own idempotency flags, so it also sees patches applied by a
    direct port call that never went through the runner. Looked up in
    sys.modules rather than imported: a package that was never imported cannot
    have patched anything, and this module must stay dependency-free.
    """
    hilrp = sys.modules.get("hilrp")
    live = list(hilrp.patched_ports()) if hilrp is not None else []
    if getattr(sys.modules.get(_ATTNLRP_MODULE), "_NAIVE_PATCHED", False):
        live.append(_ATTNLRP_MODULE)
    return sorted(live)


def check_process(methods: Iterable[str], models: Iterable[str]) -> None:
    """Refuse to start a run whose gradients this interpreter can no longer give."""
    methods, models = list(methods), list(models)
    patching = _patching(methods)
    live = live_patches()

    if patching == ["hilrp"]:
        return                      # HiLRP's own ports are designed to coexist
    if patching == ["attnlrp"]:
        if all(is_flat_vit(m) for m in models):
            return                  # flat recipe IS a HiLRP port (vit_lxt)
        live = [m for m in live if m != _ATTNLRP_MODULE]
        what = "the naive attnlrp baseline"
    else:
        what = f"methods {methods}"

    if live:
        raise RuntimeError(
            f"{what} would run through patched gradients: class-level LRP "
            f"patches are already live in this process (applied by {live}) and "
            "cannot be undone. Start a fresh Python process for this run."
        )
