"""A tiny string-keyed registry used for models, methods, datasets and metrics.

Each registry maps a name -> factory callable. Built-in entries register
themselves at import time via the @REGISTRY.register("name") decorator; users
can add their own the same way without touching the runner.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Iterator, List


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._entries: Dict[str, Callable] = {}

    def register(self, name: str) -> Callable:
        def _decorator(fn: Callable) -> Callable:
            key = name.lower()
            if key in self._entries:
                raise KeyError(f"{self.kind} '{name}' already registered")
            self._entries[key] = fn
            return fn
        return _decorator

    def add(self, name: str, fn: Callable) -> None:
        """Imperative registration (for programmatic / config-driven entries)."""
        self._entries[name.lower()] = fn

    def get(self, name: str) -> Callable:
        key = name.lower()
        if key not in self._entries:
            raise KeyError(
                f"unknown {self.kind} '{name}'. Available: {sorted(self._entries)}"
                + _missing_dependency_hint()
            )
        return self._entries[key]

    def __contains__(self, name: str) -> bool:
        return name.lower() in self._entries

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def names(self) -> List[str]:
        return sorted(self._entries)


MODELS = Registry("model")
METHODS = Registry("method")
DATASETS = Registry("dataset")
METRICS = Registry("metric")


# module path -> the ImportError that stopped its entries from registering.
# A module that fails to import leaves a HOLE in the registries, so a lookup
# miss must say so rather than report a plain "unknown method" and let a missing
# optional dependency look like a typo.
REGISTRATION_ERRORS: Dict[str, BaseException] = {}


def import_optional_submodules(package: str, names: Iterable[str]) -> None:
    """Import each submodule for its registration side effects, tolerantly.

    The benchmark modules pull heavy optional dependencies (captum for the
    Captum-backed methods, quantus for the metrics). Importing them eagerly is
    what populates the registries, but a hard failure used to make the WHOLE
    package unimportable -- including `xai_bench.methods.hilrp`, whose rules are
    deliberately self-contained and need neither. Record the failure instead;
    `Registry.get` surfaces it on the first lookup that comes up empty.
    """
    import importlib

    for name in names:
        try:
            importlib.import_module(f".{name}", package)
        except ImportError as exc:
            REGISTRATION_ERRORS[f"{package}.{name}"] = exc


def _missing_dependency_hint() -> str:
    if not REGISTRATION_ERRORS:
        return ""
    lines = "".join(
        f"\n  - {mod} did not import: {exc}"
        for mod, exc in sorted(REGISTRATION_ERRORS.items())
    )
    return (
        "\nNOTE: some modules failed to import, so their entries are missing"
        f" from this registry:{lines}"
        "\nInstall the benchmark extras to restore them:"
        ' pip install -e ".[extra]"'
    )
