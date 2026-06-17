"""A tiny string-keyed registry used for models, methods, datasets and metrics.

Each registry maps a name -> factory callable. Built-in entries register
themselves at import time via the @REGISTRY.register("name") decorator; users
can add their own the same way without touching the runner.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterator, List


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
