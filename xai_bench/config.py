"""Run configuration: a small dataclass plus YAML/dict loading.

The benchmark is fully described by which models, methods, metrics and which
dataset (with how many images and how many seeds) to sweep. Everything else has
a sensible default so a minimal config is just a handful of lines.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass
class RunConfig:
    # What to sweep
    models: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)
    metrics: List[str] = field(default_factory=list)
    # Per-metric keyword overrides, e.g. {"max_sensitivity": {"nr_samples": 5}}.
    # Keeps Quantus metric cost feasible without editing the wrappers.
    metric_kwargs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Dataset
    dataset: str = "demo"
    dataset_kwargs: Dict[str, Any] = field(default_factory=dict)
    num_images: Optional[int] = None      # None = all available
    seeds: List[int] = field(default_factory=lambda: [0])

    # Execution
    device: str = "auto"                  # "auto" | "cuda" | "cpu"
    input_size: int = 224
    output_dir: str = "results"
    run_name: Optional[str] = None        # defaults to a timestamp
    save_saliency: bool = False           # dump raw saliency arrays (large)
    limit_methods_to_supported: bool = True  # skip e.g. rollout on CNNs silently

    def resolved_device(self) -> str:
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunConfig":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def from_yaml(cls, path: str) -> "RunConfig":
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)
