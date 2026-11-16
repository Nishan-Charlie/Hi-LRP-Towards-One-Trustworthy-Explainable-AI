"""How images whose explanation carries no information are scored.

A constant attribution map (the all-zero Grad-CAM++ map after the final ReLU when
its channel weights cross the pole, an all-zero LIME surrogate, ...) identifies no
location and orders no features, so the evaluation metrics are undefined on it:
the Pointing Game has no peak, the correlations of Faithfulness Correlation and
Faithfulness Estimate have zero variance, the Gini index of Sparseness is 0/0, and
the relative change measured by Max-Sensitivity divides by a zero norm.

Dropping those images would average a method over the images on which it happens
to work (Grad-CAM++ on MaxViT-S returns a constant map on most images). They are
scored instead, so every cell averages every image:

* no information earns no credit: a constant map scores 0 on the Pointing Game,
  the energy-based Pointing Game, Faithfulness Correlation, Faithfulness Estimate
  and Sparseness;
* Max-Sensitivity, unbounded when the reference map is zero (and undefined when a
  method returns constant maps for the perturbed inputs), takes the largest value
  the method reaches on that backbone, a finite worst case.

Undefined values that are not caused by the explanation (an image without a
bounding box, for example) are left undefined and counted separately.
"""
from __future__ import annotations

import numpy as np
import torch

#: metrics on which a constant map scores 0 (bench metric names and the
#: localization columns of scripts/revision_neucom/loc.py)
ZERO_CREDIT = ("pointing_game", "faithfulness_correlation", "faithfulness_estimate",
               "sparseness", "ebpg", "bboxpg", "densepg")
#: metrics on which an undefined value takes the worst value of its group
WORST_OBSERVED = ("max_sensitivity", "ms")


def is_constant_map(a) -> bool:
    """True for a map with a non-finite entry or with a single value everywhere."""
    if isinstance(a, torch.Tensor):
        return bool((not torch.isfinite(a).all()) or float(a.max() - a.min()) == 0.0)
    a = np.asarray(a, dtype=float)
    return bool((not np.isfinite(a).all()) or float(a.max() - a.min()) == 0.0)


def _base(metric: str) -> str:
    """'ebpg_abs' -> 'ebpg', 'max_sensitivity' -> 'max_sensitivity', 'ms3' -> 'ms'."""
    for name in ZERO_CREDIT + WORST_OBSERVED:
        if metric == name or metric.startswith(name + "_") or (name == "ms" and metric[:2] == "ms"):
            return name
    return metric


def score(df, metrics, group=("model", "method"), constant_col="degenerate_map", failed_col="error"):
    """Return a copy of a per-image table with the convention above applied.

    ``constant_col`` flags rows whose map is constant; rows with a non-empty
    ``failed_col`` (an attribution that raised) are left untouched, they are
    failures to be retried, not scores.
    """
    import pandas as pd

    out = df.copy()
    ok = out[failed_col].isna() if failed_col in out.columns else pd.Series(True, index=out.index)
    # True / 1 flags a constant map; False, 0 and missing values do not
    const = (out[constant_col].eq(True) & ok) if constant_col in out.columns \
        else pd.Series(False, index=out.index)
    for m in metrics:
        if m not in out.columns:
            continue
        out[m] = pd.to_numeric(out[m], errors="coerce").replace([np.inf, -np.inf], np.nan)
        base = _base(m)
        if base in ZERO_CREDIT:
            out.loc[const, m] = 0.0
        elif base in WORST_OBSERVED:
            worst = out.groupby(list(group))[m].transform("max")
            undefined = out[m].isna() & ok
            out.loc[undefined, m] = worst[undefined]
    return out
