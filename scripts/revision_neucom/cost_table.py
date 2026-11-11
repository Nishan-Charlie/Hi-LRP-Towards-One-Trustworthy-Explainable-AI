"""Aggregate the synchronized timings (time_methods.py) into tab:cost.

Per method: mean time per explanation on each backbone, then the mean over the
backbones where the method is applied (attention-native methods: ViT-B/16 only).
Writes results/revision_neucom/analysis/cost_table.csv (read by
scripts/bench/benchmark_figures.py for the Pareto figure) and prints the table rows.
"""
import glob
import os

import pandas as pd

import common  # noqa: F401  (cwd -> repo root)
from common import OUT, MAIN_MODELS, SHORT

A = os.path.join(OUT, "analysis")
os.makedirs(A, exist_ok=True)
files = glob.glob(os.path.join(OUT, "timing_*.csv"))
if not files:
    raise SystemExit("no timing files")
T = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
T = T[T.model.isin(MAIN_MODELS)].drop_duplicates(["model", "method", "idx"])

per = T.groupby(["method", "model"]).time_ms.agg(["mean", "median", "std", "count"]).reset_index()
per.to_csv(os.path.join(A, "cost_per_backbone.csv"), index=False)
tab = per.groupby("method").agg(time_ms=("mean", "mean"), n_backbones=("model", "nunique"),
                                n_images=("count", "min")).reset_index()
tab = tab.sort_values("time_ms")
tab.to_csv(os.path.join(A, "cost_table.csv"), index=False)

print("GPU:", ", ".join(sorted(T.gpu.unique())))
print(tab.to_string(index=False))
wide = per.pivot(index="method", columns="model", values="mean").reindex(columns=MAIN_MODELS)
wide.columns = [SHORT[c] for c in wide.columns]
print(wide.round(1).to_string())
