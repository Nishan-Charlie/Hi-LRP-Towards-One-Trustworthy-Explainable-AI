"""Build the five grid tables used by the Friedman / Nemenyi and ranking-transfer
statistics (Table 9, Section 6.1) and run grid_stats.py on them.

The cells come from the same loader as the figures and tables
(scripts/bench/benchmark_figures.py): the seeded re-run of the grid with every image
scored (xai_bench/scoring.py), the median for Max-Sensitivity and the mean for the
other metrics. Only the 11 methods defined on all eight backbones enter the tests
(the Friedman procedure needs complete blocks). An infinite MS median (a method that
fails on at least half the images) is kept: the tests use ranks, so it ranks last.

usage (from scripts/revision_neucom): python grid_tables.py
writes results/revision_neucom/analysis/grid_tables.json and grid_tables_stats.json
"""
import json
import os
import subprocess
import sys

import common  # noqa: F401  (cwd -> repo root)

sys.path.insert(0, os.path.join("scripts", "bench"))
import benchmark_figures as bf  # noqa: E402

KEYS = {"PG": "pointing_game", "MS": "max_sensitivity", "FC": "faithfulness_correlation",
        "FE": "faithfulness_estimate", "SP": "sparseness"}
COMPLETE = [m for m, _, _ in bf.METHODS if m not in bf.ATTENTION_NATIVE]

get = bf.load()
T = {k: {me: [get(mo, me, metric) for mo, _, _ in bf.MODELS] for me in COMPLETE} for k, metric in KEYS.items()}
missing = [(k, me, bf.MODELS[j][0]) for k, t in T.items() for me, v in t.items() for j, x in enumerate(v) if x is None]
if missing:
    raise SystemExit(f"incomplete grid, {len(missing)} empty cells, e.g. {missing[:3]}")
path = os.path.join("results", "revision_neucom", "analysis", "grid_tables.json")
with open(path, "w") as f:
    json.dump(T, f, indent=1)   # +inf is written as Infinity, which json.load reads back
print("wrote", path)
subprocess.run([sys.executable, os.path.join("scripts", "revision_neucom", "grid_stats.py"), path], check=True)
