"""Rewrite summary.csv and failures.json of finished runner runs from their
results.json with the current aggregation (xai_bench.runner, xai_bench.scoring),
without loading any model. Use it after changing how cells are scored.

usage (from scripts/revision_neucom): python resummarize.py RUN_NAME [RUN_NAME ...]
"""
import json
import os
import sys
from types import SimpleNamespace

import common  # noqa: F401  (cwd -> repo root)

from xai_bench.runner import BenchmarkRunner  # noqa: E402

for run in sys.argv[1:]:
    run_dir = os.path.join("results", run)
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(run_dir, "results.json")) as f:
        results = json.load(f)
    r = object.__new__(BenchmarkRunner)          # aggregation only: no model, no dataset
    r.config, r.run_dir = SimpleNamespace(metrics=cfg["metrics"]), run_dir
    r._write_csv(results)
    init = {}
    fp = os.path.join(run_dir, "failures.json")
    if os.path.exists(fp):
        with open(fp) as f:
            init = json.load(f).get("method_init", {})
    r._write_failures(results, init)
    print("resummarized", run_dir)
