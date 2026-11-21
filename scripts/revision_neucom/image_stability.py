"""How much do the method rankings depend on which images are evaluated? (CPU only)

1. Localization on the 1,000 images: Spearman correlation between the method rankings on
   images 1-500 and 501-1,000 (class-ordered split: the halves share few classes), per backbone.
2. Five-metric grid on the 500 images: Spearman correlation between the rankings on two
   random halves of 250 images (median over 1,000 splits), per backbone and metric.
Cells are scored as in the paper (xai_bench.scoring; MS cells are medians).

usage: python scripts/revision_neucom/image_stability.py
"""
import glob
import os

import common  # noqa: F401  (cwd -> repo root)
import numpy as np
import pandas as pd
from common import MAIN_MODELS, OUT
from scipy.stats import spearmanr

from xai_bench.scoring import score

rng = np.random.default_rng(0)


def rho(a, b):
    a, b = a.align(b, join="inner")
    big = {np.inf: 1e9, -np.inf: -1e9}
    return spearmanr(a.replace(big), b.replace(big)).statistic


# ---- 1. localization, images 1-500 vs 501-1,000
loc = pd.concat([pd.read_csv(f) for m in MAIN_MODELS
                 for f in glob.glob(os.path.join(OUT, f"loc_{m}_*_n1000.csv"))])
loc = score(loc.rename(columns={"degenerate": "degenerate_map"}), ["ebpg_abs", "bboxpg_abs"])
first, second = set(loc[loc.idx < 500].label), set(loc[loc.idx >= 500].label)
print(f"classes: images 1-500 {len(first)}, 501-1000 {len(second)}, shared {len(first & second)}")
rows = []
for m, d in loc.groupby("model"):
    for met in ("ebpg_abs", "bboxpg_abs"):
        W = d.pivot_table(index="idx", columns="method", values=met)
        rows.append(dict(model=m, metric=met, rho=rho(W[W.index < 500].mean(), W[W.index >= 500].mean())))
L = pd.DataFrame(rows)
print(L.groupby("metric").rho.median().round(2).to_string())

# ---- 2. grid, two random halves of the 500 images
MET = ["pointing_game", "faithfulness_correlation", "faithfulness_estimate", "max_sensitivity", "sparseness"]
g = score(pd.read_csv(os.path.join("results", "revision_grid_500", "results.csv")), MET)
rows = []
for m, d in g.groupby("model"):
    for met in MET:
        W = d.pivot(index="image_idx", columns="method", values=met)
        stat = (lambda x: x.median()) if met == "max_sensitivity" else (lambda x: x.mean())
        r = []
        for _ in range(1000):
            p = rng.permutation(len(W))
            r.append(rho(stat(W.iloc[p[:len(W) // 2]]), stat(W.iloc[p[len(W) // 2:]])))
        rows.append(dict(model=m, metric=met, rho=np.median(r)))
G = pd.DataFrame(rows)
print(G.groupby("metric").rho.median().round(2).to_string())
