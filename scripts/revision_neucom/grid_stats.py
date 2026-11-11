"""Friedman/Nemenyi tests and ranking-transfer statistics on the 11 x 8 grid
(methods with complete backbone coverage). Reproduces Table 9 and the Spearman
transfer numbers of Section 6.1 from a given set of tables.

usage: python grid_stats.py tables.json
tables.json: {"PG": {method: [8 values in backbone order]}, "MS": ..., ...}
Lower-is-better metrics: MS. Higher-is-better: PG, FC, FE, SP.
"""
import itertools
import json
import sys

import numpy as np
from scipy import stats

T = json.load(open(sys.argv[1]))
LOWER = {"MS"}
K_METH = None
rng = np.random.default_rng(0)


def ranks(M, lower):
    # rank methods within each backbone (column); 1 = best
    return np.column_stack([stats.rankdata(M[:, j] if lower else -M[:, j]) for j in range(M.shape[1])])


def nemenyi_cd(k, n, alpha=0.05):
    q = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102,
         10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313}[k]
    return q * np.sqrt(k * (k + 1) / (6.0 * n))


out = {}
for metric, tab in T.items():
    names = list(tab.keys())
    M = np.array([tab[m] for m in names], float)
    k, n = M.shape
    chi2, p = stats.friedmanchisquare(*[M[i] for i in range(k)])
    R = ranks(M, metric in LOWER).mean(axis=1)
    best = int(np.argmin(R))
    cd = nemenyi_cd(k, n)
    beyond = [names[i] for i in range(k) if R[i] - R[best] > cd]
    out[metric] = dict(chi2=round(chi2, 1), p=float(p), p_bonf=min(1.0, float(p) * len(T)),
                       best=names[best], best_rank=round(float(R[best]), 2), cd=round(cd, 2),
                       beyond_cd=beyond, mean_ranks={m: round(float(r), 2) for m, r in zip(names, R)})
    print(f"{metric}: chi2={chi2:.1f} p={p:.2e} pBonf={min(1, p * len(T)):.2e} best={names[best]}"
          f"({R[best]:.2f}) CD={cd:.2f} beyond={beyond}")

# ranking transfer on PG: mean pairwise Spearman between backbone rankings
M = np.array([T["PG"][m] for m in T["PG"]], float)
k, n = M.shape


def mean_pairwise(Mx):
    rs = [stats.spearmanr(Mx[:, a], Mx[:, b])[0] for a, b in itertools.combinations(range(Mx.shape[1]), 2)]
    return float(np.nanmean(rs))


obs = mean_pairwise(M)
null = []
for _ in range(10000):
    Mp = np.column_stack([rng.permutation(M[:, j]) for j in range(n)])
    null.append(mean_pairwise(Mp))
null = np.array(null)
# 95% CI: bootstrap over the 28 pairwise backbone correlations (the procedure of the
# original submission; it treats the pairs as exchangeable)
pair_r = np.array([stats.spearmanr(M[:, a], M[:, b])[0] for a, b in itertools.combinations(range(n), 2)])
boot = [rng.choice(pair_r, len(pair_r)).mean() for _ in range(5000)]
lo, hi = np.percentile(boot, [2.5, 97.5])
p_perm = (np.sum(null >= obs) + 1) / (len(null) + 1)
cnn_eff = stats.spearmanr(M[:, 0], M[:, -1])[0]
print(f"PG transfer: mean pairwise Spearman={obs:.2f} 95%CI=[{lo:.2f},{hi:.2f}] null={null.mean():.2f}+-{null.std():.2f} "
      f"p_perm={p_perm:.1e} CNN-vs-EffViT-B2={cnn_eff:.2f}")
out["transfer"] = dict(mean_spearman=obs, ci=[lo, hi], null_mean=float(null.mean()), null_sd=float(null.std()),
                       p_perm=p_perm, cnn_vs_effb2=float(cnn_eff))
json.dump(out, open(sys.argv[1].replace(".json", "_stats.json"), "w"), indent=1)
