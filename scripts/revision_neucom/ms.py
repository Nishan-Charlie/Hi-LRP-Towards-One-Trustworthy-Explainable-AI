"""Multi-seed Max-Sensitivity (R3.5) and the Grad-CAM++ stabilisation ablation (R3.4).

Re-implements quantus.MaxSensitivity exactly as configured in the benchmark
(uniform noise U(-0.2, 0.2) on the normalised input, similarity = difference,
Frobenius norms, normalise=False, abs=False, max over samples), but stores every
per-sample sensitivity so MS@3 (the original budget) and MS@K are read from the
same perturbation draws. Each (seed, model, method, image) is seeded exactly as
in xai_bench.runner.

usage: python ms.py MODEL N SEEDS(0,1,2) K METHOD[,METHOD...]
pseudo-method 'grad_cam_plus_plus_stab': Grad-CAM++ with |2 + S*g| >= 1, i.e.
alpha bounded by 1; differs from Grad-CAM++ only near the pole of alpha.
"""
import csv
import os
import sys
import time

import numpy as np
import torch

from common import (OUT, DEVICE, load_data, load_model, signed_attribution, cam_map,
                    evaluation_seed, set_seed, METHODS)

model_name, N = sys.argv[1], int(sys.argv[2])
seeds = [int(s) for s in sys.argv[3].split(",")]
K = int(sys.argv[4])
methods = sys.argv[5].split(",")
if "attnlrp" in methods and len(methods) > 1:
    raise SystemExit("attnlrp must run in its own process")

data = load_data(N)
mw = load_model(model_name)
is_vit = mw.cam_format in ("tokens_cls", "tokens_nocls")


def explainer(meth):
    reg = None
    if meth in ("saliency", "smoothgrad", "vargrad", "rise", "attention_rollout", "attnlrp"):
        reg = METHODS.get(meth)(model_wrapper=mw, model=mw.model)

    def f(x, t):
        if meth == "grad_cam_plus_plus_stab":
            return cam_map(mw, x, t, plus_plus=True, alpha_floor=1.0)
        a = signed_attribution(meth, mw, x, t, registry_fn=reg).float()
        return a if meth == "attnlrp" else a.abs()   # = registry output fed to Quantus
    return f


out_csv = os.path.join(OUT, f"ms_{model_name}_{'_'.join(methods)}_n{N}_s{len(seeds)}_k{K}.csv")
fields = ["model", "method", "seed", "idx", "label", "a_norm", "degenerate", "time_ms"] + [f"s{j}" for j in range(K)]
done = set()
if os.path.exists(out_csv):
    with open(out_csv) as f:
        for r in csv.DictReader(f):
            done.add((r["method"], int(r["seed"]), int(r["idx"])))
fh = open(out_csv, "a", newline="")
w = csv.DictWriter(fh, fieldnames=fields)
if not done:
    w.writeheader()

for meth in methods:
    if meth in ("attention_rollout", "attnlrp") and not is_vit:
        continue
    f = explainer(meth)
    t_start = time.time()
    for seed in seeds:
        for k, d in enumerate(data):
            if (meth, seed, d["idx"]) in done:
                continue
            set_seed(evaluation_seed(seed, model_name, meth, d["idx"]))
            x = d["image"].unsqueeze(0).to(DEVICE)
            t0 = time.time()
            try:
                a = f(x, d["label"]).detach()
                an = float(torch.linalg.norm(a))
                row = dict(model=model_name, method=meth, seed=seed, idx=d["idx"], label=d["label"],
                           a_norm=an, degenerate=int(not np.isfinite(an) or float(a.max() - a.min()) == 0))
                for j in range(K):
                    noise = np.random.uniform(low=-0.2, high=0.2, size=tuple(x.shape))
                    xp = x + torch.tensor(noise, dtype=x.dtype, device=x.device)
                    ap = f(xp, d["label"]).detach()
                    row[f"s{j}"] = float(torch.linalg.norm(ap - a) / an) if an > 0 else np.nan
            except RuntimeError as e:
                if "CUDA error" in str(e) or "illegal" in str(e) or "CUBLAS" in str(e):
                    fh.flush()
                    raise SystemExit(f"CUDA fault, exiting for a clean retry: {str(e)[:120]}")
                print(f"  FAIL {meth} seed={seed} idx={d['idx']}: {str(e)[:120]}", flush=True)
                torch.cuda.empty_cache()
                continue
            row["time_ms"] = round((time.time() - t0) * 1000, 1)
            w.writerow(row)
            if k % 25 == 0:
                fh.flush()
                print(f"{model_name} {meth} seed{seed} {k}/{len(data)} ({time.time() - t_start:.0f}s)", flush=True)
    fh.flush()
    torch.cuda.empty_cache()
fh.close()
print("DONE", out_csv)
