"""R1.4: Faithfulness Correlation / Estimate under the three attribution protocols.

Uses the signed maps saved by loc.py (first 100 images, gt target) for the
inherently signed methods, and calls Quantus with the benchmark's settings
(FC: nr_runs=20, subset_size=224; FE: features_in_step=448; Quantus defaults
otherwise: normalise by max |a|, abs=False). Only the map handed to Quantus
changes between protocols.

usage: python ffe.py MODEL N METHOD[,METHOD...]
"""
import csv
import os
import sys

import numpy as np
import quantus
import torch

from common import OUT, DEVICE, load_data, load_model, evaluation_seed, set_seed

model_name, N = sys.argv[1], int(sys.argv[2])
methods = sys.argv[3].split(",")
data = load_data(N)
mw = load_model(model_name)
BS = 25
rows = []
for meth in methods:
    fp = os.path.join(OUT, "maps", f"{model_name}__{meth}.npz")
    if not os.path.exists(fp):
        print("missing", fp)
        continue
    z = np.load(fp)
    idxs = [d["idx"] for d in data if str(d["idx"]) in z.files]
    for proto in ("abs", "pos", "signed"):
        for s in range(0, len(idxs), BS):
            b = idxs[s:s + BS]
            x = np.stack([data[i]["image"].numpy() for i in b])
            y = np.array([data[i]["label"] for i in b])
            a = np.stack([z[str(i)] for i in b])[:, None]
            if proto == "abs":
                a = np.abs(a)
            elif proto == "pos":
                a = np.clip(a, 0, None)
            ok = np.abs(a).reshape(len(b), -1).max(1) > 0
            res = {}
            for mname, cls, kw in (("fc", quantus.FaithfulnessCorrelation, dict(nr_runs=20, subset_size=224)),
                                   ("fe", quantus.FaithfulnessEstimate, dict(features_in_step=448))):
                set_seed(evaluation_seed(0, model_name, f"{meth}|{proto}|{mname}", b[0]))
                metric = cls(disable_warnings=True, display_progressbar=False, return_aggregate=False, **kw)
                try:
                    res[mname] = metric(model=mw.model, x_batch=x[ok], y_batch=y[ok], a_batch=a[ok], device=DEVICE)
                except Exception as e:
                    print("fail", meth, proto, mname, str(e)[:150])
                    res[mname] = [np.nan] * int(ok.sum())
            j = 0
            for i, o in zip(b, ok):
                rows.append(dict(model=model_name, method=meth, protocol=proto, idx=i,
                                 fc=res["fc"][j] if o else np.nan, fe=res["fe"][j] if o else np.nan))
                j += int(o)
        sub = [r for r in rows if r["method"] == meth and r["protocol"] == proto]
        print(f"{model_name} {meth:22s} {proto:6s} FC={np.nanmean([r['fc'] for r in sub]):+.4f} "
              f"FE={np.nanmean([r['fe'] for r in sub]):+.4f} n={len(sub)}", flush=True)
    torch.cuda.empty_cache()
# one file per (model, method set): a second call for the same model (e.g. AttnLRP on
# ViT-B/16) must not overwrite the first
out = os.path.join(OUT, f"ffe_{model_name}_{'_'.join(methods)}_n{N}.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["model", "method", "protocol", "idx", "fc", "fe"])
    w.writeheader()
    w.writerows(rows)
print("DONE", out)
