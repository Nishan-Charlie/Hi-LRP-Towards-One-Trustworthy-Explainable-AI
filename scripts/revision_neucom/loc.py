"""Localisation for every method on one backbone (R1.2, R1.4, R3.2, R3.6).

For each image and method the attribution is computed once per explanation target
and scored under three post-processing protocols:
  abs    |a|              (the benchmark protocol)
  pos    max(a, 0)        (positive evidence only)
  signed a                (raw sign kept; EBPG becomes the net evidence share)
Targets: 'gt' (ground-truth ImageNet label, the benchmark protocol) and 'pred'
(the model's top-1). For correctly classified images both coincide and the gt row
is reused. Metrics: bounding-box Pointing Game (Quantus semantics), dense-mask
Energy-Based Pointing Game (EBPG) and dense peak-in-silhouette PG.

usage: python loc.py MODEL N METHOD[,METHOD...]
AttnLRP must run alone in its own process (class-level patches; see
xai_bench/isolation.py).
"""
import csv
import os
import sys
import time

import numpy as np
import torch

from common import (OUT, DEVICE, load_data, load_model, signed_attribution, protocol_maps,
                    bbox_pg, ebpg, dense_pg, is_degenerate, evaluation_seed, set_seed, METHODS,
                    SIGNED_METHODS)

model_name = sys.argv[1]
N = int(sys.argv[2])
methods = sys.argv[3].split(",")
if "attnlrp" in methods and len(methods) > 1:
    raise SystemExit("attnlrp must run in its own process")
SAVE_MAPS_N = 100   # signed maps of the first 100 images kept for the faithfulness study

data = load_data(N)
mw = load_model(model_name)
is_vit = mw.cam_format in ("tokens_cls", "tokens_nocls")

# predictions (top-1) for the 'pred' target
with torch.no_grad():
    preds = []
    for s in range(0, len(data), 50):
        x = torch.stack([d["image"] for d in data[s:s + 50]]).to(DEVICE)
        preds += mw.model(x).argmax(1).tolist()

fields = ["model", "method", "idx", "label", "pred", "correct", "target_mode", "target", "time_ms",
          "degenerate"] + [f"{m}_{p}" for m in ("bboxpg", "ebpg", "densepg") for p in ("abs", "pos", "signed")]
out_csv = os.path.join(OUT, f"loc_{model_name}_{'_'.join(methods)}_n{N}.csv")
done = set()
if os.path.exists(out_csv):
    with open(out_csv) as f:
        for r in csv.DictReader(f):
            done.add((r["method"], int(r["idx"]), r["target_mode"]))
fh = open(out_csv, "a", newline="")
w = csv.DictWriter(fh, fieldnames=fields)
if not done:
    w.writeheader()

for meth in methods:
    if meth in ("attention_rollout", "attnlrp") and not is_vit:
        print(f"skip {meth}: undefined on {model_name}")
        continue
    reg = None
    if meth in ("saliency", "smoothgrad", "vargrad", "rise", "attention_rollout", "attnlrp"):
        reg = METHODS.get(meth)(model_wrapper=mw, model=mw.model)
    saved = {}
    t_start = time.time()
    for k, d in enumerate(data):
        for mode in ("gt", "pred"):
            if (meth, d["idx"], mode) in done:
                continue
            target = d["label"] if mode == "gt" else preds[k]
            if mode == "pred" and target == d["label"]:
                continue  # identical to the gt row; filled in by the analysis
            set_seed(evaluation_seed(0, model_name, meth, d["idx"]))
            x = d["image"].unsqueeze(0).to(DEVICE)
            t0 = time.time()
            try:
                a = signed_attribution(meth, mw, x, target, registry_fn=reg).float()
            except RuntimeError as e:
                if "CUDA error" in str(e) or "illegal" in str(e) or "CUBLAS" in str(e):
                    fh.flush()
                    raise SystemExit(f"CUDA fault, exiting for a clean retry: {str(e)[:120]}")
                print(f"  FAIL {meth} idx={d['idx']}: {str(e)[:120]}", flush=True)
                torch.cuda.empty_cache()
                continue
            dt = (time.time() - t0) * 1000
            row = dict(model=model_name, method=meth, idx=d["idx"], label=d["label"], pred=preds[k],
                       correct=int(preds[k] == d["label"]), target_mode=mode, target=target,
                       time_ms=round(dt, 1), degenerate=int(is_degenerate(a)))
            mask = d["mask"].to(a.device) if d["mask"] is not None else None
            for p, e in protocol_maps(a, meth).items():
                row[f"bboxpg_{p}"] = bbox_pg(e, d["bboxes"])
                row[f"ebpg_{p}"] = ebpg(e, mask) if mask is not None else np.nan
                row[f"densepg_{p}"] = dense_pg(e, mask) if mask is not None else np.nan
            w.writerow(row)
            if mode == "gt" and d["idx"] < SAVE_MAPS_N and meth in SIGNED_METHODS:
                saved[d["idx"]] = a.cpu().numpy().astype(np.float32)
        if k % 100 == 0:
            fh.flush()
            print(f"{model_name} {meth} {k}/{len(data)} ({time.time() - t_start:.0f}s)", flush=True)
    fh.flush()
    if saved:
        np.savez_compressed(os.path.join(OUT, "maps", f"{model_name}__{meth}.npz"),
                            **{str(i): v for i, v in saved.items()})
    torch.cuda.empty_cache()
fh.close()
print("DONE", out_csv)
