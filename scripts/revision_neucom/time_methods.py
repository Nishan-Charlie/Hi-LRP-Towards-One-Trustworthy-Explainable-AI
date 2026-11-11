"""Cost table (tab:cost) re-timing with synchronized GPU timing.

The original table timed each explanation with time.time() and no
torch.cuda.synchronize(), so queued GPU work could fall outside the timed window,
and it averaged RN-50, ViT-B/16 and DeiT-B (DeiT-B is not a benchmark backbone).
Here every method is called exactly as the benchmark runner calls it (registry
implementation, model loaded as in xai_bench.runner, same seeding) on the eight
benchmark backbones; attention-native methods only where the runner applies them.
Images are spread evenly over the 1,000-image evaluation set; the cost of these
methods does not depend on image content (fixed 224x224 input, fixed sample counts).

usage: python time_methods.py MODEL N METHOD[,METHOD...]
AttnLRP must run alone in its own process (class-level patches).
"""
import csv
import os
import sys
import time

import torch

from common import OUT, DEVICE, load_data, evaluation_seed, set_seed, MODELS, METHODS

model_name = sys.argv[1]
N = int(sys.argv[2])
methods = sys.argv[3].split(",")
if "attnlrp" in methods and len(methods) > 1:
    raise SystemExit("attnlrp must run in its own process")
WARMUP = 3

data = load_data(1000)
step = max(1, len(data) // N)
data = data[::step][:N]

mw = MODELS.get(model_name)()          # exactly as xai_bench.runner
mw.model.to(DEVICE)
mw.model.eval()
supports_attention = getattr(mw, "supports_attention", mw.is_vit)

out_csv = os.path.join(OUT, f"timing_{model_name}_{'_'.join(methods)}_n{N}.csv")
done = set()
if os.path.exists(out_csv):
    with open(out_csv) as f:
        done = {(r["method"], int(r["idx"])) for r in csv.DictReader(f)}
fh = open(out_csv, "a", newline="")
w = csv.DictWriter(fh, fieldnames=["model", "method", "idx", "time_ms", "gpu"])
if not done:
    w.writeheader()
gpu = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu"


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def call(method, x, target):
    attr = method(x, target=target)
    if isinstance(attr, tuple):
        attr = tuple(a.detach() if isinstance(a, torch.Tensor) else a for a in attr)
    elif isinstance(attr, torch.Tensor):
        attr = attr.detach()
    return attr


for meth in methods:
    if meth in ("attention_rollout", "attention_gradient", "attnlrp") and not supports_attention:
        print(f"skip {meth}: not applied on {model_name}")
        continue
    todo = [d for d in data if (meth, d["idx"]) not in done]
    if not todo:
        continue
    method = METHODS.get(meth)(model_wrapper=mw, model=mw.model)
    x0 = data[0]["image"].unsqueeze(0).to(DEVICE)
    for _ in range(WARMUP):            # lazy init, cuDNN autotuning, allocator growth
        call(method, x0, data[0]["label"])
    sync()
    t_start = time.time()
    for k, d in enumerate(todo):
        set_seed(evaluation_seed(0, model_name, meth, d["idx"]))
        x = d["image"].unsqueeze(0).to(DEVICE)
        sync()
        t0 = time.perf_counter()
        try:
            call(method, x, d["label"])
            sync()
        except RuntimeError as e:
            if "CUDA error" in str(e) or "illegal" in str(e) or "CUBLAS" in str(e):
                fh.flush()
                raise SystemExit(f"CUDA fault, exiting for a clean retry: {str(e)[:120]}")
            print(f"  FAIL {meth} idx={d['idx']}: {str(e)[:120]}", flush=True)
            continue
        dt = (time.perf_counter() - t0) * 1000
        w.writerow(dict(model=model_name, method=meth, idx=d["idx"], time_ms=round(dt, 2), gpu=gpu))
        if k % 10 == 0:
            fh.flush()
            print(f"{model_name} {meth} {k}/{len(todo)} ({time.time() - t_start:.0f}s)", flush=True)
    fh.flush()
    torch.cuda.empty_cache()
fh.close()
print("DONE", out_csv)
