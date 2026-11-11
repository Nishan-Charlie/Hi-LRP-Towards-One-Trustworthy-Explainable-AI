"""R1.2 / R3.2: top-1 accuracy and confidence of every backbone on the 1,000-image
evaluation set (and on the 100/50-image subsets used by the five-metric grid).

Writes results/revision_neucom/accuracy_per_image.csv and accuracy_summary.csv.
Also records the ViT-B/16 accuracy WITHOUT the input-stats adapter, i.e. under the
ImageNet normalisation that the June full_benchmark_100 run fed to it.
"""
import csv
import sys

import torch

from common import OUT, DEVICE, MAIN_MODELS, MAIN_GRID_N, load_data, load_model, n_params

EXTRA = sys.argv[1:]  # optional extra timm names (capacity study)


@torch.no_grad()
def run_model(name, data, bs=50, raw=False):
    mw = load_model(name)
    model = mw.model.model if (raw and hasattr(mw.model, "in_scale")) else mw.model
    rows = []
    for s in range(0, len(data), bs):
        chunk = data[s:s + bs]
        x = torch.stack([d["image"] for d in chunk]).to(DEVICE)
        p = torch.softmax(model(x).float(), dim=1).cpu()
        top5 = p.topk(5, dim=1).indices
        for d, pi, t5 in zip(chunk, p, top5):
            pred = int(pi.argmax())
            rows.append(dict(model=name + ("@imagenet_norm" if raw else ""), idx=d["idx"], label=d["label"],
                             pred=pred, correct=int(pred == d["label"]), top5=int(d["label"] in t5.tolist()),
                             conf_pred=float(pi[pred]), conf_label=float(pi[d["label"]])))
    npar = n_params(mw)
    del mw
    torch.cuda.empty_cache()
    return rows, npar


def main():
    data = load_data(1000)
    # cache_validation_100.pt must be the first 100 images of the 1000 cache
    try:
        c100 = torch.load("data/ImageNetS/cache_validation_100.pt", weights_only=True)
    except Exception:
        c100 = torch.load("data/ImageNetS/cache_validation_100.pt", weights_only=False)
    same = all(torch.equal(c100[i]["image"], data[i]["image"]) and c100[i]["label"] == data[i]["label"]
               for i in range(100))
    print("cache_validation_100 == first 100 of cache_validation_1000:", same, flush=True)

    all_rows, summ = [], []
    models = MAIN_MODELS + EXTRA
    jobs = [(m, False) for m in models] + [("vit_base_patch16_224", True)]
    for name, raw in jobs:
        rows, npar = run_model(name, data, raw=raw)
        all_rows += rows
        tag = rows[0]["model"]
        for sub, n in (("n1000", 1000), ("grid", MAIN_GRID_N.get(name, 100)), ("n100", 100)):
            r = rows[:n]
            summ.append(dict(model=tag, subset=sub, n=len(r), params_M=round(npar / 1e6, 1),
                             top1=sum(x["correct"] for x in r) / len(r),
                             top5=sum(x["top5"] for x in r) / len(r),
                             conf_pred=sum(x["conf_pred"] for x in r) / len(r),
                             conf_label=sum(x["conf_label"] for x in r) / len(r),
                             conf_pred_correct=(sum(x["conf_pred"] for x in r if x["correct"]) /
                                                max(1, sum(x["correct"] for x in r))),
                             conf_pred_wrong=(sum(x["conf_pred"] for x in r if not x["correct"]) /
                                              max(1, sum(1 - x["correct"] for x in r)))))
        s = [x for x in summ if x["model"] == tag and x["subset"] == "n1000"][0]
        print(f"{tag:40s} params={s['params_M']:6.1f}M top1={s['top1']:.3f} top5={s['top5']:.3f} "
              f"conf={s['conf_pred']:.3f}", flush=True)

    suffix = "_extra" if EXTRA else ""
    with open(f"{OUT}/accuracy_per_image{suffix}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    with open(f"{OUT}/accuracy_summary{suffix}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
        w.writeheader()
        w.writerows(summ)
    print("cache check:", same)


if __name__ == "__main__":
    main()
