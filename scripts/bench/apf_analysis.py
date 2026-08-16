"""Analyze the APF vs standard Faithfulness Correlation (FC) ablation.

Reads results/apf_ablation/results.json (paired per-image FC and APF), then for
each (model, method) reports the paired comparison: n, FC mean/std, APF mean/std,
the mean paired shift (APF - FC), and a Wilcoxon signed-rank p-value. Also
aggregates per model to test the attention-sink prediction (the shift should be
largest on the global-softmax ViT-B/16 and negligible on linear-attention /
CNN backbones), and reports whether APF shrinks the per-image noise band.

Emits a plain-text table plus LaTeX rows for the paper. No new model inference.
"""
import json
import numpy as np
from scipy.stats import wilcoxon

RESULTS = "results/apf_ablation/results.json"

MODEL_NAMES = {
    "vit_base_patch16_224": "ViT-B/16",
    "swin_base_patch4_window7_224": "Swin-B",
    "efficientvit_b2": "EffViT-B2",
    "resnet50": "RN-50",
}
MODEL_ORDER = ["vit_base_patch16_224", "swin_base_patch4_window7_224",
               "efficientvit_b2", "resnet50"]
MODEL_ATTN = {
    "vit_base_patch16_224": "global softmax",
    "swin_base_patch4_window7_224": "windowed softmax",
    "efficientvit_b2": "linear (no softmax)",
    "resnet50": "none (CNN)",
}
METHOD_NAMES = {
    "saliency": "Saliency",
    "input_x_gradient": "Input$\\times$Grad",
    "smoothgrad": "SmoothGrad",
    "vargrad": "VarGrad",
    "grad_cam": "Grad-CAM",
}
METHOD_ORDER = ["saliency", "input_x_gradient", "smoothgrad", "vargrad", "grad_cam"]


def load_pairs(rows):
    from collections import defaultdict
    d = defaultdict(list)
    for r in rows:
        fc = r.get("faithfulness_correlation")
        ap = r.get("apf")
        if fc is None or ap is None:
            continue
        d[(r["model"], r["method"])].append((float(fc), float(ap)))
    return d


def main():
    rows = json.load(open(RESULTS))
    pairs = load_pairs(rows)

    print(f"{'Model':10s} {'Method':13s} {'n':>4s} "
          f"{'FC mean':>8s} {'FC std':>7s} {'APF mean':>9s} {'APF std':>8s} "
          f"{'shift':>7s} {'Wilcoxon p':>11s}")
    print("-" * 82)

    latex_rows = []
    per_model_shift = {m: [] for m in MODEL_ORDER}
    per_model_signed = {m: [] for m in MODEL_ORDER}
    per_model_fcstd = {m: [] for m in MODEL_ORDER}
    per_model_apstd = {m: [] for m in MODEL_ORDER}

    for model in MODEL_ORDER:
        for method in METHOD_ORDER:
            key = (model, method)
            if key not in pairs or len(pairs[key]) < 20:
                latex_rows.append((model, method, None))
                continue
            arr = np.array(pairs[key])
            fc, ap = arr[:, 0], arr[:, 1]
            n = len(fc)
            shift = float((ap - fc).mean())
            try:
                p = wilcoxon(ap, fc, zero_method="wilcox").pvalue
            except ValueError:
                p = float("nan")
            per_model_shift[model].append(abs(shift))
            per_model_signed[model].append(shift)
            per_model_fcstd[model].append(fc.std())
            per_model_apstd[model].append(ap.std())
            print(f"{MODEL_NAMES[model]:10s} {method:13s} {n:4d} "
                  f"{fc.mean():8.4f} {fc.std():7.3f} {ap.mean():9.4f} {ap.std():8.3f} "
                  f"{shift:+7.4f} {p:11.3g}")
            latex_rows.append((model, method,
                               dict(n=n, fcm=fc.mean(), fcs=fc.std(),
                                    apm=ap.mean(), aps=ap.std(), shift=shift, p=p)))
        print()

    print("=" * 82)
    print("Per-model summary (attention-sink prediction: net positive shift largest on global softmax):")
    print(f"{'Model':10s} {'attention':20s} {'net shift':>10s} {'mean|shift|':>11s} {'mean FC std':>12s} {'mean APF std':>13s}")
    for model in MODEL_ORDER:
        if not per_model_shift[model]:
            continue
        net = np.mean(per_model_signed[model])
        ms = np.mean(per_model_shift[model])
        fs = np.mean(per_model_fcstd[model])
        as_ = np.mean(per_model_apstd[model])
        print(f"{MODEL_NAMES[model]:10s} {MODEL_ATTN[model]:20s} {net:+10.4f} {ms:11.4f} {fs:12.3f} {as_:13.3f}")

    # LaTeX table body
    print("\n" + "=" * 82)
    print("LaTeX rows (model-grouped):")
    for model in MODEL_ORDER:
        block = [r for r in latex_rows if r[0] == model]
        printed_model = False
        for (_, method, s) in block:
            mlabel = MODEL_NAMES[model] if not printed_model else ""
            printed_model = True
            if s is None:
                print(f"{mlabel} & {METHOD_NAMES[method]} & -- & -- & -- & -- \\\\")
            else:
                star = "$^{*}$" if (s["p"] < 0.05) else ""
                print(f"{mlabel} & {METHOD_NAMES[method]} & "
                      f"${s['fcm']:.3f}$ & ${s['apm']:.3f}$ & "
                      f"${s['shift']:+.3f}${star} & ${s['fcs']:.2f}/{s['aps']:.2f}$ \\\\")
        print("\\midrule")


if __name__ == "__main__":
    main()
