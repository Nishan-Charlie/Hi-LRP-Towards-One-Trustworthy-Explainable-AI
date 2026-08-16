"""Consolidate the Grad-CAM vs Grad-CAM++ Max-Sensitivity stability study.

Reads results/gradcam_ms_100/results.json (7 non-isotropic backbones, generic
CAM) and results/vit_gradcam_ms.json (ViT-B/16 via the standard ViT target
layer). Emits mean/median/std/max and the fraction of undefined (degenerate)
maps per method, plus LaTeX rows. This is the empirical backing for Section 8.2:
Grad-CAM++ instability is heavy-tailed and confined to global-attention backbones.
"""
import json
import numpy as np
import collections

MN = {"resnet50": "RN-50", "vit_base_patch16_224": "ViT-B/16",
      "swin_base_patch4_window7_224": "Swin-B", "pvt_v2_b2": "PVT-v2",
      "maxvit_small_tf_224": "MaxViT-S", "mobilevitv2_100": "MobileViT",
      "efficientvit_b1": "EffViT-B1", "efficientvit_b2": "EffViT-B2"}
ATTN = {"resnet50": "CNN", "vit_base_patch16_224": "global softmax",
        "swin_base_patch4_window7_224": "windowed", "pvt_v2_b2": "spatial-reduction",
        "maxvit_small_tf_224": "multi-axis (global grid)", "mobilevitv2_100": "conv-hybrid",
        "efficientvit_b1": "linear", "efficientvit_b2": "linear"}
ORDER = ["resnet50", "vit_base_patch16_224", "swin_base_patch4_window7_224",
         "pvt_v2_b2", "maxvit_small_tf_224", "mobilevitv2_100",
         "efficientvit_b1", "efficientvit_b2"]


def stat(vals, total):
    v = np.array([x for x in vals if x is not None and not np.isnan(x)])
    if len(v) == 0:
        return None
    return dict(n=len(v), undef=total - len(v), mean=v.mean(), std=v.std(),
                median=float(np.median(v)), mx=v.max())


def main():
    rows = json.load(open("results/gradcam_ms_100/results.json"))
    ms = collections.defaultdict(lambda: collections.defaultdict(list))
    tot = collections.Counter()
    for x in rows:
        ms[(x["model"], x["method"])]["ms"].append(x.get("max_sensitivity"))
        tot[(x["model"], x["method"])] += 1

    # ViT-B from the proper-target-layer run
    vit = {r["method"]: r for r in json.load(open("results/vit_gradcam_ms.json"))}

    def get(model, method):
        if model == "vit_base_patch16_224":
            r = vit[method]
            return dict(n=r["n_ms"], undef=100 - r["n_ms"], mean=r["ms_mean"],
                        std=r["ms_std"], median=r["ms_median"], mx=r["ms_max"])
        return stat(ms[(model, method)]["ms"], tot[(model, method)])

    print(f"{'Model':10s} {'attention':22s} "
          f"{'GC med(mean)':>16s} {'GC++ med(mean)':>18s} {'GC++ max':>9s} {'GC++ undef':>11s}")
    print("-" * 92)
    latex = []
    for model in ORDER:
        g = get(model, "grad_cam")
        gpp = get(model, "grad_cam_plus_plus")
        gstr = f"{g['median']:.2f} ({g['mean']:.2f})"
        pstr = f"{gpp['median']:.2f} ({gpp['mean']:.2f})"
        print(f"{MN[model]:10s} {ATTN[model]:22s} {gstr:>16s} {pstr:>18s} "
              f"{gpp['mx']:9.1f} {gpp['undef']:9d}/100")
        latex.append((model, g, gpp))

    print("\n=== LaTeX rows: Model & attn & GC med(mean) & GC++ med(mean) & GC++ max & GC++ undef ===")
    for model, g, gpp in latex:
        print(f"{MN[model]} & {ATTN[model]} & "
              f"${g['median']:.2f}\\,({g['mean']:.2f})$ & "
              f"${gpp['median']:.2f}\\,({gpp['mean']:.2f})$ & "
              f"${gpp['mx']:.1f}$ & ${gpp['undef']}/100$ \\\\")


if __name__ == "__main__":
    main()
