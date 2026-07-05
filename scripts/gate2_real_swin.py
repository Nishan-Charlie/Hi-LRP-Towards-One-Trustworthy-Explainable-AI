"""Gate-2-real for HiLRP on pretrained Swin-T (timm) over cached ImageNet-S.

Produces, in results/hilrp_gate2real/:
  conservation.csv    per-image conservation trace (sum(R)/logit at each stage)
  pointing_game.txt   pointing-game hit rate of the pixel map vs GT bboxes
  maps_grid.png       image + bbox | pixel map | token map (per image, 6 images)
  class_sensitivity.png  same image attributed to top-1 vs 2nd class + correlation
  stage_localization.png the capability figure: relevance at 56/28/14/7 per image

Conservation here is diagnostic reporting; the sharp guarantees live in
xai_bench/methods/hilrp/tests/. What THIS script establishes is the part a
conservation test cannot: maps land on objects and change with the target class.

Run:  <mri-diffuser python> scripts/gate2_real_swin.py
"""
import os
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import timm
from xai_bench.methods.hilrp.swin_lxt import attribute_swin, ensure_patched

OUT = os.path.join("results", "hilrp_gate2real")
os.makedirs(OUT, exist_ok=True)

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
GAMMA = 0.25
N_CONSERVATION = 20
N_GRID = 6


def denorm(img_t):
    img = img_t.permute(1, 2, 0).numpy() * STD + MEAN
    return np.clip(img, 0, 1)


def show_relevance(ax, R, title, res_note=None):
    # robust percentile normalization: relevance is heavy-tailed, so symmetric-max
    # renders spiky maps (e.g. MobileViT) near-blank. See hilrp.viz.
    from xai_bench.methods.hilrp.viz import normalize_for_display
    m, v = normalize_for_display(R, percentile=99.0)
    ax.imshow(m, cmap="bwr", vmin=-v, vmax=v, interpolation="nearest")
    ax.set_title(title + (f"\n{res_note}" if res_note else ""), fontsize=8)
    ax.axis("off")


def draw_bboxes(ax, bboxes, h, w):
    for bb in bboxes:
        x0, y0, x1, y1 = bb[0] * w, bb[1] * h, bb[2] * w, bb[3] * h
        ax.add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                        fill=False, edgecolor="lime", linewidth=2))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True).eval().to(device)
    ensure_patched()
    data = torch.load(os.path.join("data", "ImageNetS", "cache_validation_100.pt"),
                      map_location="cpu", weights_only=False)

    # ---------------- conservation trace + pointing game over N images ----------
    rows, hits, total = [], 0, 0
    results_cache = {}
    for idx in range(N_CONSERVATION):
        item = data[idx]
        x = item["image"].unsqueeze(0).to(device)
        res = attribute_swin(model, x, gamma=GAMMA)
        results_cache[idx] = res
        rows.append([idx, item["label"], res["target"], res["logit"]] +
                    [s for _, s in res["stage_sums"]] +
                    [res["pixel_map"].sum().item() / res["logit"]])

        # pointing game: argmax of the pixel map inside any GT bbox?
        pm = res["pixel_map"].numpy()
        py, px = np.unravel_index(np.argmax(pm), pm.shape)
        h, w = pm.shape
        hit = any(bb[0] * w <= px <= bb[2] * w and bb[1] * h <= py <= bb[3] * h
                  for bb in item["metadata"]["bboxes"])
        hits += hit
        total += 1

    names = [n for n, _ in results_cache[0]["stage_sums"]]
    header = "idx,label,pred,logit," + ",".join(names) + ",pixels"
    with open(os.path.join(OUT, "conservation.csv"), "w") as f:
        f.write(header + "\n")
        for r in rows:
            f.write(",".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r) + "\n")
    arr = np.array([r[4:] for r in rows])
    print("conservation (sum(R)/logit), mean +- std over %d images:" % total)
    for j, n in enumerate(names + ["pixels"]):
        print(f"  {n:16s} {arr[:, j].mean():+.3f} +- {arr[:, j].std():.3f}")

    pg = hits / total
    with open(os.path.join(OUT, "pointing_game.txt"), "w") as f:
        f.write(f"pointing game (pixel map argmax in GT bbox): {hits}/{total} = {pg:.3f}\n")
    print(f"pointing game: {hits}/{total} = {pg:.2f}")

    # -------------------------------- maps grid --------------------------------
    fig, axes = plt.subplots(N_GRID, 3, figsize=(7.5, 2.5 * N_GRID))
    for i in range(N_GRID):
        item, res = data[i], results_cache[i]
        img = denorm(item["image"])
        axes[i, 0].imshow(img); axes[i, 0].axis("off")
        draw_bboxes(axes[i, 0], item["metadata"]["bboxes"], *img.shape[:2])
        axes[i, 0].set_title(f"label {item['label']} / pred {res['target']}", fontsize=8)
        show_relevance(axes[i, 1], res["pixel_map"], "pixel relevance (224x224)")
        show_relevance(axes[i, 2], res["input_map"], "token relevance (56x56)")
    fig.suptitle(f"HiLRP on Swin-T (gamma={GAMMA}, CP attention)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "maps_grid.png"), dpi=150)
    plt.close(fig)

    # ----------------------------- class sensitivity ---------------------------
    # Compare the predicted class vs a semantically distant class. NOTE: top-1 vs
    # top-2 on a single-object image is the WRONG test (both classes are explained
    # by the same object, so high correlation is expected). Signed correlation can
    # go strongly negative (evidence-against flips sign); positive-part correlation
    # asks "does evidence FOR each class sit in different places".
    rng = np.random.RandomState(0)
    signed_cs, pos_cs = [], []
    demo = None
    for idx in range(5):
        x = data[idx]["image"].unsqueeze(0).to(device)
        res_a = results_cache.get(idx) or attribute_swin(model, x, gamma=GAMMA)
        far = int(rng.choice([c for c in range(1000) if abs(c - res_a["target"]) > 200]))
        res_d = attribute_swin(model, x, target=far, gamma=GAMMA)
        a, d = res_a["pixel_map"].numpy(), res_d["pixel_map"].numpy()
        signed_cs.append(float(np.corrcoef(a.flatten(), d.flatten())[0, 1]))
        pos_cs.append(float(np.corrcoef(np.maximum(a, 0).flatten(),
                                        np.maximum(d, 0).flatten())[0, 1]))
        if demo is None:
            demo = (idx, res_a, res_d, far)

    idx, res_a, res_d, far = demo
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    axes[0].imshow(denorm(data[idx]["image"])); axes[0].axis("off"); axes[0].set_title("input", fontsize=9)
    show_relevance(axes[1], res_a["pixel_map"], f"target = pred ({res_a['target']})")
    show_relevance(axes[2], res_d["pixel_map"], f"target = distant ({far})")
    fig.suptitle(f"class sensitivity over 5 imgs: signed corr {np.mean(signed_cs):+.2f}, "
                 f"positive-part corr {np.mean(pos_cs):+.2f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "class_sensitivity.png"), dpi=150)
    plt.close(fig)
    print(f"class sensitivity (pred vs distant, 5 imgs): signed {np.mean(signed_cs):+.3f}, "
          f"pos-part {np.mean(pos_cs):+.3f}  (lower = more class-specific)")

    # -------------------------- stage localization figure ----------------------
    n_show = 4
    fig, axes = plt.subplots(n_show, 6, figsize=(13, 2.3 * n_show))
    for i in range(n_show):
        item, res = data[i], results_cache[i]
        img = denorm(item["image"])
        axes[i, 0].imshow(img); axes[i, 0].axis("off")
        axes[i, 0].set_title("input", fontsize=8)
        show_relevance(axes[i, 1], res["pixel_map"], "pixels 224")
        for j, (name, m) in enumerate(res["stage_maps"][:4]):
            show_relevance(axes[i, 2 + j], m, name)
    fig.suptitle("Stage-localized relevance (conserved flow at every resolution) "
                 "-- unavailable to output-perturbation methods", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "stage_localization.png"), dpi=150)
    plt.close(fig)

    print(f"\nfigures written to {OUT}")


if __name__ == "__main__":
    main()
