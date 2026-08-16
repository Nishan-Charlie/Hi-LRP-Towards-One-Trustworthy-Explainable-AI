"""SSL-scalar attribution demo (the label-free branch, RQ2 seed).

Same architecture (ViT-S/16), same label-free scalar (cosine similarity of the
CLS embedding to the embedding of the horizontally flipped view), two
pretrainings:

  * DINO (self-supervised)   timm 'vit_small_patch16_224.dino'
  * AugReg (supervised)      timm 'vit_small_patch16_224.augreg_in21k_ft_in1k'

Question the maps answer: "what evidence in the image supports its own
view-invariance". No labels, no classifier head anywhere. If DINO's known
object-centric behavior is real, its similarity evidence should localize the
object (measured with the pointing game against GT boxes, even though the
scalar never saw a label); the supervised backbone provides the RQ2 contrast
under an identical scalar.

Outputs in results/ssl_scalar/: maps figure + pointing/conservation summary.

Run:  <mri-diffuser python> scripts/hilrp/ssl_scalar_demo.py
"""
import os
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import timm

OUT = os.path.join("results", "ssl_scalar")
os.makedirs(OUT, exist_ok=True)

MODELS = {
    "dino": "vit_small_patch16_224.dino",
    "supervised": "vit_small_patch16_224.augreg_in21k_ft_in1k",
}
N_IMAGES = 12
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def denorm(img_t):
    return np.clip(img_t.permute(1, 2, 0).numpy() * STD + MEAN, 0, 1)


def pointing(pm, bboxes):
    h, w = pm.shape
    py, px = np.unravel_index(np.argmax(pm), pm.shape)
    return any(bb[0] * w <= px <= bb[2] * w and bb[1] * h <= py <= bb[3] * h
               for bb in bboxes)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = torch.load(os.path.join("data", "ImageNetS", "cache_validation_100.pt"),
                      map_location="cpu", weights_only=False)

    from xai_bench.methods.hilrp.vit_lxt import attribute_ssl_similarity, ensure_patched
    from xai_bench.methods.hilrp.pvt_lxt import audit_norm_patches
    ensure_patched()

    results = {}
    for tag, name in MODELS.items():
        model = timm.create_model(name, pretrained=True).eval().to(device)
        bad = audit_norm_patches(model, verbose=True)
        assert not bad, f"unpatched norms in {name}"

        hits, sims, cons, maps = 0, [], [], []
        for idx in range(N_IMAGES):
            x = data[idx]["image"].unsqueeze(0).to(device)
            x_ref = torch.flip(x, dims=(3,))            # horizontally flipped view
            res = attribute_ssl_similarity(model, x, x_ref)
            hits += pointing(res["pixel_map"].numpy(), data[idx]["metadata"]["bboxes"])
            sims.append(res["similarity"])
            cons.append(res["stage_sums"][0][1])
            maps.append(res["pixel_map"])
        results[tag] = dict(hits=hits, sims=sims, cons=cons, maps=maps)
        print(f"[{tag:10s}] view-similarity {np.mean(sims):.3f} +- {np.std(sims):.3f}   "
              f"label-free pointing {hits}/{N_IMAGES} = {hits / N_IMAGES:.2f}   "
              f"pixelR/scalar {np.mean(cons):+.2f} +- {np.std(cons):.2f}")

    # figure: image | dino map | supervised map
    n_show = 6
    fig, axes = plt.subplots(n_show, 3, figsize=(8, 2.4 * n_show))
    for i in range(n_show):
        img = denorm(data[i]["image"])
        axes[i, 0].imshow(img); axes[i, 0].axis("off")
        for bb in data[i]["metadata"]["bboxes"]:
            h, w = img.shape[:2]
            axes[i, 0].add_patch(mpatches.Rectangle(
                (bb[0] * w, bb[1] * h), (bb[2] - bb[0]) * w, (bb[3] - bb[1]) * h,
                fill=False, edgecolor="lime", linewidth=2))
        axes[i, 0].set_title("input", fontsize=8)
        for j, tag in enumerate(["dino", "supervised"]):
            m = results[tag]["maps"][i].numpy()
            v = np.abs(m).max() + 1e-12
            axes[i, 1 + j].imshow(m, cmap="bwr", vmin=-v, vmax=v)
            axes[i, 1 + j].set_title(f"{tag}: view-similarity evidence", fontsize=8)
            axes[i, 1 + j].axis("off")
    fig.suptitle("Label-free HiLRP: what supports the image's own view-invariance\n"
                 "(same ViT-S/16 architecture, same scalar, different pretraining)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ssl_similarity_maps.png"), dpi=150)
    print(f"figure written to {OUT}")


if __name__ == "__main__":
    main()
