"""HiLRP in one file: explain one image on any supported backbone.

    python examples/quickstart.py
    python examples/quickstart.py --model pvt_v2_b2 --image images/tiger.jpg
    python examples/quickstart.py --model swin_base_patch4_window7_224 --target 282

Writes a figure with the input, the pixel-level relevance map, and (on Swin) the
map at every resolution stage, and prints the conservation trace. This is the
script behind the README tutorial.
"""
import argparse

import matplotlib.pyplot as plt
import timm
import torch
from PIL import Image

import hilrp


def load(model_name, image_path):
    """A pretrained timm model and one image, preprocessed the way it expects."""
    model = timm.create_model(model_name, pretrained=True).eval()
    cfg = timm.data.resolve_data_config({}, model=model)
    x = timm.data.create_transform(**cfg)(Image.open(image_path).convert("RGB")).unsqueeze(0)
    return model, x, cfg


def to_image(x, cfg):
    """Undo the model's normalization for display: [1, 3, H, W] -> [H, W, 3] in [0, 1]."""
    mean, std = torch.tensor(cfg["mean"]).view(3, 1, 1), torch.tensor(cfg["std"]).view(3, 1, 1)
    return (x[0].cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="swin_tiny_patch4_window7_224")
    parser.add_argument("--image", default="images/cat.jpg")
    parser.add_argument("--target", type=int, default=None, help="class index (default: predicted)")
    parser.add_argument("--out", default="hilrp_quickstart.png")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, x, cfg = load(args.model, args.image)
    model, x = model.to(device), x.to(device)

    res = hilrp.attribute(model, x, target=args.target)

    print(f"{args.model}: class {res['target']}, logit {res['logit']:.2f}")
    print("conservation trace, sum(R) / logit:")
    for capture, ratio in res["stage_sums"]:
        print(f"  {capture:14s} {ratio:+.2f}")
    print(f"  {'pixels':14s} {res['pixel_map'].sum().item() / res['logit']:+.2f}")

    stage_maps = res.get("stage_maps", [])               # Swin only
    fig, axes = plt.subplots(1, 2 + len(stage_maps), figsize=(2.6 * (2 + len(stage_maps)), 2.9))
    axes[0].imshow(to_image(x, cfg))
    axes[0].set_title("input", fontsize=8)
    axes[0].axis("off")
    hilrp.show_relevance(axes[1], res["pixel_map"], title="pixels")
    for ax, (capture, stage_map) in zip(axes[2:], stage_maps):
        hilrp.show_relevance(ax, stage_map, title=capture)
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
