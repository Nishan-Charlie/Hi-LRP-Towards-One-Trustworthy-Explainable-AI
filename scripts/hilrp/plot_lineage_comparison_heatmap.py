import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from xai_bench.methods.hilrp.vit_lxt import attribute_vit, ensure_patched as vit_patch

def denorm(img_t, mean, std):
    img = img_t.permute(1, 2, 0).numpy() * std + mean
    return np.clip(img, 0, 1)

def show_relevance(ax, R, title):
    R = R.numpy() if torch.is_tensor(R) else R
    v = np.abs(R).max() + 1e-12
    ax.imshow(R, cmap="bwr", vmin=-v, vmax=v, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.axis("off")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading vit_base_patch16_224...")
    vit_patch()
    model = timm.create_model("vit_base_patch16_224", pretrained=True).eval().to(device)
    
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    mean = np.array(config['mean'])
    std = np.array(config['std'])
    
    out_dir = os.path.join("results", "custom_visualizations")
    os.makedirs(out_dir, exist_ok=True)
    
    # Use the first image from the validation cache
    data_path = os.path.join("data", "ImageNetS", "cache_validation_100.pt")
    if os.path.exists(data_path):
        data = torch.load(data_path, map_location="cpu", weights_only=False)
        x = data[0]["image"].unsqueeze(0).to(device)
        img_name = "imagenet_sample_0"
    else:
        # Fallback
        import glob
        image_paths = glob.glob("images/*.*")
        if not image_paths:
            print("No images found to process.")
            return
        img_path = image_paths[0]
        pil_img = Image.open(img_path).convert('RGB')
        x = transform(pil_img).unsqueeze(0).to(device)
        img_name = os.path.basename(img_path)
    
    print("Running AttnLRP...")
    res_attnlrp = attribute_vit(model, x, attn_mode="attnlrp", gamma=0.25)
    
    print("Running HiLRP...")
    res_hilrp = attribute_vit(model, x, attn_mode="cp", gamma=0.25)
    
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    
    img_denorm = denorm(x[0].cpu(), mean, std)
    axes[0].imshow(img_denorm)
    axes[0].axis("off")
    axes[0].set_title(f"Input Image\nPred: {res_hilrp['target']}", fontsize=10)
    
    show_relevance(axes[1], res_attnlrp["pixel_map"], "AttnLRP\n(Softmax Taylor + Uniform Bilinear)")
    show_relevance(axes[2], res_hilrp["pixel_map"], "HiLRP\n(Conservation Preserved)")
    
    fig.tight_layout()
    save_path = os.path.join(out_dir, f"{img_name}_attnlrp_vs_hilrp.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")

if __name__ == '__main__':
    main()
