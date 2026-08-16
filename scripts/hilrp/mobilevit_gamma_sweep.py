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

from xai_bench.methods.hilrp.mobilevit_lxt import attribute_mobilevit

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
    print("Loading mobilevitv2_100...")
    model = timm.create_model("mobilevitv2_100", pretrained=True).eval().to(device)
    
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    mean = np.array(config['mean'])
    std = np.array(config['std'])
    
    out_dir = os.path.join("results", "custom_visualizations")
    os.makedirs(out_dir, exist_ok=True)
    
    img_path = "images/cat.jpg"
    pil_img = Image.open(img_path).convert('RGB')
    x = transform(pil_img).unsqueeze(0).to(device)
    img_name = os.path.basename(img_path).split('.')[0]
    
    # Let's do a gamma sweep for conv_gamma
    gammas = [0.0, 0.1, 0.25, 0.5, 1.0]
    
    fig, axes = plt.subplots(1, len(gammas) + 1, figsize=(3 * (len(gammas) + 1), 3.5))
    
    img_denorm = denorm(x[0].cpu(), mean, std)
    axes[0].imshow(img_denorm)
    axes[0].axis("off")
    axes[0].set_title(f"Input Image", fontsize=10)
    
    for i, conv_gamma in enumerate(gammas):
        print(f"Running HiLRP with conv_gamma={conv_gamma}...")
        res = attribute_mobilevit(model, x, gamma=0.25, conv_gamma=conv_gamma)
        show_relevance(axes[i+1], res["pixel_map"], f"conv_gamma={conv_gamma}\n(gamma=0.25)")
    
    fig.tight_layout()
    save_path = os.path.join(out_dir, f"{img_name}_mobilevit_gamma_sweep.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")

if __name__ == '__main__':
    main()
