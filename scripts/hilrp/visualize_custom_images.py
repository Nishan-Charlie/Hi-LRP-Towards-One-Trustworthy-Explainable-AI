import os
import glob
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import HiLRP for PVT as an example (it had the best score!)
from xai_bench.methods.hilrp.pvt_lxt import attribute_pvt, ensure_patched

def denorm(img_t, mean, std):
    img = img_t.permute(1, 2, 0).numpy() * std + mean
    return np.clip(img, 0, 1)

def show_relevance(ax, R, title):
    R = R.numpy() if torch.is_tensor(R) else R
    v = np.abs(R).max() + 1e-12
    ax.imshow(R, cmap="bwr", vmin=-v, vmax=v, interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.axis("off")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading model pvt_v2_b2...")
    model_name = "pvt_v2_b2"
    model = timm.create_model(model_name, pretrained=True).eval().to(device)
    ensure_patched()
    
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    
    mean = np.array(config['mean'])
    std = np.array(config['std'])
    
    out_dir = os.path.join("results", "custom_visualizations")
    os.makedirs(out_dir, exist_ok=True)
    
    image_paths = glob.glob("images/*.*")
    print(f"Found {len(image_paths)} images.")
    
    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        try:
            pil_img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Could not read {img_path}: {e}")
            continue
            
        x = transform(pil_img).unsqueeze(0).to(device)
        
        # Run HiLRP attribution
        res = attribute_pvt(model, x, gamma=0.25)
        
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        
        img_denorm = denorm(x[0].cpu(), mean, std)
        axes[0].imshow(img_denorm)
        axes[0].axis("off")
        axes[0].set_title(f"Input: {img_name}\nPred: {res['target']} (logit: {res['logit']:.2f})", fontsize=8)
        
        show_relevance(axes[1], res["pixel_map"], "HiLRP Pixel Relevance")
        
        fig.tight_layout()
        save_path = os.path.join(out_dir, f"{img_name}_hilrp.png")
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Saved {save_path}")

if __name__ == '__main__':
    main()
