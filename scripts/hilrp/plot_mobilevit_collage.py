import os
import glob
import subprocess
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

def denorm(img_t, mean, std):
    img = img_t.permute(1, 2, 0).numpy() * std + mean
    return np.clip(img, 0, 1)

def show_relevance(ax, R, title):
    R = R.numpy() if torch.is_tensor(R) else R
    v = np.abs(R).max() + 1e-12
    if v == 0:
        v = 1.0
    ax.imshow(R, cmap="bwr", vmin=-v, vmax=v, interpolation="nearest")
    if title:
        ax.set_title(title, fontsize=10)
    ax.axis("off")

worker_script = """import sys
import torch
from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

sys.path.insert(0, ".")

img_path = sys.argv[1]
method = sys.argv[2]
out_path = sys.argv[3]

device = "cuda" if torch.cuda.is_available() else "cpu"
model = timm.create_model("mobilevitv2_100", pretrained=True).eval().to(device)

config = resolve_data_config({}, model=model)
transform = create_transform(**config)
pil_img = Image.open(img_path).convert('RGB')
x = transform(pil_img).unsqueeze(0).to(device)

if method == "hilrp":
    from xai_bench.methods.hilrp.mobilevit_lxt import attribute_mobilevit
    # USE conv_gamma=0.1 to stabilize MobileViT!
    res = attribute_mobilevit(model, x, gamma=0.25, conv_gamma=0.1)
elif method == "naive":
    from scripts.lineage_naive_legs import naive_patch_once, attribute_naive
    naive_patch_once()
    pm, sums = attribute_naive(model, x)
    res = {'pixel_map': pm}

torch.save(res, out_path)
"""

def main():
    with open("scripts/worker_mobilevit.py", "w") as f:
        f.write(worker_script)
    
    image_paths = sorted(glob.glob("images/*.*") + glob.glob("Images/*.*"))[:15] # limit to 15 for grid
    print(f"Found {len(image_paths)} images.")
    
    out_dir = os.path.join("results", "custom_visualizations")
    os.makedirs(out_dir, exist_ok=True)
    
    python_exe = sys.executable
    
    # Grid size for collage
    n_images = len(image_paths)
    rows = int(np.ceil(n_images / 3))
    cols = 3
    
    # Each image takes 3 subplots (Input, Naive, HiLRP)
    fig, axes = plt.subplots(rows, cols * 3, figsize=(4 * cols * 3, 4 * rows))
    
    for idx, img_path in enumerate(image_paths):
        row = idx // cols
        col = idx % cols
        
        img_name = os.path.basename(img_path).split('.')[0]
        try:
            pil_img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Skipping {img_path}: {e}")
            continue
            
        print(f"Processing {img_name}...")
        
        dummy_model = timm.create_model("mobilevitv2_100", pretrained=False)
        config = resolve_data_config({}, model=dummy_model)
        transform = create_transform(**config)
        x = transform(pil_img).unsqueeze(0)
        mean = np.array(config['mean'])
        std = np.array(config['std'])
        img_denorm = denorm(x[0], mean, std)
        
        hilrp_out = f"temp_hilrp_{img_name}.pt"
        naive_out = f"temp_naive_{img_name}.pt"
        
        subprocess.run([python_exe, "scripts/worker_mobilevit.py", img_path, "hilrp", hilrp_out])
        subprocess.run([python_exe, "scripts/worker_mobilevit.py", img_path, "naive", naive_out])
        
        if not os.path.exists(hilrp_out) or not os.path.exists(naive_out):
            print(f"Failed to process {img_name}.")
            continue
            
        res_hilrp = torch.load(hilrp_out, weights_only=False)
        res_naive = torch.load(naive_out, weights_only=False)
        
        os.remove(hilrp_out)
        os.remove(naive_out)
        
        # Subplot indices
        base_ax = col * 3
        
        ax_in = axes[row, base_ax]
        ax_naive = axes[row, base_ax + 1]
        ax_hilrp = axes[row, base_ax + 2]
        
        ax_in.imshow(img_denorm)
        ax_in.axis("off")
        title_in = "Input" if row == 0 else ""
        if title_in:
            ax_in.set_title(title_in, fontsize=18)
        
        title_naive = "AttnLRP (Naive)" if row == 0 else ""
        show_relevance(ax_naive, res_naive["pixel_map"], title_naive)
        if title_naive:
            ax_naive.set_title(title_naive, fontsize=18)
            
        title_hilrp = "HiLRP (conv_gamma=0.1)" if row == 0 else ""
        show_relevance(ax_hilrp, res_hilrp["pixel_map"], title_hilrp)
        if title_hilrp:
            ax_hilrp.set_title(title_hilrp, fontsize=18)

    # Hide any unused subplots
    for idx in range(n_images, rows * cols):
        row = idx // cols
        col = idx % cols
        base_ax = col * 3
        for offset in range(3):
            axes[row, base_ax + offset].axis("off")
            
    fig.tight_layout()
    save_path = os.path.join(out_dir, "mobilevit_collage.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")
            
    if os.path.exists("scripts/worker_mobilevit.py"):
        os.remove("scripts/worker_mobilevit.py")

if __name__ == "__main__":
    main()
