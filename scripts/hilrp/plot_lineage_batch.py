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
        v = 1.0 # prevent warning
    ax.imshow(R, cmap="bwr", vmin=-v, vmax=v, interpolation="nearest")
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
model_name = sys.argv[2]
method = sys.argv[3]
out_path = sys.argv[4]

device = "cuda" if torch.cuda.is_available() else "cpu"
model = timm.create_model(model_name, pretrained=True).eval().to(device)

config = resolve_data_config({}, model=model)
transform = create_transform(**config)
pil_img = Image.open(img_path).convert('RGB')
x = transform(pil_img).unsqueeze(0).to(device)

if method == "hilrp":
    if model_name.startswith("pvt"):
        from xai_bench.methods.hilrp.pvt_lxt import attribute_pvt
        res = attribute_pvt(model, x)
    elif model_name.startswith("swin"):
        from xai_bench.methods.hilrp.swin_lxt import attribute_swin
        res = attribute_swin(model, x)
    elif model_name.startswith("mobilevit"):
        from xai_bench.methods.hilrp.mobilevit_lxt import attribute_mobilevit
        res = attribute_mobilevit(model, x)
elif method == "naive":
    from scripts.lineage_naive_legs import naive_patch_once, attribute_naive
    naive_patch_once()
    
    if model_name.startswith("swin"):
        from timm.models.swin_transformer import WindowAttention
        def vanilla_window_attention_forward(self, x, mask=None):
            B_, N, C = x.shape
            qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn + self._get_rel_pos_bias()
            if mask is not None:
                num_win = mask.shape[0]
                attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
            attn = self.attn_drop(attn)
            x = attn @ v
            x = x.transpose(1, 2).reshape(B_, N, -1)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x
        WindowAttention.forward = vanilla_window_attention_forward

    pm, sums = attribute_naive(model, x)
    res = {'pixel_map': pm}

torch.save(res, out_path)
"""

def main():
    with open("scripts/worker.py", "w") as f:
        f.write(worker_script)
    
    image_paths = glob.glob("images/*.*")
    if not image_paths:
        image_paths = glob.glob("Images/*.*")
    print(f"Found {len(image_paths)} images.")
    
    models = ["mobilevitv2_100"]
    
    out_dir = os.path.join("results", "custom_visualizations", "batch")
    os.makedirs(out_dir, exist_ok=True)
    
    python_exe = sys.executable
    
    for img_path in image_paths:
        img_name = os.path.basename(img_path).split('.')[0]
        
        try:
            pil_img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Skipping {img_path}: {e}")
            continue
            
        for model_name in models:
            print(f"Processing {img_name} with {model_name}...")
            
            dummy_model = timm.create_model(model_name, pretrained=False)
            config = resolve_data_config({}, model=dummy_model)
            transform = create_transform(**config)
            x = transform(pil_img).unsqueeze(0)
            mean = np.array(config['mean'])
            std = np.array(config['std'])
            img_denorm = denorm(x[0], mean, std)
            
            hilrp_out = f"temp_hilrp_{img_name}_{model_name}.pt"
            naive_out = f"temp_naive_{img_name}_{model_name}.pt"
            
            subprocess.run([python_exe, "scripts/worker.py", img_path, model_name, "hilrp", hilrp_out])
            subprocess.run([python_exe, "scripts/worker.py", img_path, model_name, "naive", naive_out])
            
            if not os.path.exists(hilrp_out) or not os.path.exists(naive_out):
                print(f"Failed to process {img_name} with {model_name}. Outputs not found.")
                continue
                
            res_hilrp = torch.load(hilrp_out, weights_only=False)
            res_naive = torch.load(naive_out, weights_only=False)
            
            os.remove(hilrp_out)
            os.remove(naive_out)
            
            fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
            
            axes[0].imshow(img_denorm)
            axes[0].axis("off")
            axes[0].set_title(f"Input Image\n{model_name}", fontsize=10)
            
            show_relevance(axes[1], res_naive["pixel_map"], "AttnLRP (Naive)\n(No hierarchical rules)")
            show_relevance(axes[2], res_hilrp["pixel_map"], "HiLRP\n(Conservation Preserved)")
            
            fig.tight_layout()
            save_path = os.path.join(out_dir, f"{img_name}_{model_name}_comparison.png")
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
            print(f"Saved {save_path}")
            
    if os.path.exists("scripts/worker.py"):
        os.remove("scripts/worker.py")

if __name__ == "__main__":
    main()
