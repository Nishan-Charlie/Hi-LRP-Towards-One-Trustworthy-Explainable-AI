#!/usr/bin/env python
"""
run_imagenets_xai.py
Downloads a few sample images from the ImageNet-S dataset via Hugging Face `datasets`,
runs inference using a Vision Transformer (ViT-S/16 by default), and extracts 
explainability heatmaps (Grad-CAM, IG, Attention Rollout) and pixel graph clusters.
"""

import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
from PIL import Image
from skimage.transform import resize
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torchvision.transforms as T
import timm

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from captum.attr import IntegratedGradients
import community as community_louvain

# Import datasets
try:
    from datasets import load_dataset
except ImportError:
    print("Please install datasets: pip install datasets")
    sys.exit(1)

warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ── Preprocessing ─────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_image_tensor(img_pil, input_size=224):
    img = img_pil.convert("RGB").resize((input_size, input_size))
    tf  = T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return img, tf(img).unsqueeze(0).to(DEVICE)

# ── Reshape transforms for pytorch-grad-cam ──────────────────────────────────
def make_vit_reshape(h=14, w=14, n_skip=1):
    def reshape(tensor):
        r = tensor[:, n_skip:, :]
        B, N, C = r.shape
        return r.reshape(B, h, w, C).permute(0, 3, 1, 2)
    return reshape

# ── Grad-CAM & Grad-CAM++ ─────────────────────────────────────────────────────
def compute_gradcam(model, inp, cls_idx, layers, reshape):
    cam = GradCAM(model=model, target_layers=layers, reshape_transform=reshape)
    return cam(input_tensor=inp, targets=[ClassifierOutputTarget(cls_idx)])[0]

def compute_gradcam_pp(model, inp, cls_idx, layers, reshape):
    cam = GradCAMPlusPlus(model=model, target_layers=layers, reshape_transform=reshape)
    return cam(input_tensor=inp, targets=[ClassifierOutputTarget(cls_idx)])[0]

# ── Integrated Gradients (captum) ─────────────────────────────────────────────
def compute_ig(model, inp, cls_idx, n_steps=50):
    model.eval()
    x    = inp.clone().detach().requires_grad_(True)
    ig   = IntegratedGradients(lambda t: model(t)[:, cls_idx].unsqueeze(1))
    base = torch.zeros_like(x)
    attr = ig.attribute(x, base, n_steps=n_steps, internal_batch_size=4)
    hm   = attr.squeeze(0).abs().sum(0).detach().cpu().numpy()
    return hm / hm.max() if hm.max() > 0 else hm

# ── Attention Rollout (Abnar & Zuidema 2020) ──────────────────────────────────
def compute_attention_rollout(model, inp, patch_h=14, patch_w=14, has_cls=True, discard_ratio=0.9):
    all_attn = []

    def hook_fn(module, input, output):
        try:
            x = input[0]
            B, N, C = x.shape
            H = module.num_heads
            D = C // H
            if hasattr(module, "qkv"):
                qkv = module.qkv(x).reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
                q, k, _ = qkv.unbind(0)
            elif hasattr(module, "q_proj") and hasattr(module, "k_proj"):
                q = module.q_proj(x).reshape(B, N, H, D).permute(0, 2, 1, 3)
                k = module.k_proj(x).reshape(B, N, H, D).permute(0, 2, 1, 3)
            else:
                return
            attn = (q @ k.transpose(-2, -1) * (D ** -0.5)).softmax(dim=-1)
            all_attn.append(attn.mean(dim=1).detach().cpu())
        except Exception:
            pass

    hooks = []
    for blk in getattr(model, "blocks", []):
        if hasattr(blk, "attn"):
            hooks.append(blk.attn.register_forward_hook(hook_fn))

    with torch.no_grad():
        model(inp)
    for h in hooks:
        h.remove()

    if not all_attn:
        return None

    N       = all_attn[0].shape[-1]
    I       = torch.eye(N)
    rollout = I.clone()

    for attn in all_attn:
        a   = attn[0].clone()
        thr = a.flatten().kthvalue(int(a.numel() * discard_ratio)).values
        a[a < thr] = 0.0
        a   = a + I
        a   = a / a.sum(-1, keepdim=True)
        rollout = a @ rollout

    importance = rollout[0, 1:] if has_cls else rollout.mean(0)
    importance = importance.reshape(patch_h, patch_w).numpy()
    if importance.max() > 0:
        importance /= importance.max()
    return resize(importance, (224, 224), mode="reflect", anti_aliasing=True)

# ── Heatmap utilities ─────────────────────────────────────────────────────────
def norm_resize(hm, size=224):
    hm = resize(np.squeeze(hm), (size, size), mode="reflect", anti_aliasing=True)
    return hm / hm.max() if hm.max() > 0 else hm

def combine_hm(gc, ig):
    c = gc * ig
    c = np.clip(c, 0, 1)
    return c / c.max() if c.max() > 0 else c

# ── Graph construction & clustering ───────────────────────────────────────────
def create_graph(heatmap, threshold=0.3):
    G = nx.Graph()
    H, W = heatmap.shape
    for i in range(H):
        for j in range(W):
            if heatmap[i, j] > threshold:
                G.add_node((i, j), value=float(heatmap[i, j]))
                for di, dj in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < H and 0 <= nj < W and heatmap[ni, nj] > threshold:
                        G.add_edge((i, j), (ni, nj))
    return G

def cluster_louvain(G):
    return community_louvain.best_partition(G)

def plot_clustered_graph(G, partition, title, save_path=None):
    if len(G) == 0:
        return
    pos    = {(x, y): (y, -x) for x, y in G.nodes()}
    clusters = {}
    for n, cid in partition.items():
        clusters.setdefault(cid, []).append(n)
    reps   = {}
    total  = len(G)
    for cid, nodes in clusters.items():
        reps[nodes[0]] = len(nodes) / total * 100
    labels  = {n: f"n{i+1}" for i, n in enumerate(reps)}
    cmap_v  = cm.get_cmap("viridis", max(len(reps), 1))
    rlist   = list(reps.keys())

    plt.figure(figsize=(6, 6))
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    for idx, (n, pct) in enumerate(reps.items()):
        nx.draw_networkx_nodes(G, pos, nodelist=[n], node_size=700, node_color=[cmap_v(idx)], alpha=0.8)
        px, py = pos[n]
        plt.text(px - 1.5, py + 0.2,  labels[n], fontsize=11, ha="center", va="center", color="black")
        plt.text(px + 1.0, py + 0.05, f"{pct:.1f}%", fontsize=11, ha="left", va="center")
    for i in range(len(rlist)):
        for j in range(i + 1, len(rlist)):
            plt.plot([pos[rlist[i]][0], pos[rlist[j]][0]],
                     [pos[rlist[i]][1], pos[rlist[j]][1]], "r--", lw=2, alpha=0.7)
    plt.title(title)
    plt.axis("off")
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# ── 6-panel heatmap visualisation ────────────────────────────────────────────
def plot_heatmaps(img_pil, gc, gcc, ig, ar, comb, title="", save_path=None):
    img_np = np.array(img_pil.resize((224, 224))) / 255.0
    R      = lambda x: norm_resize(x) if x is not None else None

    fig, axes = plt.subplots(1, 6, figsize=(26, 4))
    fig.suptitle(title, fontsize=11, fontweight="bold", y=1.02)

    def show_panel(ax, base, overlay, label, alpha=0.5):
        if base is not None:
            ax.imshow(base)
        if overlay is not None:
            ax.imshow(overlay, cmap="jet", alpha=alpha)
        ax.set_title(label, fontsize=9)
        ax.axis("off")

    show_panel(axes[0], img_np, None,   "Original Image")
    show_panel(axes[1], img_np, R(gc),  "Grad-CAM")
    show_panel(axes[2], img_np, R(gcc), "Grad-CAM++")
    show_panel(axes[3], None,   R(ig),  "Integrated Gradients")

    if ar is not None:
        show_panel(axes[4], img_np, R(ar), "Attention Rollout")
    else:
        axes[4].text(0.5, 0.5, "Attention Rollout\nN/A", ha="center", va="center", transform=axes[4].transAxes, fontsize=9, color="grey")
        axes[4].set_title("Attention Rollout", fontsize=9)
        axes[4].axis("off")

    show_panel(axes[5], img_np, R(comb), "Combined (GC x IG)", alpha=0.9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    num_samples = 5
    out_dir = "results/ImageNetS"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("data/ImageNetS/samples", exist_ok=True)
    
    print(f"Loading {num_samples} samples from ImageNet-S dataset...")
    # Load ImageNet-S validation split. Note: loading a subset.
    dataset = load_dataset("braceletboy/imagenet-s", split="validation", streaming=True)
    
    samples = []
    for i, item in enumerate(dataset):
        if i >= num_samples:
            break
        samples.append(item)
    
    print("Loading ViT-S/16 model...")
    # We use ViT-S/16 as the requested model
    model_name = "vit_small_patch16_224"
    model = timm.create_model(model_name, pretrained=True)
    model = model.to(DEVICE).eval()
    
    layers = [model.blocks[-1].norm1]
    reshape = make_vit_reshape(h=14, w=14, n_skip=1)
    
    for i, sample in enumerate(samples):
        # image is a PIL image
        img_pil = sample['image']
        if img_pil.mode != "RGB":
            img_pil = img_pil.convert("RGB")
        
        # Save sample locally
        img_path = f"data/ImageNetS/samples/sample_{i}.jpg"
        img_pil.save(img_path)
        
        # Ground truth class if needed. ImageNet-S gives 'label' (int)
        cls_idx = sample['label']
        print(f"\nProcessing Image {i} (Label: {cls_idx})...")
        
        img_pil, inp = get_image_tensor(img_pil, 224)
        
        # Inference
        with torch.no_grad():
            preds = model(inp)
            pred_cls = preds.argmax(dim=-1).item()
        print(f"  Predicted class: {pred_cls}")
        
        # We will use predicted class to get explainability for what the model thinks it is
        target_cls = pred_cls 
        
        # Grad-CAM
        try:
            gc = compute_gradcam(model, inp, target_cls, layers, reshape)
        except Exception as e:
            print(f"  GradCAM error: {e}")
            gc = np.zeros((224, 224))
            
        # Grad-CAM++
        try:
            gcc = compute_gradcam_pp(model, inp, target_cls, layers, reshape)
        except Exception as e:
            print(f"  GradCAM++ error: {e}")
            gcc = np.zeros((224, 224))
            
        # Integrated Gradients
        try:
            ig = compute_ig(model, inp, target_cls)
        except Exception as e:
            print(f"  IG error: {e}")
            ig = np.zeros((224, 224))
            
        # Attention Rollout
        try:
            ar = compute_attention_rollout(model, inp, 14, 14, has_cls=True)
        except Exception as e:
            print(f"  Attn Rollout error: {e}")
            ar = np.zeros((224, 224))
            
        # Combined heatmap
        gc_r = norm_resize(gc)
        ig_r = norm_resize(ig)
        comb = combine_hm(gc_r, ig_r)
        
        # Plot and save heatmaps
        hm_save_path = f"{out_dir}/sample_{i}_heatmaps.png"
        plot_heatmaps(img_pil, gc, gcc, ig, ar, comb, 
                      title=f"ImageNet-S Sample {i} | ViT-S/16 | Pred: {target_cls}",
                      save_path=hm_save_path)
        print(f"  Saved heatmaps to {hm_save_path}")
        
        # Graph clustering
        G = create_graph(comb)
        if len(G) > 0:
            part = cluster_louvain(G)
            graph_save_path = f"{out_dir}/sample_{i}_graph.png"
            plot_clustered_graph(G, part, f"Sample {i} - Louvain Clustering", save_path=graph_save_path)
            print(f"  Saved graph clustering to {graph_save_path}")

if __name__ == "__main__":
    main()
