#!/usr/bin/env python3
"""
generate_foundation_nb.py
Run: python generate_foundation_nb.py
Outputs: foundation_model_xai.ipynb
"""
import json, uuid

def uid():
    return str(uuid.uuid4())[:8]

def code_cell(src):
    return {"cell_type": "code", "execution_count": None, "id": uid(),
            "metadata": {}, "outputs": [], "source": src.strip()}

def md_cell(src):
    return {"cell_type": "markdown", "id": uid(), "metadata": {}, "source": src.strip()}

cells = []

# ─────────────────────────────────────────────────────────────────────────────
cells.append(md_cell("""
# Foundation Model XAI Benchmark
### Comparing Explainability Across Self-Supervised & Supervised Vision Transformers

**Training Paradigms Covered:**
| # | Paradigm | Models |
|---|---|---|
| 1 | Supervised (baseline) | ViT-B/16 (IN21k → IN1k) |
| 2 | Self-Distillation | DINOv1-B/16, DINOv1-B/8 |
| 3 | Improved Self-Distillation | DINOv2-S/14, DINOv2-B/14, DINOv2-B/14+Reg |
| 4 | Masked Autoencoding (pixel) | MAE ViT-B/16, MAE ViT-L/16 |
| 5 | Masked Token Prediction | BEiT3-B/16 |
| 6 | Language-Image Contrastive | CLIP ViT-B/16 (OpenAI), CLIP ViT-B/16 (LAION-2B), MetaCLIP ViT-B/16 |
| 7 | MIM + Contrastive Hybrid | EVA02-B/14 |

**XAI Methods Applied per Model:**
- Grad-CAM · Grad-CAM++ · Integrated Gradients · Attention Rollout · Combined (GC×IG)
- Pixel Graph Clustering (Louvain / Spectral / Agglomerative)

**Quantitative Metrics:**
- Deletion AUC · Insertion AUC · Faithfulness Correlation · Gini Sparseness

> **Note:** "DINOv3" is not an official release. DINOv2 with registers (reg4) represents
> the latest advancement in the DINO family and is included here as its successor.
"""))

# ─────────────────────────────────────────────────────────────────────────────
cells.append(code_cell("""
# ============================================================
# Cell 0 — Package Installation
# Run once, then restart the kernel if needed.
# ============================================================
import subprocess, sys

pkgs = [
    "pytorch-grad-cam",
    "captum",
    "python-louvain",
    "networkx",
    "scikit-image",
    "scikit-learn",
    "timm>=0.9.0",
    "pandas",
    "matplotlib",
    "Pillow",
    "scipy",
]
for p in pkgs:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", p], check=False)

print("Installation complete. Restart kernel if this was the first install.")
"""))

# ─────────────────────────────────────────────────────────────────────────────
cells.append(code_cell("""
# ============================================================
# Cell 1 — Imports & Shared XAI Helper Functions
# ============================================================
import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import pandas as pd
from PIL import Image
from skimage.transform import resize
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats

import torch
import torchvision.transforms as T
import timm

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from captum.attr import IntegratedGradients
import community as community_louvain

print(f"PyTorch : {torch.__version__}")
print(f"timm    : {timm.__version__}")
print(f"CUDA    : {torch.cuda.is_available()}")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device  : {DEVICE}")

# ── Image list (15 images, same as CNN/ViT benchmark) ────────────────────────
IMAGE_CLASS_PAIRS = [
    ("images/parrot.jpg",           88),
    ("images/elephant.jpg",        386),
    ("images/cat.jpg",             281),
    ("images/german-shepherd.jpg", 235),
    ("images/tiger.jpg",           292),
    ("images/goldfish.jpg",          1),
    ("images/cheetah.jpg",         293),
    ("images/plane.jpg",           404),
    ("images/fire truck.jpg",      555),
    ("images/bus.jpg",             874),
    ("images/ambulance.jpeg",      407),
    ("images/racing car.jpg",      751),
    ("images/submarine.jpg",       833),
    ("images/tractor.jpeg",        866),
    ("images/whale.jpg",           148),
]
print(f"{len(IMAGE_CLASS_PAIRS)} images configured.")

# ── Preprocessing ─────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def load_image(img_path, input_size=224):
    img = Image.open(img_path).convert("RGB").resize((input_size, input_size))
    tf  = T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return img, tf(img).unsqueeze(0).to(DEVICE)

def norm_resize(hm, size=224):
    if hm is None:
        return None
    hm = hm - hm.min()
    hm = hm / (hm.max() + 1e-8)
    if hm.shape[0] != size or hm.shape[1] != size:
        hm = resize(hm, (size, size), anti_aliasing=True)
    return hm.astype(np.float32)

def combine_hm(a, b):
    if a is None or b is None:
        return a if a is not None else b
    return (a * b)

# ── Reshape transforms ────────────────────────────────────────────────────────

def make_vit_reshape(h=14, w=14, n_skip=1):
    \"\"\"Standard ViT reshape: skip n_skip tokens (CLS [+registers]), reshape to 2D grid.\"\"\"
    def _reshape(tensor, target_size):
        tensor = tensor[:, n_skip:n_skip + h * w, :]
        B, N, C = tensor.shape
        return tensor.reshape(B, h, w, C).permute(0, 3, 1, 2)
    return _reshape

def make_swin_reshape(h=7, w=7):
    \"\"\"Swin-T/B: no CLS, last stage outputs [B, h*w, C].\"\"\"
    def _reshape(tensor, target_size):
        B, N, C = tensor.shape
        return tensor.reshape(B, h, w, C).permute(0, 3, 1, 2)
    return _reshape

# ── Grad-CAM helpers ──────────────────────────────────────────────────────────

def compute_gradcam(model, inp, cls_idx, layers, reshape):
    with GradCAM(model=model, target_layers=layers,
                 reshape_transform=reshape) as cam:
        out = cam(input_tensor=inp,
                  targets=[ClassifierOutputTarget(cls_idx)])
    return out[0]

def compute_gradcam_pp(model, inp, cls_idx, layers, reshape):
    with GradCAMPlusPlus(model=model, target_layers=layers,
                         reshape_transform=reshape) as cam:
        out = cam(input_tensor=inp,
                  targets=[ClassifierOutputTarget(cls_idx)])
    return out[0]

# ── Integrated Gradients ──────────────────────────────────────────────────────

def compute_ig(model, inp, cls_idx, n_steps=50):
    ig   = IntegratedGradients(model)
    base = torch.zeros_like(inp).to(DEVICE)
    attr = ig.attribute(inp, baselines=base, target=cls_idx,
                        n_steps=n_steps, return_convergence_delta=False)
    hm   = attr.squeeze().cpu().detach().numpy()
    return np.mean(np.abs(hm), axis=0)

# ── Attention Rollout ─────────────────────────────────────────────────────────

def compute_attention_rollout(model, inp, patch_h, patch_w, has_cls=True):
    attns = []
    hooks = []
    def _hook(module, inp_, out_):
        if isinstance(out_, tuple):
            out_ = out_[0]
        if out_.dim() == 4:
            attns.append(out_.detach().cpu())
    for block in model.blocks:
        h = block.attn.register_forward_hook(_hook)
        hooks.append(h)
    with torch.no_grad():
        model(inp)
    for h in hooks:
        h.remove()
    if not attns:
        return None
    rollout = torch.eye(attns[0].shape[-1])
    for a in attns:
        a_mean  = a.mean(dim=1)[0]
        a_id    = torch.eye(a_mean.shape[-1])
        a_plus  = (a_mean + a_id) / 2
        a_plus /= a_plus.sum(dim=-1, keepdim=True)
        rollout = torch.mm(a_plus, rollout)
    if has_cls:
        mask = rollout[0, 1:1 + patch_h * patch_w]
    else:
        mask = rollout[0, :patch_h * patch_w]
    mask = mask.reshape(patch_h, patch_w).numpy()
    return mask

# ── Graph analysis ────────────────────────────────────────────────────────────

def create_graph(hm, threshold=0.3):
    h, w = hm.shape
    G = nx.Graph()
    for r in range(h):
        for c in range(w):
            if hm[r, c] > threshold:
                G.add_node((r, c))
    for r in range(h):
        for c in range(w):
            if (r, c) not in G:
                continue
            for dr, dc in [(0, 1), (1, 0)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in G:
                    w_edge = (hm[r, c] + hm[nr, nc]) / 2
                    G.add_edge((r, c), (nr, nc), weight=float(w_edge))
    return G

def cluster_louvain(G):
    return community_louvain.best_partition(G, weight="weight")

def cluster_spectral(G, k=None):
    if len(G) < 3:
        return {n: 0 for n in G.nodes()}
    k = k or min(5, len(G) // 5 + 1)
    A = nx.to_numpy_array(G, weight="weight")
    lb = SpectralClustering(k, affinity="precomputed",
                             random_state=42, n_init=5).fit_predict(A)
    return dict(zip(G.nodes(), lb))

def cluster_agglomerative(G, k=None):
    if len(G) < 3:
        return {n: 0 for n in G.nodes()}
    k = k or min(5, len(G) // 5 + 1)
    A = nx.to_numpy_array(G, weight="weight")
    D = 1 - cosine_similarity(A)
    D = np.clip(D, 0, None)
    lb = AgglomerativeClustering(k, metric="precomputed",
                                  linkage="average").fit_predict(D)
    return dict(zip(G.nodes(), lb))

def graph_silhouette(G):
    \"\"\"Mean silhouette score from Louvain clustering (graph quality proxy).\"\"\"
    if len(G) < 4:
        return float("nan")
    try:
        partition = cluster_louvain(G)
        labels    = [partition[n] for n in G.nodes()]
        if len(set(labels)) < 2:
            return float("nan")
        A = nx.to_numpy_array(G, weight="weight")
        D = 1 - cosine_similarity(A)
        D = np.clip(D, 0, None)
        return float(silhouette_score(D, labels, metric="precomputed"))
    except Exception:
        return float("nan")

# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_heatmaps(img_pil, gc, gcc, ig, ar, comb, title=""):
    img_np = np.array(img_pil.resize((224, 224))) / 255.0
    R      = lambda x: norm_resize(x) if x is not None else None
    fig, axes = plt.subplots(1, 6, figsize=(26, 4))
    fig.suptitle(title, fontsize=10, fontweight="bold", y=1.02)
    def show_panel(ax, base, overlay, label, alpha=0.5):
        if base is not None:
            ax.imshow(base)
        if overlay is not None:
            ax.imshow(overlay, cmap="jet", alpha=alpha)
        ax.set_title(label, fontsize=8)
        ax.axis("off")
    show_panel(axes[0], img_np, None,   "Original")
    show_panel(axes[1], img_np, R(gc),  "Grad-CAM")
    show_panel(axes[2], img_np, R(gcc), "Grad-CAM++")
    show_panel(axes[3], None,   R(ig),  "Integ. Grads")
    if ar is not None:
        show_panel(axes[4], img_np, R(ar), "Attn Rollout")
    else:
        axes[4].text(0.5, 0.5, "Attention Rollout\\nN/A", ha="center", va="center",
                     transform=axes[4].transAxes, fontsize=8, color="grey")
        axes[4].axis("off")
    show_panel(axes[5], img_np, R(comb), "Combined (GC×IG)", alpha=0.9)
    plt.tight_layout()
    plt.show()

print("\\nAll XAI helper functions loaded.")
"""))

# ─────────────────────────────────────────────────────────────────────────────
cells.append(code_cell("""
# ============================================================
# Cell 2 — Quantitative Metrics
# ============================================================
# Implements Deletion AUC, Insertion AUC, Faithfulness Correlation,
# Gini Sparseness — all without external dependencies.

def _get_prob(model, tensor, class_idx):
    with torch.no_grad():
        logits = model(tensor)
        return torch.softmax(logits, dim=1)[0, class_idx].item()


def deletion_auc(model, inp, saliency, class_idx, steps=20):
    \"\"\"
    Deletion AUC: progressively black-out the most salient pixels.
    LOWER score = better (removing salient pixels hurts the model more).
    Range: [0, 1]
    \"\"\"
    model.eval()
    h, w = inp.shape[-2], inp.shape[-1]
    sal  = norm_resize(saliency, size=h)
    flat = sal.flatten()
    idx  = np.argsort(flat)[::-1]          # most salient first

    scores = []
    for k in np.linspace(0, 1, steps + 1):
        n_mask = int(k * len(flat))
        mask   = np.ones(len(flat), dtype=np.float32)
        if n_mask > 0:
            mask[idx[:n_mask]] = 0.0
        mask_t = torch.tensor(mask.reshape(h, w),
                               dtype=torch.float32, device=DEVICE)
        masked = inp * mask_t.unsqueeze(0).unsqueeze(0)
        scores.append(_get_prob(model, masked, class_idx))

    return float(np.trapz(scores, dx=1.0 / steps))


def insertion_auc(model, inp, saliency, class_idx, steps=20):
    \"\"\"
    Insertion AUC: progressively reveal the most salient pixels from a black baseline.
    HIGHER score = better.
    Range: [0, 1]
    \"\"\"
    model.eval()
    h, w  = inp.shape[-2], inp.shape[-1]
    sal   = norm_resize(saliency, size=h)
    flat  = sal.flatten()
    idx   = np.argsort(flat)[::-1]

    baseline = torch.zeros_like(inp, device=DEVICE)
    scores   = []
    for k in np.linspace(0, 1, steps + 1):
        n_rev = int(k * len(flat))
        mask  = np.zeros(len(flat), dtype=np.float32)
        if n_rev > 0:
            mask[idx[:n_rev]] = 1.0
        mask_t   = torch.tensor(mask.reshape(h, w),
                                 dtype=torch.float32, device=DEVICE)
        revealed = baseline * (1 - mask_t.unsqueeze(0).unsqueeze(0)) + \\
                   inp      *      mask_t.unsqueeze(0).unsqueeze(0)
        scores.append(_get_prob(model, revealed, class_idx))

    return float(np.trapz(scores, dx=1.0 / steps))


def faithfulness_corr(model, inp, saliency, class_idx,
                       n_subsets=40, subset_size=100):
    \"\"\"
    Faithfulness Correlation (Alvarez-Melis & Jaakkola 2018).
    Pearson r between attribution sum and prediction drop over random pixel subsets.
    HIGHER = more faithful.
    \"\"\"
    model.eval()
    h, w     = inp.shape[-2], inp.shape[-1]
    sal      = norm_resize(saliency, size=h)
    flat     = sal.flatten()
    n_pixels = len(flat)

    base_p = _get_prob(model, inp, class_idx)
    attr_sums, pred_drops = [], []

    for _ in range(n_subsets):
        sub  = np.random.choice(n_pixels, size=subset_size, replace=False)
        attr_sums.append(float(flat[sub].sum()))

        mask   = np.ones(n_pixels, dtype=np.float32)
        mask[sub] = 0.0
        mask_t = torch.tensor(mask.reshape(h, w),
                               dtype=torch.float32, device=DEVICE)
        masked = inp * mask_t.unsqueeze(0).unsqueeze(0)
        pred_drops.append(base_p - _get_prob(model, masked, class_idx))

    if np.std(attr_sums) < 1e-10 or np.std(pred_drops) < 1e-10:
        return 0.0
    return float(np.corrcoef(attr_sums, pred_drops)[0, 1])


def gini_sparseness(saliency):
    \"\"\"
    Gini coefficient of the attribution map.
    HIGHER = more concentrated / focused explanation.
    Range: [0, 1]
    \"\"\"
    vals = np.abs(saliency.flatten())
    vals = np.sort(vals)
    n    = len(vals)
    if n == 0 or vals.sum() < 1e-10:
        return 0.0
    idx  = np.arange(1, n + 1)
    return float((2 * np.sum(idx * vals) - (n + 1) * vals.sum()) /
                 (n * vals.sum()))


def run_all_metrics(model, inp, saliency_dict, class_idx,
                    model_name, paradigm, img_name,
                    steps_del=10, steps_ins=10, n_subsets=20):
    \"\"\"
    Run all quantitative metrics for one image.
    Returns list of dicts (one per XAI method).
    \"\"\"
    records = []
    for method, sal in saliency_dict.items():
        if sal is None:
            continue
        sal_n = norm_resize(sal)
        d_auc = f_corr = i_auc = spars = float("nan")
        try:
            d_auc  = deletion_auc(model, inp, sal_n, class_idx, steps=steps_del)
        except Exception as e:
            print(f"      [{method}] Deletion AUC error: {e}")
        try:
            i_auc  = insertion_auc(model, inp, sal_n, class_idx, steps=steps_ins)
        except Exception as e:
            print(f"      [{method}] Insertion AUC error: {e}")
        try:
            f_corr = faithfulness_corr(model, inp, sal_n, class_idx,
                                        n_subsets=n_subsets)
        except Exception as e:
            print(f"      [{method}] Faithfulness error: {e}")
        try:
            spars  = gini_sparseness(sal_n)
        except Exception:
            pass

        records.append({
            "Model":            model_name,
            "Paradigm":         paradigm,
            "Image":            img_name,
            "Method":           method,
            "Deletion_AUC":     d_auc,
            "Insertion_AUC":    i_auc,
            "Faithfulness_Corr": f_corr,
            "Sparseness":       spars,
        })
    return records


print("Quantitative metric functions loaded.")
"""))

# ─────────────────────────────────────────────────────────────────────────────
cells.append(code_cell("""
# ============================================================
# Cell 3 — Pipeline Runner & Results Framework
# ============================================================

ALL_RESULTS = []   # accumulates dicts from run_all_metrics()

def run_foundation_pipeline(cfg, image_pairs,
                             run_viz=True,
                             run_graph=True,
                             run_quant=True):
    \"\"\"
    Universal pipeline for any ViT-family foundation model.

    cfg keys
    --------
    name        : display name, e.g. "DINOv2-B/14"
    paradigm    : training paradigm label (for the results table)
    timm_name   : timm model identifier
    get_layers  : callable(model) -> layer or [layer] for Grad-CAM
    reshape     : reshape_transform or None
    patch_h/w   : spatial grid size (number of patches per side)
    has_cls     : bool — does the model have a CLS token?
    do_rollout  : bool — apply Attention Rollout?
    input_size  : int (default 224)
    \"\"\"
    global ALL_RESULTS

    name       = cfg["name"]
    paradigm   = cfg.get("paradigm", "Unknown")
    in_sz      = cfg.get("input_size", 224)
    patch_h    = cfg.get("patch_h", 14)
    patch_w    = cfg.get("patch_w", 14)
    has_cls    = cfg.get("has_cls", True)
    do_rollout = cfg.get("do_rollout", True)

    sep = "=" * 72
    print(sep)
    print(f"  MODEL    : {name}")
    print(f"  Paradigm : {paradigm}")
    print(f"  timm     : {cfg['timm_name']}")
    print(sep)

    # ── Load model ────────────────────────────────────────────────────────
    try:
        model = timm.create_model(cfg["timm_name"], pretrained=True)
        model = model.to(DEVICE).eval()
    except Exception as err:
        print(f"  ERROR loading model: {err}")
        return

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Params   : {n_params:.1f}M")

    try:
        layers = cfg["get_layers"](model)
        if not isinstance(layers, list):
            layers = [layers]
    except Exception as err:
        print(f"  ERROR resolving target layer: {err}")
        return

    reshape = cfg.get("reshape")

    # ── Per-image loop ─────────────────────────────────────────────────────
    for img_path, cls_idx in image_pairs:
        if not os.path.exists(img_path):
            print(f"  [skip] not found: {img_path}")
            continue
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"\\n  Image : {img_name:<28}  class={cls_idx}")

        img_pil, inp = load_image(img_path, in_sz)

        # — Grad-CAM —
        try:
            gc = compute_gradcam(model, inp, cls_idx, layers, reshape)
        except Exception as e:
            print(f"    GradCAM   error: {e}")
            gc = np.zeros((in_sz, in_sz))

        # — Grad-CAM++ —
        try:
            gcc = compute_gradcam_pp(model, inp, cls_idx, layers, reshape)
        except Exception as e:
            print(f"    GradCAM++ error: {e}")
            gcc = np.zeros((in_sz, in_sz))

        # — Integrated Gradients —
        try:
            ig = compute_ig(model, inp, cls_idx)
        except Exception as e:
            print(f"    IG        error: {e}")
            ig = np.zeros((in_sz, in_sz))

        # — Attention Rollout —
        ar = None
        if do_rollout and hasattr(model, "blocks"):
            try:
                ar = compute_attention_rollout(model, inp, patch_h, patch_w,
                                               has_cls=has_cls)
            except Exception as e:
                print(f"    Attn Rollout error: {e}")

        # — Combined —
        gc_r  = norm_resize(gc)
        ig_r  = norm_resize(ig)
        comb  = combine_hm(gc_r, ig_r)

        # — Visualisation —
        if run_viz:
            plot_heatmaps(img_pil, gc, gcc, ig, ar, comb,
                          title=f"{name}  |  {img_name}  (class={cls_idx})")

        # — Graph clustering —
        if run_graph and comb is not None:
            G = create_graph(comb)
            sil = graph_silhouette(G)
            print(f"    Graph: {len(G)} nodes, {G.number_of_edges()} edges  "
                  f"| Silhouette: {sil:.3f}" if not np.isnan(sil) else
                  f"    Graph: {len(G)} nodes  | Silhouette: N/A")

        # — Quantitative metrics —
        if run_quant:
            saliency_dict = {
                "GradCAM":    gc,
                "GradCAM++":  gcc,
                "IntegGrad":  ig,
                "Combined":   comb,
            }
            if ar is not None:
                saliency_dict["AttnRollout"] = ar
            records = run_all_metrics(
                model, inp, saliency_dict, cls_idx,
                model_name=name, paradigm=paradigm, img_name=img_name,
            )
            ALL_RESULTS.extend(records)
            # Quick print of per-image mean scores
            for r in records:
                print(f"    [{r['Method']:12s}]  "
                      f"Del={r['Deletion_AUC']:.3f}  "
                      f"Ins={r['Insertion_AUC']:.3f}  "
                      f"Faith={r['Faithfulness_Corr']:.3f}  "
                      f"Sparse={r['Sparseness']:.3f}")

    print()
    print(f"  {name} — pipeline complete.")
    print()


print("Pipeline runner ready.")
print(f"ALL_RESULTS will accumulate across all models.")
"""))

# ─────────────────────────────────────────────────────────────────────────────
cells.append(md_cell("## Baseline — Supervised ViT-B/16 (ImageNet-21k → 1k)"))

cells.append(code_cell("""
# ============================================================
# Cell 4 — Supervised Baseline: ViT-B/16
# Pre-training  : ImageNet-21k supervised classification
# Fine-tuning   : ImageNet-1k
# Paper         : Dosovitskiy et al., ICLR 2021
# ============================================================

SUPERVISED = dict(
    name       = "ViT-B/16-Supervised",
    paradigm   = "Supervised (IN21k→IN1k)",
    timm_name  = "vit_base_patch16_224.augreg_in21k_ft_in1k",
    get_layers = lambda m: m.blocks[-1].norm1,
    reshape    = make_vit_reshape(h=14, w=14, n_skip=1),
    patch_h    = 14,
    patch_w    = 14,
    has_cls    = True,
    do_rollout = True,
    input_size = 224,
)

run_foundation_pipeline(SUPERVISED, IMAGE_CLASS_PAIRS)
"""))

# ─────────────────────────────────────────────────────────────────────────────
cells.append(md_cell("""
## 1 — Self-Distillation: DINO Family

**Key idea:** A student ViT is trained to match the output distribution of a
momentum-updated teacher ViT — with no labels, no text, no reconstruction.

| Model | Paper | Training Data |
|---|---|---|
| DINOv1-B/16 | Caron et al., ICCV 2021 | ImageNet-1k (unsupervised) |
| DINOv1-B/8  | Same | ImageNet-1k (unsupervised) |
| DINOv2-S/14 | Oquab et al., TMLR 2024 | LVD-142M (curated) |
| DINOv2-B/14 | Same | LVD-142M |
| DINOv2-B/14+Reg | Darcet et al., ICLR 2024 | LVD-142M (+register tokens) |

> **DINOv3 note:** No official DINOv3 exists as of 2025.
> DINOv2 with registers (`reg4`) is the latest architectural advancement
> in the DINO line and is treated as its successor here.
"""))

cells.append(code_cell("""
# ============================================================
# Cell 5 — DINOv1 Models
# ============================================================
# DINOv1 uses self-distillation with multi-crop augmentation.
# Produces highly semantic, class-discriminative attention maps.
# patch_size=16 → grid = 224/16 = 14×14

DINO_V1_MODELS = [
    dict(
        name       = "DINOv1-B/16",
        paradigm   = "Self-Distillation (DINOv1)",
        timm_name  = "vit_base_patch16_224.dino",
        get_layers = lambda m: m.blocks[-1].norm1,
        reshape    = make_vit_reshape(h=14, w=14, n_skip=1),
        patch_h    = 14, patch_w = 14, has_cls = True,
        do_rollout = True, input_size = 224,
    ),
    dict(
        name       = "DINOv1-B/8",
        paradigm   = "Self-Distillation (DINOv1)",
        timm_name  = "vit_base_patch8_224.dino",
        # patch_size=8 → grid = 224/8 = 28×28
        get_layers = lambda m: m.blocks[-1].norm1,
        reshape    = make_vit_reshape(h=28, w=28, n_skip=1),
        patch_h    = 28, patch_w = 28, has_cls = True,
        do_rollout = True, input_size = 224,
    ),
]

for cfg in DINO_V1_MODELS:
    run_foundation_pipeline(cfg, IMAGE_CLASS_PAIRS)
"""))

cells.append(code_cell("""
# ============================================================
# Cell 6 — DINOv2 Models (+ Registers variant)
# ============================================================
# DINOv2 adds DINO + iBOT (masked image modeling) objectives.
# patch_size=14 → grid = 224/14 = 16×16  (256 patch tokens)
#
# DINOv2+Reg: 4 register tokens appended AFTER CLS in sequence.
#   Sequence: [CLS, reg1, reg2, reg3, reg4, p1, ..., p256]
#   → n_skip = 5 to skip CLS + 4 registers before the patch tokens.

DINO_V2_MODELS = [
    dict(
        name       = "DINOv2-S/14",
        paradigm   = "Self-Distillation (DINOv2)",
        timm_name  = "vit_small_patch14_dinov2.lvd142m",
        get_layers = lambda m: m.blocks[-1].norm1,
        reshape    = make_vit_reshape(h=16, w=16, n_skip=1),
        patch_h    = 16, patch_w = 16, has_cls = True,
        do_rollout = True, input_size = 224,
    ),
    dict(
        name       = "DINOv2-B/14",
        paradigm   = "Self-Distillation (DINOv2)",
        timm_name  = "vit_base_patch14_dinov2.lvd142m",
        get_layers = lambda m: m.blocks[-1].norm1,
        reshape    = make_vit_reshape(h=16, w=16, n_skip=1),
        patch_h    = 16, patch_w = 16, has_cls = True,
        do_rollout = True, input_size = 224,
    ),
    dict(
        name       = "DINOv2-B/14+Reg4",
        paradigm   = "Self-Distillation (DINOv2+Registers)",
        timm_name  = "vit_base_patch14_reg4_dinov2.lvd142m",
        # Skip CLS (1) + register tokens (4) = n_skip=5
        get_layers = lambda m: m.blocks[-1].norm1,
        reshape    = make_vit_reshape(h=16, w=16, n_skip=5),
        patch_h    = 16, patch_w = 16, has_cls = True,
        do_rollout = True, input_size = 224,
    ),
]

for cfg in DINO_V2_MODELS:
    run_foundation_pipeline(cfg, IMAGE_CLASS_PAIRS)
"""))

# ─────────────────────────────────────────────────────────────────────────────
cells.append(md_cell("""
## 2 — Masked Autoencoding: MAE

**Key idea:** Randomly mask 75% of image patches and train the encoder-decoder
to reconstruct raw pixel values. The encoder learns rich patch-level representations
by predicting missing spatial content.

| Model | Paper | Training Data |
|---|---|---|
| MAE ViT-B/16 | He et al., CVPR 2022 | ImageNet-1k (self-supervised) |
| MAE ViT-L/16 | Same | ImageNet-1k |

timm provides MAE checkpoints **fine-tuned** on ImageNet-1k with a linear head,
making them directly compatible with our classification-based XAI pipeline.
"""))

cells.append(code_cell("""
# ============================================================
# Cell 7 — MAE Models
# Pre-training  : Masked Autoencoding (pixel reconstruction)
# Fine-tuning   : ImageNet-1k classification (linear probe)
# Paper         : He et al., CVPR 2022
# ============================================================

MAE_MODELS = [
    dict(
        name       = "MAE-ViT-B/16",
        paradigm   = "Masked Autoencoding (MAE)",
        timm_name  = "vit_base_patch16_224.mae",
        get_layers = lambda m: m.blocks[-1].norm1,
        reshape    = make_vit_reshape(h=14, w=14, n_skip=1),
        patch_h    = 14, patch_w = 14, has_cls = True,
        do_rollout = True, input_size = 224,
    ),
    dict(
        name       = "MAE-ViT-L/16",
        paradigm   = "Masked Autoencoding (MAE)",
        timm_name  = "vit_large_patch16_224.mae",
        get_layers = lambda m: m.blocks[-1].norm1,
        reshape    = make_vit_reshape(h=14, w=14, n_skip=1),
        patch_h    = 14, patch_w = 14, has_cls = True,
        do_rollout = True, input_size = 224,
    ),
]

for cfg in MAE_MODELS:
    run_foundation_pipeline(cfg, IMAGE_CLASS_PAIRS)
"""))

# ─────────────────────────────────────────────────────────────────────────────
cells.append(md_cell("""
## 3 — Masked Token Prediction: BEiT3

**Key idea:** Tokenize image patches into discrete visual tokens (via VQ-VAE / DALL-E tokenizer),
then train the ViT to predict masked visual tokens — analogous to BERT's masked language modeling.

BEiT3 further unifies vision and language with a shared multi-way Transformer.

| Model | Paper | Training Data |
|---|---|---|
| BEiT3-B/16 | Wang et al., CVPR 2023 | ImageNet-22k + language data |
"""))

cells.append(code_cell("""
# ============================================================
# Cell 8 — BEiT3 Models
# Pre-training  : Masked Token Prediction (visual + language tokens)
# Fine-tuning   : ImageNet-22k → 1k classification
# Paper         : Wang et al., CVPR 2023
# ============================================================
# Note: BEiT3 uses a standard ViT backbone with patch_size=16.
# timm exposes the vision-side classification head directly.

BEIT3_MODELS = [
    dict(
        name       = "BEiT3-B/16",
        paradigm   = "Masked Token Prediction (BEiT3)",
        timm_name  = "beit3_base_patch16_224.in22k_ft_in1k",
        get_layers = lambda m: m.blocks[-1].norm1,
        reshape    = make_vit_reshape(h=14, w=14, n_skip=1),
        patch_h    = 14, patch_w = 14, has_cls = True,
        do_rollout = True, input_size = 224,
    ),
    dict(
        name       = "BEiT3-L/16",
        paradigm   = "Masked Token Prediction (BEiT3)",
        timm_name  = "beit3_large_patch16_224.in22k_ft_in1k",
        get_layers = lambda m: m.blocks[-1].norm1,
        reshape    = make_vit_reshape(h=14, w=14, n_skip=1),
        patch_h    = 14, patch_w = 14, has_cls = True,
        do_rollout = True, input_size = 224,
    ),
]

for cfg in BEIT3_MODELS:
    run_foundation_pipeline(cfg, IMAGE_CLASS_PAIRS)
"""))

# ─────────────────────────────────────────────────────────────────────────────
cells.append(md_cell("""
## 4 — Language-Image Contrastive: CLIP Family

**Key idea:** Align image and text representations in a shared embedding space
using contrastive loss over (image, caption) pairs at web scale.

The visual encoder learns semantically rich representations driven by natural language supervision.

| Model | Paper | Training Data |
|---|---|---|
| CLIP ViT-B/16 (OpenAI) | Radford et al., ICML 2021 | 400M web image-text pairs |
| CLIP ViT-B/16 (LAION-2B) | Schuhmann et al., NeurIPS 2022 | LAION-2B (2B pairs) |
| MetaCLIP ViT-B/16 | Xu et al., ICLR 2024 | 2.5B metadata-curated pairs |

> timm provides CLIP visual encoders **fine-tuned** on ImageNet-1k, making them
> compatible with our classification-based Grad-CAM pipeline.
"""))

cells.append(code_cell("""
# ============================================================
# Cell 9 — CLIP Models (Language-Image Contrastive)
# Pre-training  : Contrastive Language-Image Pretraining (CLIP)
# Fine-tuning   : ImageNet-12k → 1k (via timm linear probe)
# ============================================================

CLIP_MODELS = [
    dict(
        name       = "CLIP-ViT-B/16-OpenAI",
        paradigm   = "Language-Image Contrastive (CLIP/OpenAI)",
        timm_name  = "vit_base_patch16_clip_224.openai_ft_in12k_in1k",
        get_layers = lambda m: m.blocks[-1].norm1,
        reshape    = make_vit_reshape(h=14, w=14, n_skip=1),
        patch_h    = 14, patch_w = 14, has_cls = True,
        do_rollout = True, input_size = 224,
    ),
    dict(
        name       = "CLIP-ViT-B/16-LAION2B",
        paradigm   = "Language-Image Contrastive (OpenCLIP/LAION)",
        timm_name  = "vit_base_patch16_clip_224.laion2b_ft_in12k_in1k",
        get_layers = lambda m: m.blocks[-1].norm1,
        reshape    = make_vit_reshape(h=14, w=14, n_skip=1),
        patch_h    = 14, patch_w = 14, has_cls = True,
        do_rollout = True, input_size = 224,
    ),
    dict(
        name       = "MetaCLIP-ViT-B/16",
        paradigm   = "Language-Image Contrastive (MetaCLIP)",
        timm_name  = "vit_base_patch16_clip_224.metaclip_2pt5b",
        get_layers = lambda m: m.blocks[-1].norm1,
        reshape    = make_vit_reshape(h=14, w=14, n_skip=1),
        patch_h    = 14, patch_w = 14, has_cls = True,
        do_rollout = True, input_size = 224,
    ),
]

for cfg in CLIP_MODELS:
    run_foundation_pipeline(cfg, IMAGE_CLASS_PAIRS)
"""))

# ─────────────────────────────────────────────────────────────────────────────
cells.append(md_cell("""
## 5 — MIM + Contrastive Hybrid: EVA-02

**Key idea:** EVA-02 combines Masked Image Modeling (predict CLIP features of masked patches)
with contrastive image-text alignment. The model uses a ViT encoder pre-trained to reconstruct
the masked visual features predicted by a CLIP teacher.

| Model | Paper | Training Data |
|---|---|---|
| EVA02-B/14 (MIM, IN22k) | Fang et al., ECCV 2024 | ImageNet-22k |

This is the most architecturally complex model in the benchmark — trained with
a combined MIM + contrastive signal — and serves as the ceiling comparison.
"""))

cells.append(code_cell("""
# ============================================================
# Cell 10 — EVA-02 (MIM + Contrastive Hybrid)
# Pre-training  : MIM with CLIP teacher (masked feature prediction)
# Fine-tuning   : ImageNet-22k classification
# Paper         : Fang et al., ECCV 2024
# Note          : patch_size=14, input=224 → grid = 224/14 = 16×16
# ============================================================

EVA_MODELS = [
    dict(
        name       = "EVA02-B/14-MIM",
        paradigm   = "MIM + Contrastive Hybrid (EVA02)",
        timm_name  = "eva02_base_patch14_224.mim_in22k",
        get_layers = lambda m: m.blocks[-1].norm1,
        reshape    = make_vit_reshape(h=16, w=16, n_skip=1),
        patch_h    = 16, patch_w = 16, has_cls = True,
        do_rollout = True, input_size = 224,
    ),
]

for cfg in EVA_MODELS:
    run_foundation_pipeline(cfg, IMAGE_CLASS_PAIRS)
"""))

# ─────────────────────────────────────────────────────────────────────────────
cells.append(md_cell("## Summary Results & Export"))

cells.append(code_cell("""
# ============================================================
# Cell 11 — Aggregate Results Table
# ============================================================

if not ALL_RESULTS:
    print("No results collected yet. Run the model cells above first.")
else:
    df = pd.DataFrame(ALL_RESULTS)

    # ── Per-model × per-method mean across all images ─────────────────────
    summary = (
        df.groupby(["Paradigm", "Model", "Method"])
          [["Deletion_AUC", "Insertion_AUC", "Faithfulness_Corr", "Sparseness"]]
          .agg(["mean", "std"])
          .round(4)
    )

    print("\\n" + "=" * 100)
    print("RESULTS TABLE — Mean ± Std across all images")
    print("=" * 100)
    print(summary.to_string())

    # ── Best method per model ─────────────────────────────────────────────
    mean_df = (
        df.groupby(["Model", "Method"])
          [["Insertion_AUC", "Faithfulness_Corr"]]
          .mean()
          .reset_index()
    )
    best = mean_df.loc[mean_df.groupby("Model")["Insertion_AUC"].idxmax()]
    print("\\n\\nBest XAI method per model (by Insertion AUC):")
    print(best.to_string(index=False))
"""))

cells.append(code_cell("""
# ============================================================
# Cell 12 — Export to CSV and LaTeX
# ============================================================
import os

if not ALL_RESULTS:
    print("No results to export.")
else:
    df = pd.DataFrame(ALL_RESULTS)

    # — CSV (full results) —
    csv_path = "foundation_xai_results_full.csv"
    df.to_csv(csv_path, index=False)
    print(f"Full results saved: {csv_path}  ({len(df)} rows)")

    # — CSV (aggregated mean) —
    mean_summary = (
        df.groupby(["Paradigm", "Model", "Method"])
          [["Deletion_AUC", "Insertion_AUC", "Faithfulness_Corr", "Sparseness"]]
          .mean()
          .round(4)
          .reset_index()
    )
    agg_path = "foundation_xai_results_aggregated.csv"
    mean_summary.to_csv(agg_path, index=False)
    print(f"Aggregated results saved: {agg_path}  ({len(mean_summary)} rows)")

    # — LaTeX table (paper-ready) —
    # Pivot: rows = Model, cols = Method × Metric
    pivot = mean_summary.pivot_table(
        index=["Paradigm", "Model"],
        columns="Method",
        values=["Deletion_AUC", "Insertion_AUC", "Faithfulness_Corr", "Sparseness"],
    ).round(3)

    latex_str = pivot.to_latex(
        multicolumn=True,
        multirow=True,
        float_format="{:.3f}".format,
        caption="Quantitative XAI evaluation across foundation model training paradigms. "
                "Deletion AUC (lower = better), Insertion AUC (higher = better), "
                "Faithfulness Correlation (higher = better), Sparseness/Gini (higher = more focused).",
        label="tab:foundation_xai_results",
    )

    tex_path = "foundation_xai_table.tex"
    with open(tex_path, "w") as f:
        f.write(latex_str)
    print(f"LaTeX table saved: {tex_path}")
    print()
    print("--- LaTeX preview ---")
    print(latex_str[:2000])
"""))

cells.append(code_cell("""
# ============================================================
# Cell 13 — Cross-Paradigm Comparison Plot
# ============================================================
# Bar chart comparing mean Insertion AUC and Faithfulness Correlation
# across training paradigms and XAI methods.

if not ALL_RESULTS:
    print("No results to plot.")
else:
    df = pd.DataFrame(ALL_RESULTS)

    for metric, label, better in [
        ("Insertion_AUC",    "Insertion AUC (↑ better)",       True),
        ("Faithfulness_Corr","Faithfulness Correlation (↑ better)", True),
        ("Deletion_AUC",     "Deletion AUC (↓ better)",        False),
        ("Sparseness",       "Gini Sparseness (↑ more focused)", True),
    ]:
        fig, ax = plt.subplots(figsize=(14, 5))
        plot_df = (
            df.groupby(["Model", "Method"])[metric]
              .mean()
              .unstack("Method")
        )
        plot_df.plot(kind="bar", ax=ax, width=0.7, colormap="tab10")
        ax.set_title(f"Foundation Model XAI Benchmark — {label}", fontsize=12)
        ax.set_xlabel("Model", fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.legend(title="XAI Method", bbox_to_anchor=(1.01, 1), loc="upper left")
        ax.tick_params(axis="x", rotation=40)
        plt.tight_layout()
        plt.savefig(f"fig_{metric.lower()}.pdf", dpi=300, bbox_inches="tight")
        plt.show()
        print(f"Saved: fig_{metric.lower()}.pdf")
"""))

# ─────────────────────────────────────────────────────────────────────────────
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "cells": cells,
}

out_path = "foundation_model_xai.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook written: {out_path}")
print(f"Total cells: {len(cells)}")
