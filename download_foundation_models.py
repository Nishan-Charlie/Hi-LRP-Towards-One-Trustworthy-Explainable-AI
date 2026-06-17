#!/usr/bin/env python
# ============================================================
# download_foundation_models.py
# Pre-download / cache all pretrained foundation models used by
# foundation_model_xai.ipynb from the Hugging Face Hub.
#
# timm.create_model(..., pretrained=True) fetches the weights
# from HF Hub into the local cache, so the notebook can then run
# without re-downloading.
#
# Usage:
#   python download_foundation_models.py
# ============================================================

import os
import sys
import time

# ── Load HF token from .env (HF_TOKEN / HUGGING_FACE_HUB_TOKEN) ───────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Lightweight fallback if python-dotenv is not installed.
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path) as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    os.environ.setdefault(key.strip(), val.strip())

# ── Foundation models referenced in foundation_model_xai.ipynb ────────────────
# Grouped by training paradigm (timm identifiers).
MODELS = [
    # Supervised baseline
    "vit_base_patch16_224.augreg_in21k_ft_in1k",      # ViT-B/16-Supervised
    # Self-Distillation — DINOv1
    "vit_base_patch16_224.dino",                       # DINOv1-B/16
    "vit_base_patch8_224.dino",                        # DINOv1-B/8
    # Self-Distillation — DINOv2 (+ registers)
    "vit_small_patch14_dinov2.lvd142m",               # DINOv2-S/14
    "vit_base_patch14_dinov2.lvd142m",                # DINOv2-B/14
    "vit_base_patch14_reg4_dinov2.lvd142m",           # DINOv2-B/14+Reg4
    # Masked Autoencoding — MAE
    "vit_base_patch16_224.mae",                        # MAE-ViT-B/16
    "vit_large_patch16_224.mae",                       # MAE-ViT-L/16
    # Masked Token Prediction — BEiT3
    "beit3_base_patch16_224.in22k_ft_in1k",           # BEiT3-B/16
    "beit3_large_patch16_224.in22k_ft_in1k",          # BEiT3-L/16
    # Language-Image Contrastive — CLIP family
    "vit_base_patch16_clip_224.openai_ft_in12k_in1k", # CLIP-ViT-B/16-OpenAI
    "vit_base_patch16_clip_224.laion2b_ft_in12k_in1k",# CLIP-ViT-B/16-LAION2B
    "vit_base_patch16_clip_224.metaclip_2pt5b",       # MetaCLIP-ViT-B/16
    # MIM + Contrastive Hybrid — EVA-02
    "eva02_base_patch14_224.mim_in22k",               # EVA02-B/14-MIM
]


def main():
    try:
        import timm
    except ImportError:
        sys.exit("timm is not installed. Run:  pip install 'timm>=0.9.0'")

    # timm / huggingface_hub read HF_TOKEN automatically; just ensure both names set.
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        print("Using Hugging Face token from .env")
    else:
        print("No HF token found in environment; downloading anonymously")

    print(f"timm version: {timm.__version__}")
    print(f"Downloading {len(MODELS)} foundation models from the Hugging Face Hub.\n")

    ok, failed = [], []
    for i, name in enumerate(MODELS, 1):
        print(f"[{i:>2}/{len(MODELS)}] {name} ...", end=" ", flush=True)
        t0 = time.time()
        try:
            model = timm.create_model(name, pretrained=True)
            n_params = sum(p.numel() for p in model.parameters()) / 1e6
            del model
            print(f"done ({n_params:.1f}M params, {time.time() - t0:.1f}s)")
            ok.append(name)
        except Exception as e:
            print(f"FAILED: {e}")
            failed.append((name, str(e)))

    print("\n" + "=" * 60)
    print(f"Downloaded : {len(ok)}/{len(MODELS)}")
    if failed:
        print(f"Failed     : {len(failed)}")
        for name, err in failed:
            print(f"  - {name}: {err}")
        # Non-zero exit so CI/automation notices, but caching of the rest stands.
        sys.exit(1)
    print("All foundation models cached successfully.")


if __name__ == "__main__":
    main()
