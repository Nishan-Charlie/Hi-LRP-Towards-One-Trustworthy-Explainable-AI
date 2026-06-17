#!/usr/bin/env python
# ============================================================
# download_models.py
# Pre-download / cache all pretrained ViT models used by
# vit_xai_benchmark.ipynb from the Hugging Face Hub.
#
# timm.create_model(..., pretrained=True) fetches the weights
# from HF Hub and stores them in the local HF cache, so the
# notebook can then run without re-downloading.
#
# Usage:
#   python download_models.py
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

# ── The 10 models referenced in the benchmark (timm identifiers) ──────────────
MODELS = [
    "vit_base_patch16_224",          # ViT-B/16
    "vit_small_patch16_224",         # ViT-S/16
    "deit_base_patch16_224",         # DeiT-B
    "deit_small_patch16_224",        # DeiT-S
    "swin_tiny_patch4_window7_224",  # Swin-T
    "swin_base_patch4_window7_224",  # Swin-B
    "beit_base_patch16_224",         # BEiT-B
    "convit_base",                   # ConViT-B
    "cait_s24_224",                  # CaiT-S24
    "maxvit_tiny_tf_224",            # MaxViT-T
]


def main():
    try:
        import timm
    except ImportError:
        sys.exit("timm is not installed. Run:  pip install timm")

    # timm / huggingface_hub automatically read HF_TOKEN (or HUGGING_FACE_HUB_TOKEN)
    # from the environment — no need to call login(), which can hang on Windows
    # while touching the credential store. Make sure both names are set.
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        print("Using Hugging Face token from .env")
    else:
        print("No HF token found in environment; downloading anonymously")

    print(f"timm version: {timm.__version__}")
    print(f"Downloading {len(MODELS)} pretrained models from the Hugging Face Hub.\n")

    ok, failed = [], []
    for i, name in enumerate(MODELS, 1):
        print(f"[{i:>2}/{len(MODELS)}] {name} ...", end=" ", flush=True)
        t0 = time.time()
        try:
            # pretrained=True triggers the weight download into the HF cache.
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
        sys.exit(1)
    print("All models cached successfully.")


if __name__ == "__main__":
    main()
