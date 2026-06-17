#!/usr/bin/env python
# ============================================================
# download_datasets.py
# Download scripts for the datasets used in the benchmark.
# 
# Usage:
#   python download_datasets.py --funnybirds --imagenet-s
# ============================================================

import os
import sys
import argparse
import urllib.request
import zipfile
import tarfile
from pathlib import Path

DATA_DIR = Path("data")

def download_file(url: str, dest_path: Path):
    print(f"Downloading {url} to {dest_path}...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    def progress_bar(count, block_size, total_size):
        if total_size > 0:
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(f"\rDownloading... {min(percent, 100)}%")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=progress_bar)
        print("\nDownload complete.")
    except Exception as e:
        print(f"\nFailed to download {url}: {e}")
        sys.exit(1)

def extract_zip(zip_path: Path, extract_to: Path):
    print(f"Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete.")

def download_funnybirds():
    """Download and extract the FunnyBirds dataset (synthetic)."""
    funnybirds_dir = DATA_DIR / "FunnyBirds"
    if funnybirds_dir.exists() and any(funnybirds_dir.iterdir()):
        print("FunnyBirds dataset already seems to exist at data/FunnyBirds. Skipping.")
        return
    
    # The official URL from the authors
    url = "http://download.visinf.tu-darmstadt.de/data/funnybirds/FunnyBirds.zip"
    zip_path = DATA_DIR / "FunnyBirds.zip"
    
    download_file(url, zip_path)
    extract_zip(zip_path, funnybirds_dir)
    
    # Clean up the zip file to save space
    print("Cleaning up zip file...")
    zip_path.unlink()
    print("FunnyBirds setup complete.")

def download_imagenets():
    """
    Provide instructions or download ImageNet-S.
    Because ImageNet-S requires the original ImageNet-1K images, 
    we provide the HuggingFace datasets library approach as a fallback,
    or instructions for manual preparation.
    """
    imagenets_dir = DATA_DIR / "ImageNetS"
    
    print("\n--- ImageNet-S ---")
    print("ImageNet-S requires the original ImageNet-1K images.")
    print("To prepare it properly from scratch, please follow the official repository:")
    print("  https://github.com/LUSSeg/ImageNet-S")
    print("\nAlternatively, you can load a replica using the Hugging Face `datasets` library:")
    print("  from datasets import load_dataset")
    print("  dataset = load_dataset('braceletboy/imagenet-s')")
    
    print(f"\nCreating placeholder directory at {imagenets_dir}...")
    imagenets_dir.mkdir(parents=True, exist_ok=True)
    
    readme_path = imagenets_dir / "README.md"
    if not readme_path.exists():
        with open(readme_path, "w") as f:
            f.write("# ImageNet-S Dataset\n")
            f.write("Please place your prepared ImageNet-S dataset here, or use HF `datasets`.\n")
            f.write("See https://github.com/LUSSeg/ImageNet-S for data preparation instructions.\n")
    print("ImageNet-S placeholder setup complete.")

def main():
    parser = argparse.ArgumentParser(description="Download datasets for the XAI Benchmark.")
    parser.add_argument("--funnybirds", action="store_true", help="Download FunnyBirds dataset")
    parser.add_argument("--imagenet-s", action="store_true", help="Setup ImageNet-S dataset")
    parser.add_argument("--all", action="store_true", help="Download all available datasets")
    
    args = parser.parse_args()
    
    if not any([args.funnybirds, args.imagenet_s, args.all]):
        parser.print_help()
        print("\nPlease specify which dataset(s) to download.")
        sys.exit(1)
        
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.funnybirds or args.all:
        download_funnybirds()
        
    if args.imagenet_s or args.all:
        download_imagenets()

if __name__ == "__main__":
    main()
