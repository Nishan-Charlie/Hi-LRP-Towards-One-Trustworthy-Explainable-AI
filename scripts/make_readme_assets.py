"""Build the web-optimized figure set the README embeds.

The paper figures under `figures/` are full-resolution (up to 5 MB each), which
makes the README slow to load on GitHub. This script downscales the selected
subset to a README-appropriate width and writes them to `docs/assets/`.

Source figures stay untouched: `docs/assets/` is derived, regenerate any time.

Run from the repo root:  python scripts/make_readme_assets.py
"""
from __future__ import annotations

import os

from PIL import Image

MAX_WIDTH = 1400
JPEG_QUALITY = 86
SRC_DIR = "figures"
OUT_DIR = os.path.join("docs", "assets")

# (source path relative to figures/, output basename). Heatmap panels are
# photographic and go to JPEG; line-art diagrams stay PNG so text stays crisp.
ASSETS = [
    ("bench/arch_method_comparison.png", "hero_architecture_comparison.jpg"),
    ("hilrp/hilrp_pipeline.png", "method_pipeline.png"),
    ("hilrp/stage_localization.png", "stage_localization.jpg"),
    ("bench/qualitative_gallery.png", "qualitative_gallery.jpg"),
    ("bench/cam_upsampling_artifact.png", "cam_upsampling_artifact.jpg"),
    ("hilrp/sanity_check.png", "sanity_check.jpg"),
    ("hilrp/rq2_maps.png", "rq2_pretraining_objectives.jpg"),
    ("hilrp/clip_multimodal.png", "clip_multimodal.jpg"),
]


def convert(src: str, dst: str) -> None:
    im = Image.open(src)
    if im.mode in ("RGBA", "P", "LA"):
        flat = Image.new("RGB", im.size, "white")
        flat.paste(im, mask=im.convert("RGBA").split()[-1])
        im = flat
    else:
        im = im.convert("RGB")

    if im.width > MAX_WIDTH:
        height = round(im.height * MAX_WIDTH / im.width)
        im = im.resize((MAX_WIDTH, height), Image.LANCZOS)

    if dst.endswith(".jpg"):
        im.save(dst, "JPEG", quality=JPEG_QUALITY, optimize=True, progressive=True)
    else:
        im.save(dst, "PNG", optimize=True)
    kb = os.path.getsize(dst) // 1024
    print(f"  {im.width:4d}x{im.height:<5d} {kb:5d} KB  {dst}")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"writing README assets to {OUT_DIR}/ (max width {MAX_WIDTH}px)")
    for rel_src, out_name in ASSETS:
        src = os.path.join(SRC_DIR, rel_src)
        if not os.path.exists(src):
            print(f"  MISSING, skipped: {src}")
            continue
        convert(src, os.path.join(OUT_DIR, out_name))


if __name__ == "__main__":
    main()
