"""Multi-modal HiLRP on CLIP (ViT-B/16), text-conditioned attribution on clean
single-object images (koala, panda, owl). Qualitative demonstration: a caption
matching the image concentrates positive relevance on the object; an unrelated
caption does not. Regenerates figures/hilrp/clip_multimodal.png.

Maps are cached to results/figure_cache/clip_multimodal.npz on the first run:

  python scripts/hilrp/clip_custom_images.py            compute + plot
  python scripts/hilrp/clip_custom_images.py --replot   plot from the cache
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))                       # <repo>/scripts

import numpy as np
from PIL import Image

from figstyle import apply_paper_style, lxt_transformers_shim, save, COL_W

lxt_transformers_shim()
import torch
import open_clip

from hilrp.viz import normalize_for_display

apply_paper_style()
import matplotlib.pyplot as plt

OUT = "figures/hilrp"
CACHE = os.path.join("results", "figure_cache", "clip_multimodal.npz")
os.makedirs(OUT, exist_ok=True)

# (image file, matching caption, unrelated caption)
ITEMS = [
    ("images/koala.jpg", "a photo of a koala", "a photo of a car"),
    ("images/panda.jpg", "a photo of a giant panda", "a photo of a car"),
    ("images/owl.jpg",   "a photo of an owl", "a photo of a car"),
]


def compute():
    from hilrp.clip_lxt import attribute_clip_image_text, ensure_patched
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    model.eval()
    tok = open_clip.get_tokenizer("ViT-B-16")
    ensure_patched()

    rows, store = [], {}
    for i, (path, cap_ok, cap_no) in enumerate(ITEMS):
        pil = Image.open(path).convert("RGB")
        x = preprocess(pil).unsqueeze(0)                      # CLIP-normalized [1,3,224,224]
        disp = np.array(pil.resize((224, 224))) / 255.0       # for display
        with torch.no_grad():
            te = model.encode_text(tok([cap_ok, cap_no]))
            te = te / te.norm(dim=-1, keepdim=True)
        r_ok = attribute_clip_image_text(model, x, te[0].detach())
        r_no = attribute_clip_image_text(model, x, te[1].detach())
        m_ok = np.asarray(r_ok["pixel_map"])
        m_no = np.asarray(r_no["pixel_map"])
        rows.append((disp, cap_ok, m_ok, float(r_ok["similarity"]),
                     cap_no, m_no, float(r_no["similarity"])))
        store.update({f"img{i}": disp, f"ok{i}": m_ok, f"no{i}": m_no,
                      f"sim{i}": np.array([r_ok["similarity"], r_no["similarity"]],
                                          dtype=np.float64)})
        print(f"{path}: sim(ok)={r_ok['similarity']:.2f} sim(no)={r_no['similarity']:.2f}",
              flush=True)

    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, **store)
    print(f"cached maps -> {CACHE}")
    return rows


def from_cache():
    z = np.load(CACHE)
    rows = []
    for i, (_, cap_ok, cap_no) in enumerate(ITEMS):
        sim = z[f"sim{i}"]
        rows.append((z[f"img{i}"], cap_ok, z[f"ok{i}"], float(sim[0]),
                     cap_no, z[f"no{i}"], float(sim[1])))
    return rows


def make_figure(rows):
    n = len(rows)
    # Column width: this goes in at \linewidth, so author it at 3.5 in and the
    # type stays the size it is written at.
    panel = COL_W / 3
    fig, ax = plt.subplots(n, 3, figsize=(COL_W, (panel + 0.30) * n + 0.30))
    headers = ("Input", "Matching Caption", "Unrelated Caption")
    for r, (disp, cap_ok, m_ok, sim_ok, cap_no, m_no, sim_no) in enumerate(rows):
        ax[r, 0].imshow(disp); ax[r, 0].axis("off")
        for c, (m, cap, sim) in enumerate([(m_ok, cap_ok, sim_ok),
                                           (m_no, cap_no, sim_no)], start=1):
            v_map, v = normalize_for_display(m, percentile=99, smooth_sigma=1.0)
            ax[r, c].imshow(v_map, cmap="bwr", vmin=-v, vmax=v)
            ax[r, c].axis("off")
            # caption goes UNDER the panel as free text: at 1.2 in wide a title
            # collides with the row above, and axis("off") suppresses xlabels
            ax[r, c].text(0.5, -0.04, f'"{cap}"\nsim = {sim:.2f}', fontsize=6,
                          transform=ax[r, c].transAxes, ha="center", va="top",
                          linespacing=1.35)
        if r == 0:
            for c in range(3):
                ax[r, c].set_title(headers[c], fontsize=7)
    fig.tight_layout(pad=0.2, h_pad=1.6)
    fig.suptitle("Multi-Modal HiLRP: Text-Conditioned Attribution on CLIP", y=1.02)
    save(fig, os.path.join(OUT, "clip_multimodal"), also_pdf=False)


if __name__ == "__main__":
    if "--replot" in sys.argv and os.path.exists(CACHE):
        make_figure(from_cache())
    else:
        make_figure(compute())
