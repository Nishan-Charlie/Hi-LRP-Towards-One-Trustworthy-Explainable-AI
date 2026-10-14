# HiLRP

### Toward One Trustworthy Explanation for Vision Transformer: Conservation-Valid Attribution via Attention Primitives

Nishankar Sathiyamohan, Pubudu Sanjeewani, Asanka Perera, Selvarajah Thuseethan

<p align="left">
  <a href="https://nishan-charlie.github.io/Hi-LRP-Towards-One-Trustworthy-Explainable-AI/"><img alt="project page" src="https://img.shields.io/badge/project-page-c0303a"></a>
  <a href="https://arxiv.org/abs/2609.01282"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2609.01282-b31b1b"></a>
  <a href="https://github.com/Nishan-Charlie/Hi-LRP-Towards-One-Trustworthy-Explainable-AI/actions/workflows/conservation-tests.yml"><img alt="tests" src="https://github.com/Nishan-Charlie/Hi-LRP-Towards-One-Trustworthy-Explainable-AI/actions/workflows/conservation-tests.yml/badge.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-blue">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c">
  <img alt="license" src="https://img.shields.io/badge/license-MIT-green">
</p>

**[Project page](https://nishan-charlie.github.io/Hi-LRP-Towards-One-Trustworthy-Explainable-AI/)** · Official PyTorch implementation of **HiLRP**, a relevance-propagation method that explains
hierarchical and hybrid Vision Transformers in a single backward pass, and of the benchmark
used to evaluate it.

<p align="center">
  <img src="https://raw.githubusercontent.com/Nishan-Charlie/Hi-LRP-Towards-One-Trustworthy-Explainable-AI/main/docs/assets/hero_architecture_comparison.jpg" width="100%"
       alt="HiLRP compared with Grad-CAM, Grad-CAM++, Integrated Gradients and SmoothGrad on Swin-B, PVT-v2-b2, EfficientViT-B2 and MobileViT-v2">
</p>

```python
import hilrp

res = hilrp.attribute(model, x)     # any supported timm backbone, one backward pass
res["pixel_map"]                    # [H, W] signed relevance at the input pixels
```

## Abstract

Vision Transformer design has fragmented. Each new backbone recombines convolutional stems,
windowed, linear or multi-axis attention, patch merging and spatial reduction, and each
recombination breaks an assumption that some attribution method relies on. HiLRP starts from
the observation that the attention and resolution-reduction operators of current ViTs compose
four operation classes, and that each class admits one relevance rule that conserves. A
backbone is then covered by construction instead of by a per-architecture derivation, and the
same backward pass yields a signed relevance map at the input pixels and at every resolution
stage. On the backbones where Grad-CAM degrades, HiLRP localizes substantially better
(EfficientViT-B2: 0.936 against 0.518 Pointing, n = 1000), and the gap replicates on a second
dataset.

## Highlights

- **One call, six architecture families.** Swin, PVT-v2, MaxViT, MobileViT-v2, EfficientViT
  and plain ViT / DeiT, dispatched from the timm model itself.
- **Maps at every resolution from one pass.** Pixels and every token stage, which
  output-perturbation methods cannot produce.
- **Beyond class logits.** The same pass explains a label-free self-supervised scalar or a
  CLIP image-text similarity, with no new rule.
- **Tested rules.** Conservation of the rules is proved and enforced in CI to machine
  precision; the production ports have their own suite. What that does and does not
  guarantee on trained models is stated under [Limitations](#limitations).

## Installation

HiLRP is a pip package. It needs Python 3.11.11 or newer.

```bash
pip install "hilrp @ git+https://github.com/Nishan-Charlie/Hi-LRP-Towards-One-Trustworthy-Explainable-AI.git"
```

That installs `hilrp` and everything it needs to run, including the attribution backend
(`lxt`, `zennit`). To work on the code or reproduce the paper, install from a clone instead:

```bash
git clone https://github.com/Nishan-Charlie/Hi-LRP-Towards-One-Trustworthy-Explainable-AI.git
cd Hi-LRP-Towards-One-Trustworthy-Explainable-AI

pip install -e .                 # the library, editable
pip install -e ".[all]"          # plus what the benchmark harness and the tests need
```

Version notes that matter:

- **`zennit` must be 1.0 or newer**, which is why Python 3.11.11+ is required. Under 0.5.x
  the BatchNorm merge is not forward-exact and the MobileViT-v2 and EfficientViT maps change,
  so `hilrp` refuses to run on it.
- **`lxt` is pinned to 2.1.** The modules HiLRP builds on do not exist in 0.6.x.
- **`timm` 1.0.16 or newer.** Relevance maps are bit-identical on 1.0.16 and 1.0.30. The
  published benchmark numbers were produced on 1.0.16.
- On Windows, set `PYTHONUTF8=1` before installing, or the metadata step of `lxt` fails with
  a `UnicodeDecodeError`.

## Quick start

```bash
# from a clone of this repository
python examples/quickstart.py                                    # Swin-T on images/cat.jpg
python examples/quickstart.py --model pvt_v2_b2 --image images/tiger.jpg
```

This prints the explained class and the conservation trace, and writes the figure shown in
the tutorial below.

## Tutorial

### 1. Load a model and an image

HiLRP works on unmodified timm models. Preprocess the image the way that model expects;
timm will tell you how.

```python
import timm, torch
from PIL import Image

model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True).eval()

cfg = timm.data.resolve_data_config({}, model=model)
transform = timm.data.create_transform(**cfg)
image = Image.open("images/cat.jpg").convert("RGB")
x = transform(image).unsqueeze(0)               # [1, 3, 224, 224]
```

### 2. Explain a prediction

```python
import hilrp

res = hilrp.attribute(model, x)                 # explains the predicted class
res = hilrp.attribute(model, x, target=281)     # or any class index you choose

res["target"], res["logit"]    # 281 (tabby cat), 8.06
res["pixel_map"]               # torch.Tensor [224, 224], signed
```

One forward and one backward pass, no sampling, so the result is deterministic. Positive
values are evidence for the class and negative values are evidence against it.

### 3. Visualize the map

Relevance is heavy-tailed, so a plain min-max colormap washes the map out. Always render
through the helper, which clips at the 99th percentile of the absolute value.

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
hilrp.show_relevance(ax, res["pixel_map"])      # red supports the class, blue opposes it
fig.savefig("cat_hilrp.png", dpi=160)

# or get the clipped array yourself
heatmap, vmax = hilrp.normalize_for_display(res["pixel_map"])
```

<p align="center">
  <img src="https://raw.githubusercontent.com/Nishan-Charlie/Hi-LRP-Towards-One-Trustworthy-Explainable-AI/main/docs/assets/quickstart_swin_cat.jpg" width="100%"
       alt="Input image, HiLRP relevance at the pixels, and HiLRP relevance at the 56, 28, 14 and 7 token stages of Swin-T">
</p>

### 4. One map per resolution stage

Relevance is tracked at every interface of the network, so on Swin the same pass also returns
a map at each stage. No extra computation is involved.

```python
for capture, stage_map in res["stage_maps"]:
    print(capture, tuple(stage_map.shape))

# tokens_56 (56, 56)
# stage1_in_28 (28, 28)
# stage2_in_14 (14, 14)
# stage3_in_7 (7, 7)
# final_7 (7, 7)
```

Fine stages show texture and parts, coarse stages show the object.

### 5. Read the conservation trace

`stage_sums` reports how much of the logit is accounted for at each capture, from the input
tokens to the head. A value of 1.00 means the relevance at that capture sums exactly to the
logit.

```python
for capture, ratio in res["stage_sums"]:
    print(f"{capture:14s} {ratio:+.2f}")

# tokens_56      +1.07
# stage1_in_28   +1.03
# stage2_in_14   +1.10
# stage3_in_7    +1.26
# final_7        +1.00

res["pixel_map"].sum().item() / res["logit"]    # 0.77 at the pixels
```

Use it as a diagnostic. How close it stays to 1 depends on the backbone; see
[Limitations](#limitations).

### 6. Use any supported backbone

The call does not change. `hilrp.attribute` reads the architecture from the timm model and
picks the right rules.

```python
for name in ["pvt_v2_b2", "maxvit_small_tf_224", "mobilevitv2_100", "efficientvit_b2", "deit_base_patch16_224"]:
    model = timm.create_model(name, pretrained=True).eval()
    transform = timm.data.create_transform(**timm.data.resolve_data_config({}, model=model))
    res = hilrp.attribute(model, transform(image).unsqueeze(0))
```

| Family             | Attention                           | Checked on                                  | Conservation on trained weights |
| ------------------ | ----------------------------------- | ------------------------------------------- | ------------------------------- |
| Swin               | windowed and shifted, patch merging | `swin_tiny`, `swin_base`                    | exact up to bias absorption     |
| PVT-v2             | spatial reduction                   | `pvt_v2_b0`, `pvt_v2_b2`                    | exact up to bias absorption     |
| MaxViT             | multi-axis                          | `maxvit_tiny_tf_224`, `maxvit_small_tf_224` | measured                        |
| MobileViT-v2       | separable, conv hybrid              | `mobilevitv2_050`, `mobilevitv2_100`        | measured                        |
| EfficientViT (MIT) | linear                              | `efficientvit_b0`, `b1`, `b2`               | measured                        |
| ViT, DeiT          | global self-attention               | `vit_tiny`, `vit_base`, `deit_base`         | measured                        |

`hilrp.supported_architectures()` returns the same list in code. CNNs are out of scope.

### 7. Options

```python
hilrp.attribute(model, x, gamma=0.25)             # gamma rule on Linear and Conv2d (default 0.25)
hilrp.attribute(model, x, conv_gamma=0.05)        # a separate gamma for convolutions
hilrp.attribute(model, x, attn_mode="attnlrp")    # Swin and ViT only: AttnLRP rules inside attention
```

The default `attn_mode="cp"` detaches the attention matrix and is the conserving choice.
`"attnlrp"` is kept as a labeled diagnostic.

### 8. Compare against other methods safely

HiLRP installs its rules as class-level patches on `nn.LayerNorm`, `nn.GELU` and the timm
attention classes. The patches are process-global and are never undone. Forward passes are
numerically unchanged, but any _other_ gradient-based method computed afterwards runs through
the patched rules and is no longer the method you meant to compare against.

```python
baseline = my_grad_cam(model, x)        # 1. every other gradient-based method FIRST
res = hilrp.attribute(model, x)         # 2. the process is patched from here on
hilrp.patched_ports()                   # ['hilrp.swin_lxt']
```

Or use a separate process. The benchmark runner enforces this for you: it rejects a config
that mixes `hilrp` with any other method.

### 9. Explain something other than a class logit

The backward pass starts from any differentiable scalar. For a self-supervised model with no
classifier head, explain its similarity to another view of the same image:

```python
from hilrp.vit_lxt import attribute_ssl_similarity

model = timm.create_model("vit_base_patch16_224.dino", pretrained=True).eval()
res = attribute_ssl_similarity(model, x, x_ref=torch.flip(x, dims=[3]))
res["similarity"], res["pixel_map"]     # 0.996, [224, 224]
```

For CLIP image-text similarity see `hilrp.clip_lxt.attribute_clip_image_text` and the worked
example in [`scripts/hilrp/clip_demo.py`](https://github.com/Nishan-Charlie/Hi-LRP-Towards-One-Trustworthy-Explainable-AI/blob/main/scripts/hilrp/clip_demo.py).

## Method

Rather than deriving a new rule per architecture, HiLRP decomposes attention and resolution
reduction into four operation classes. Each has one conserving rule, so any arrangement of
them is covered.

<p align="center">
  <img src="https://raw.githubusercontent.com/Nishan-Charlie/Hi-LRP-Towards-One-Trustworthy-Explainable-AI/main/docs/assets/method_pipeline.png" width="100%"
       alt="HiLRP pipeline: one backward pass over a hierarchical or hybrid ViT, built from four conserving primitives">
</p>

| #   | Primitive          | Instances                                        | Rule                                          |
| --- | ------------------ | ------------------------------------------------ | --------------------------------------------- |
| P1  | Linear map         | Q/K/V/out projections, MLP, conv, pooling        | epsilon / gamma rule                          |
| P2  | Bilinear mix       | QKᵀ, A·V, linear-attention products              | uniform split, or detach one factor as a gate |
| P3  | Normalize / gate   | softmax, LayerNorm, linear-attention denominator | detach the denominator, identity rule         |
| P4  | Reindex / sparsity | heads, windows, cyclic shifts, patch merging     | permutation, exact                            |

The lemmas and the full attention-coverage table are in [`docs/METHOD.md`](https://github.com/Nishan-Charlie/Hi-LRP-Towards-One-Trustworthy-Explainable-AI/blob/main/docs/METHOD.md).

## Results

Paired per-image Pointing Game, same images for both methods, 95% bootstrap confidence
intervals, n = 1000.

| Backbone        | Dataset    | HiLRP                    | Grad-CAM             | Paired McNemar |
| --------------- | ---------- | ------------------------ | -------------------- | -------------- |
| EfficientViT-B2 | ImageNet-S | **0.936** [0.921, 0.950] | 0.518 [0.486, 0.549] | p = 2.1e-104   |
| PVT-v2-b2       | ImageNet-S | **0.958** [0.945, 0.970] | 0.846 [0.823, 0.868] | p = 1.3e-24    |
| EfficientViT-B2 | VOC 2007   | **0.820** [0.795, 0.843] | 0.582 [0.551, 0.612] | p = 3.3e-41    |
| PVT-v2-b2       | VOC 2007   | **0.865** [0.844, 0.886] | 0.621 [0.591, 0.650] | p = 5.7e-48    |

HiLRP is not best everywhere: Grad-CAM saturates the Pointing Game on Swin and MobileViT-v2.
The cross-architecture benchmark, agreement with Shapley values, conservation measurements,
sanity checks and the provenance of every number are in
[`docs/RESULTS.md`](https://github.com/Nishan-Charlie/Hi-LRP-Towards-One-Trustworthy-Explainable-AI/blob/main/docs/RESULTS.md).

## Reproducing the paper

```bash
pip install -e ".[all]"

# 1. Build the dataset caches once
python scripts/data/cache_imagenets.py --num 1000 --split validation
python scripts/data/cache_voc.py --num 1000

# 2. Headline comparison (one method per process)
python scripts/bench/scaled_eval.py gradcam efficientvit_b2 1000
python scripts/bench/scaled_eval.py hilrp   efficientvit_b2 1000
python scripts/bench/scaled_eval.py stats   efficientvit_b2

# 3. Benchmark grid: baselines and HiLRP in separate runs
python -m xai_bench.runner --config configs/full_benchmark_100.yaml
python -m xai_bench.runner --config configs/hilrp_grid.yaml
```

The script behind each table and figure is listed in
[`docs/RESULTS.md`](https://github.com/Nishan-Charlie/Hi-LRP-Towards-One-Trustworthy-Explainable-AI/blob/main/docs/RESULTS.md#reproducing-results).

## Repository layout

```
hilrp/            the method, as an installable package
├── api.py          hilrp.attribute and the per-family dispatch
├── *_lxt.py        one port per architecture family
├── hierarchical.py, attention_primitives.py, rules.py     the rules
├── viz.py          display normalization
└── tests/, tests_backend/
xai_bench/        the benchmark harness (not in the pip package; run it from a clone)
configs/          benchmark run configs
scripts/          experiments, tables and figures for the paper
examples/         quickstart.py, the script behind the tutorial
docs/             METHOD.md, RESULTS.md, figures used on this page
```

## Tests

```bash
python -m pytest hilrp/tests/ -v            # 24 passed: the rules, float64, no backend needed
python -m pytest xai_bench/tests/ -v        # benchmark harness: process isolation, seeding
python -m pytest hilrp/tests_backend/ -v    # 26 passed: the ports on real timm architectures
```

Run the last one as its own command. Importing a port patches classes for the life of the
interpreter. With the library installed from pip and no clone, the same suites run as
`python -m pytest --pyargs hilrp.tests` and `python -m pytest --pyargs hilrp.tests_backend`.

## Limitations

- **Conservation is proved for the rules, not guaranteed for every trained model.** With
  bias-free weights Swin and PVT-v2 conserve at every capture to float32 precision, so on
  those two the deviation on trained weights is what the biases absorb. On MaxViT,
  MobileViT-v2, EfficientViT and plain ViTs the relevance sum drifts with depth and is a
  measured quantity. No backbone conserves all the way to the pixels.
- **The patches are process-global.** See tutorial step 8.
- **Not a universal win.** Grad-CAM matches or beats HiLRP on the Pointing Game for Swin and
  MobileViT-v2, and the reason Grad-CAM fails on EfficientViT is measured but not explained.
- **Scope.** CNNs are not supported. Cross-attention decoders and deformable attention are
  covered by the rules but not yet exercised on trained weights.

Numbers for each point are in [`docs/RESULTS.md`](https://github.com/Nishan-Charlie/Hi-LRP-Towards-One-Trustworthy-Explainable-AI/blob/main/docs/RESULTS.md#limitations).

## Citation

```bibtex
@article{nishankar2026hilrp,
  title={HiLRP: Toward One Trustworthy Explanation for Vision Transformer: Conservation-Valid Attribution via Attention Primitives},
  author={Nishankar, Sathiyamohan and Sanjeewani, Pubudu and Perera, Asanka and Thuseethan, Selvarajah},
  journal={arXiv preprint arXiv:2609.01282},
  year={2026}
}
```

## License and acknowledgements

MIT, see [`LICENSE`](https://github.com/Nishan-Charlie/Hi-LRP-Towards-One-Trustworthy-Explainable-AI/blob/main/LICENSE). Model weights come from `timm` and carry their own licenses.
ImageNet-S and PASCAL VOC 2007 are subject to their respective terms.

Built on [LXT / AttnLRP](https://github.com/rachtibat/LRP-eXplains-Transformers),
[zennit](https://github.com/chr5tphr/zennit),
[timm](https://github.com/huggingface/pytorch-image-models),
[Quantus](https://github.com/understandable-machine-intelligence-lab/Quantus) and
[Captum](https://github.com/pytorch/captum). No model was fine-tuned: every result uses
frozen, publicly released checkpoints.
