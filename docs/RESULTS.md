# HiLRP: full results, protocol and limitations

The [README](../README.md) carries the headline result only. Everything else that
backs the paper, and everything added to the repository since, is here.

- [Protocol](#protocol)
- [Results](#results)
- [Reproducing each result](#reproducing-results)
- [Limitations](#limitations)

## Protocol

**Datasets.** ImageNet-S (cached, bounding boxes plus pixel-perfect segmentation masks) as
the primary evaluation set, and PASCAL VOC 2007 test (detection boxes, mean 3 boxes per
image) as an independent second dataset with a different annotation type.

**Baselines.** 13 published methods across four families: gradient (Saliency, Integrated
Gradients, Input×Gradient, SmoothGrad, VarGrad, GradientSHAP), CAM (Grad-CAM, Grad-CAM++),
attention-native (Attention Rollout, AttnLRP), perturbation (Occlusion, RISE, LIME). Plus
sampled permutation Shapley as an axiomatic reference, and a naive-AttnLRP extension as the
lineage control.

**Metrics.** Five axes: faithfulness (Faithfulness Correlation, Faithfulness Estimate, APF),
localization (bounding-box Pointing Game, Energy-Based Pointing Game on dense masks),
robustness (Max-Sensitivity), complexity (Sparseness), and computational cost. Conservation
(`sum(R)/logit`) is measured separately as the method-side discriminator.

**Experimental setup.** All backbones are frozen, publicly released timm checkpoints. No
fine-tuning anywhere. Comparisons are paired on the same images, with 95% bootstrap
confidence intervals (2000 resamples) and paired McNemar or Wilcoxon tests. The
community-standard stack is used unchanged (Quantus, Captum, `pytorch-grad-cam`) so that
reported failures are properties of standard practice rather than of a bespoke setup. Two
tiers: a large tier at n = 1000 for cheap metrics, and a deep-validation tier at n = 50 to
100 for Shapley agreement and conservation.

> **HiLRP must run in its own process.** Its backend applies class-level forward patches to
> `nn.LayerNorm`, `nn.GELU`, and timm attention classes. Forward numerics are unchanged, but
> every other method's gradients in the same process are altered by them. Never put `hilrp` in
> a `methods:` list alongside baselines. The same lesson produced the two-pass Shapley
> protocol: contaminated baselines had artificially depressed scores (saliency 0.08 vs 0.21
> honest).
>
> This is enforced, not just documented. A config that lists `hilrp` or `attnlrp` next to any
> other method is rejected when it is loaded, and a baseline run refuses to start in an
> interpreter where a port has already patched
> ([`xai_bench/isolation.py`](../xai_bench/isolation.py)). One `hilrp` run may still sweep
> several backbones: each port's map is identical with and without the other ports' patches
> live. Scripts that call the ports directly, outside the runner, are not guarded and keep
> their own two-pass or separate-process protocol.

## Results

### Where each number comes from

"Manuscript" is the paper in [Citation](../README.md#citation), currently under review. Rows marked *not
in the manuscript* were added to this repository afterwards and have not been reviewed with
it.

| Result | n, dataset | Produced by | In the manuscript |
|---|---|---|---|
| Main result, HiLRP vs Grad-CAM | 1000, ImageNet-S and VOC 2007 | `scripts/bench/scaled_eval.py` | yes |
| Conservation is the discriminator | 100, ImageNet-S | `scripts/hilrp/lineage_comparison.py`, `lineage_naive_legs.py` | yes |
| Agreement with Shapley values | 50, ImageNet-S | `scripts/hilrp/shapley_pilot.py` | yes |
| Cross-architecture baseline transfer | 100, ImageNet-S | `xai_bench.runner`, baseline configs | 11 of 13 rows: see below |
| HiLRP Pointing per backbone | 50, ImageNet-S | `xai_bench.runner`, `configs/hilrp_grid*.yaml` | a different run: see below |
| Verified numerics, reference rules | toy model | `hilrp/tests/` | as a claim, not as this table |
| Grad-CAM target-layer check | 1000, ImageNet-S and VOC 2007 | `scripts/bench/gradcam_target_layers.py` | not in the manuscript |
| Conservation per port (input, head, pixels) | 20, ImageNet-S | `scripts/hilrp/conservation_grid.py` | not in the manuscript |
| Verified numerics, production ports | random weights | `hilrp/tests_backend/` | not in the manuscript |

Three things to know when comparing against the manuscript:

- **HiLRP Pointing per backbone differs between the two.** The table below is the n = 50
  metric grid. The manuscript's benchmark table reports a separate HiLRP run, with
  different values on some backbones (ViT-B 0.70 against 0.78 here, MobileViT-v2 0.86 against
  0.78). Quote one source, not a mix.
- **Two rows of the cross-architecture table differ; the other eleven are identical.**
  GradientSHAP is from a different run on every transformer backbone (EffViT-B1 0.86 here
  against 0.64). AttnLRP on ViT-B is 0.83 here against 0.78, and the manuscript also fills the
  hierarchical cells with the naive AttnLRP extension, which this table leaves undefined.
- **The runner-produced rows predate seed handling.** `seeds` was accepted by the config but
  not applied, so in those rows the stochastic methods (SmoothGrad, VarGrad, GradientSHAP,
  RISE, LIME) and stochastic metrics (Faithfulness Correlation, Max-Sensitivity) are single
  unseeded draws. Deterministic methods scored with the Pointing Game are unaffected, and the
  Shapley sampler has always been seeded.

The three rows added in this revision were produced with `timm` 1.0.16 and `lxt` 2.1.

### HiLRP Pointing per backbone

HiLRP ports live in `hilrp/*_lxt.py` and are dispatched by model name.
Pointing Game is on ImageNet-S, n = 50, from the HiLRP metric grid.

| Backbone | Attention type | HiLRP port | Pointing |
|---|---|---|---|
| ViT-B/16 | isotropic self-attention | `vit_lxt.py` | 0.78 |
| DeiT-B/16 | isotropic, distilled | `vit_lxt.py` | 0.84 |
| Swin-B | windowed and shifted, patch merging | `swin_lxt.py` | 0.96 |
| PVT-v2-b2 | spatial-reduction attention | `pvt_lxt.py` | 0.98 |
| MaxViT-S | multi-axis (block plus grid) | `maxvit_lxt.py` | 0.96 |
| MobileViT-v2 | separable, conv hybrid | `mobilevit_lxt.py` | 0.78 |
| EfficientViT-B1 | linear cross-covariance | `efficientvit_lxt.py` | 0.94 |
| EfficientViT-B2 | linear cross-covariance | `efficientvit_lxt.py` | 0.96 |
| CLIP ViT-B/16 | cross-modal image-text | `clip_lxt.py` | image-text scalar |
| CrossViT | dual-branch cross-attention | `crossvit_lxt.py` | experimental |

Mean over the eight classification backbones: 0.90. ResNet-50 and ConvNeXt-B are carried as
CNN baselines only; HiLRP targets the ViT family, so it is not run on them.

### Main result: localization where CAM's assumption breaks

Paired per-image Pointing Game, same images for every method, 95% bootstrap CIs.

| Backbone | Attention type | HiLRP | Grad-CAM | n | Paired McNemar |
|---|---|---|---|---|---|
| EfficientViT-B2 | linear cross-covariance | **0.936** [0.921, 0.950] | 0.518 [0.486, 0.549] | 1000 | p = 2.1e-104 |
| PVT-v2-b2 | spatial-reduction | **0.958** [0.945, 0.970] | 0.846 [0.823, 0.868] | 1000 | p = 1.3e-24 |
| EfficientViT-B2 (VOC 2007) | linear cross-covariance | **0.820** [0.795, 0.843] | 0.582 [0.551, 0.612] | 1000 | p = 3.3e-41 |
| PVT-v2-b2 (VOC 2007) | spatial-reduction | **0.865** [0.844, 0.886] | 0.621 [0.591, 0.650] | 1000 | p = 5.7e-48 |

The gap replicates on a second dataset with a different annotation type, so it is not an
ImageNet-S artifact.

### Grad-CAM on EfficientViT: a target-layer check

An earlier version of the README said EfficientViT's linear attention "removes the terminal
spatial feature map that CAM depends on". That is not what the code does. `forward_features`
returns a real 7×7 map on this family and the benchmark's Grad-CAM targets it. What differs
from a CNN is what follows: a 1×1 expansion conv (`head.in_conv`), global pooling, then a
two-layer MLP, so the map Grad-CAM reads is not the map pooling consumes.

To separate "wrong layer" from "Grad-CAM fails here", the same Grad-CAM was re-run at every
candidate spatial layer on the same images. Pointing Game, n = 1000.

| Target layer | Map | EffViT-B2, ImageNet-S | EffViT-B2, VOC 2007 | EffViT-B1, ImageNet-S |
|---|---|---|---|---|
| `stages.0` | 56×56 | 0.714 | 0.571 | 0.684 |
| `stages.1` | 28×28 | 0.658 | 0.515 | 0.677 |
| `stages.2` | 14×14 | 0.510 | 0.346 | 0.678 |
| `stages.3`, the benchmark protocol (`forward_features`) | 7×7 | 0.518 | 0.582 | 0.647 |
| `head.in_conv`, the map pooling consumes | 7×7 | 0.271 | 0.315 | 0.327 |
| HiLRP, for reference | pixels | 0.936 | 0.820 | not run at this n |

No layer closes the gap. The best Grad-CAM layer on EfficientViT-B2 still trails HiLRP by 22
points on ImageNet-S (paired McNemar p = 1.1e-43), and the textbook choice, the last map
before pooling, is the worst on all three. The protocol number in the tables above is
therefore not an artifact of the target layer.

What this does not establish is why. Both EfficientViTs use linear attention and both have
the non-linear head, so this benchmark cannot attribute the failure to either one. Reproduce
with `python scripts/bench/gradcam_target_layers.py efficientvit_b2 1000 [voc]`; the
protocol-layer hits it writes match the cached run behind the main table image for image.

### Conservation is the actual discriminator

Pointing does not separate the top tier. Conservation does. `sum(R)/logit` at the
input-adjacent cut, 100 images.

| Backbone | Family | Naive Pointing | Naive conservation | HiLRP Pointing | HiLRP conservation |
|---|---|---|---|---|---|
| ViT-B/16 | flat ViT | 0.770 | +0.15 | 0.770 | +0.61 |
| Swin-B | hierarchical | 0.980 | **+3.08 ± 0.78**, inflated 3x | 0.950 | +0.95 |
| PVT-v2-b2 | spatial-reduction | 0.800 | **0.00, zeroed** | 0.980 | +0.39 |
| EfficientViT-B2 | linear attention | 0.930 | **0.00, zeroed** | 0.960 | traceable |
| MobileViT-v2 | conv hybrid | 0.950 | **0.00, zeroed** | 0.860 | +1.05 (head) |

A zero-sum map still has an argmax, so it can point convincingly while its region
contributions do not sum to the logit, violating the completeness property that is LRP's
entire justification. Full discussion: [`docs/lineage_table.md`](lineage_table.md).

Two HiLRP cells above are not input-adjacent sums: "traceable" (EfficientViT-B2) and
"+1.05 (head)" (MobileViT-v2). The measured input-adjacent, head-adjacent and pixel sums for
every port are in
[Scope of the conservation claim](#limitations). HiLRP's own sums are far from 1 on those two
backbones below the head, so the contrast this table supports is "not zeroed", not "conserved".

### Agreement with axiomatic Shapley values

Faithfulness Correlation is uninformative in this benchmark (absolute mean below 0.08 in
every cell, Friedman p = 0.57), so the primary yardstick is sampled permutation Shapley over
SLIC regions. Mean Spearman agreement, n = 50, m = 64.

| Method | Swin-T | PVT-v2-b2 | EfficientViT-B2 |
|---|---|---|---|
| **HiLRP** | **0.528** | **0.531** | **0.417** |
| Grad-CAM | 0.515 | 0.449 | 0.077 |
| SmoothGrad | 0.312 | 0.260 | 0.321 |
| Integrated Gradients | 0.192 | 0.279 | 0.281 |
| Saliency | 0.214 | 0.137 | 0.240 |

On Swin and PVT, HiLRP and Grad-CAM are statistically indistinguishable (paired Wilcoxon
p = 0.45 on Swin). The separation appears on EfficientViT, the same backbone where
Grad-CAM's Pointing collapses. Two independently computed yardsticks, one story.

### Cross-architecture baseline transfer

Pointing Game, n = 100. A dash marks a cell that is structurally undefined: the toolbox has
no rule for that attention mechanism, and a generic fallback would confound a transfer
failure with a wrong-algorithm artifact.

| Method | RN-50 | ViT-B/16 | Swin-B | PVT-v2 | MaxViT-S | MobileViT | EffViT-B1 | EffViT-B2 |
|---|---|---|---|---|---|---|---|---|
| Saliency | 0.82 | 0.65 | 0.62 | 0.62 | 0.58 | 0.90 | 0.86 | 0.66 |
| Integrated Grad | 0.92 | 0.64 | 0.66 | 0.64 | 0.62 | 0.90 | 0.90 | 0.90 |
| Input×Grad | 0.86 | 0.64 | 0.67 | 0.66 | 0.56 | 0.88 | 0.74 | 0.64 |
| SmoothGrad | 0.97 | 0.85 | 0.79 | **0.92** | 0.70 | 0.92 | 0.92 | **0.92** |
| VarGrad | **0.98** | 0.76 | 0.71 | 0.88 | 0.68 | 0.90 | 0.90 | **0.92** |
| GradientSHAP | 0.92 | 0.70 | 0.73 | 0.70 | 0.65 | 0.88 | 0.86 | 0.84 |
| Grad-CAM | **0.98** | **0.93** | **1.00** | 0.84 | **1.00** | **1.00** | 0.70 | 0.55 |
| Grad-CAM++ | 0.97 | 0.49 | **1.00** | 0.88 | 0.80 | **1.00** | 0.44 | 0.56 |
| Attention Rollout | - | 0.64 | - | - | - | - | - | - |
| AttnLRP | - | 0.83 | - | - | - | - | - | - |
| Occlusion | 0.95 | 0.76 | 0.78 | 0.78 | 0.66 | 0.92 | **0.96** | 0.88 |
| RISE | 0.82 | 0.73 | 0.68 | 0.68 | 0.70 | 0.90 | 0.92 | 0.78 |
| LIME | 0.94 | 0.87 | 0.86 | 0.86 | 0.80 | 0.90 | 0.80 | 0.90 |

### Verified numerics

Thresholds are tied to dtype and enforced on every push by
[`.github/workflows/conservation-tests.yml`](../.github/workflows/conservation-tests.yml), so a
regression in any relevance rule fails the build instead of silently degrading a published
number.

**Reference rules** (`tests/`): the float64 implementation of the rules and a two-stage toy
model, no attribution backend. This is where conservation is proved to machine precision.

| Property | Achieved (float64) | Test ceiling |
|---|---|---|
| Patch-merge rule, isolated | 3.4e-16 | < 1e-13 |
| Un-concat regroup plus exact inverse | 0.0 | < 1e-14 |
| Decomposed forward vs timm `PatchMerging` | 1.1e-15 | < 1e-14 |
| Shifted-window masked-pair relevance leak | 8e-43 | < 1e-30 |
| End-to-end 2-stage Swin, bias-free | 6.5e-9 | < 1e-2 |
| Four-primitive toy suite | 1e-11 | machine precision |
| Negative control: affine-LN rule leaks | 1.1 | must be > 0.5 |
| Negative control: naive epsilon plus bias unstable | 6.5e3 | must be > 1.0 |

**Production ports** (`tests_backend/`): the `*_lxt.py` ports that produce the published maps,
through LXT and zennit, on the smallest timm variant of each family with random weights. A
passing reference suite does not validate these, so they have their own CI job.

| Property | Achieved (float32) | Test ceiling |
|---|---|---|
| Forward unchanged by patches and composite, all six ports | 7e-7 | < 1e-5 |
| Map unchanged once every other port has patched, all six ports | 0 | ≤ 1e-6 |
| Conservation at every capture, bias-free Swin-T and PVT-v2-b0 | 7e-6 | < 1e-4 |

Conservation is asserted only on Swin and PVT-v2, and only bias-free, because that is the only
place the ports guarantee it. On EfficientViT, MobileViT-v2, MaxViT and flat ViTs the sum
drifts even without biases, so it is reported as a measurement in
[Limitations](#limitations) and not tested as a guarantee.

The reference job's install is deliberately minimal (CPU torch, torchvision, timm, numpy,
pytest) so the guarantee never comes to depend on the benchmark stack or the attribution
backend. The backend job installs `lxt==2.1`, `zennit` and `timm==1.0.16` separately.
`xai_bench` records missing optional dependencies rather than raising on import, so
`hilrp` stays importable in a bare environment; a registry lookup that comes
up empty names the missing package instead of reporting an unknown method. Run
`python -c "import xai_bench; print(xai_bench.registration_report())"` to see what registered.

### Key findings

**1. Attribution rankings do not transfer across ViT families.** Attention rollout scores
0.64 on ViT-B, worse than plain gradients, and is undefined on six of the seven ViT variants.
Grad-CAM breaks on the two EfficientViT backbones (B1 0.70, B2 0.55) at every target layer
tried, not on conv hybrids generally: on a healthy MobileViT-v2 it scores 1.00. Whether
linear attention or EfficientViT's non-linear head is responsible is not separated here.
Captum LRP does not run at all
on the four timm transformers, producing 0 rows. That is the literal infrastructure gap the
method fills.

**2. Grad-CAM's high Pointing is partly the dataset centre prior.** A content-free
centre-Gaussian blob also scores 1.00 on this eval set, whose centre-point prior is 0.97.
Grad-CAM maps correlate 0.72 and 0.63 with that blob against HiLRP's 0.25 and 0.14.

<p align="center">
  <img src="assets/qualitative_gallery.jpg" width="82%"
       alt="HiLRP compared with Grad-CAM, Grad-CAM++ and Integrated Gradients across six ImageNet classes">
</p>

HiLRP traces object structure (the cheetah's eyes and muzzle, the ambulance's markings, the
aircraft's engines and tail) where CAM produces a square blob and Integrated Gradients
produces scattered high-frequency noise.

**3. The blocky CAM structure is an upsampling artifact, not a rendering choice.**
Hierarchical backbones end at a 7x7 terminal grid, so a CAM map is 49 numbers upsampled 32x.

<p align="center">
  <img src="assets/cam_upsampling_artifact.jpg" width="100%"
       alt="CAM square artifacts on hierarchical backbones originate from the 7x7 terminal grid upsampled 32 times">
</p>

**4. Tracking relevance at every interface gives stage-localized maps for free.** The same
backward pass yields a map at every resolution: pixels, 56² tokens, 28², 14², 7². On Swin-B
the relevance sum at the token captures stays within 0.91 to 1.15 of the logit (0.74 at
pixels, n = 20), so each token-level map accounts for roughly the whole prediction. Texture
becomes parts becomes object.

<p align="center">
  <img src="assets/stage_localization.jpg" width="100%"
       alt="Relevance decomposed at pixels, tokens, and stages 1 to 3 of a hierarchical ViT">
</p>

**5. The maps depend on the weights.** Adebayo-style parameter randomization, independent and
cascading: agreement with the intact map falls monotonically as weights are destroyed from
the head downward. HiLRP is explaining the model, not the image.

<p align="center">
  <img src="assets/sanity_check.jpg" width="100%"
       alt="Parameter randomization sanity check: HiLRP maps degrade as model weights are progressively randomized">
</p>

**6. Pretraining objective, not architecture, shapes the explanation.** Frozen ViT-B, six
pretraining objectives, one common label-free scalar `cos(cls(x), cls(flip(x)))`.
Cross-objective map agreement is 0.36, not 1.0. The DINO family reaches 0.97 label-free
Pointing with no labels anywhere, and object emergence replicates across DINO generations;
MAE routes close to 0 relevance to pixels (0.07), consistent with a reconstruction objective.
This is possible because each model is explained in its own objective instead of through a
linear probe.

<p align="center">
  <img src="assets/rq2_pretraining_objectives.jpg" width="100%"
       alt="Relevance maps under supervised, DINO, DINOv2, DINOv2-reg and MAE pretraining on a frozen ViT-B">
</p>

**7. The same backward pass extends to multi-modal similarity with no new rule.**
Text-conditioned relevance on the CLIP ViT-B/16 image tower, scalar
`cos(f_img(x), f_text(caption))` with the text encoder frozen and the cosine denominator
detached. A matching caption concentrates relevance on the object, an unrelated caption does
not. Forward equivalence 1.9e-6.

<p align="center">
  <img src="assets/clip_multimodal.jpg" width="68%"
       alt="Text-conditioned CLIP attribution comparing a matching caption against an unrelated caption">
</p>

## Reproducing Results

**1. Dataset preparation.** Build the caches once. All eval scripts auto-pick the largest
cache present.

```bash
python scripts/data/cache_imagenets.py --num 1000 --split validation
python scripts/data/cache_voc.py --num 1000               # optional second dataset
```

**2. Configuration.** Run configs live in `configs/`. Start from `configs/default.yaml` for
the schema; `configs/full_benchmark_100.yaml` and `configs/hilrp_grid.yaml` are the two
protocol configs. Each declares `models`, `methods`, `metrics`, `metric_kwargs`, `dataset`,
`num_images`, `seeds`, and `run_name`.

`seeds` is honored. Every (model, method, image) is evaluated once per seed, the seed is
written on its row, and the runner reseeds Python, NumPy and Torch before each evaluation
from (seed, model, method, image). A stochastic method (SmoothGrad, VarGrad, GradientSHAP,
RISE, LIME) or metric therefore reproduces on the same hardware and library versions, and a
row does not change if the run is resumed or the sweep is reordered.

**3. Run the experiments.**

```bash
# Baselines
python -m xai_bench.runner --config configs/full_benchmark_100.yaml

# HiLRP rows: a DEDICATED run, never mixed with baselines in one process
python -m xai_bench.runner --config configs/hilrp_grid.yaml
```

**4. Regenerate tables and figures.**

```bash
python scripts/bench/benchmark_tables.py       # -> paper/results_tables_generated.tex
python scripts/bench/benchmark_figures.py      # -> figures/bench/*.pdf, *.png
python scripts/make_readme_assets.py           # -> docs/assets/*
```

**Individual results.** All scripts assume the repository root as the working directory.

| Result | Script |
|---|---|
| Gate 1 and 2 conservation, stage maps | `scripts/hilrp/gate2_real_swin.py` |
| Pointing game, four architectures | `scripts/hilrp/gate3_pointing.py` |
| n = 1000 scaled eval, bootstrap CI, McNemar | `scripts/bench/scaled_eval.py` |
| Grad-CAM target-layer check | `scripts/bench/gradcam_target_layers.py` |
| Conservation per port (input, head, pixels), one backbone per process | `scripts/hilrp/conservation_grid.py` |
| Shapley agreement (SLIC and SAM segments) | `scripts/hilrp/shapley_pilot.py`, `gate4_shapley_sam.py` |
| Lineage: naive AttnLRP vs HiLRP | `scripts/hilrp/lineage_naive_legs.py` |
| RQ2 pretraining objectives | `scripts/hilrp/rq2_pretraining.py` |
| CLIP multi-modal attribution | `scripts/hilrp/clip_demo.py` |
| Parameter-randomization sanity check | `scripts/hilrp/sanity_check.py` |
| Conditional equivariance | `scripts/hilrp/equivariance_test.py` |
| Gamma selection ablation | `scripts/hilrp/gamma_selection.py` |

## Limitations

**Scope of the conservation claim.** Three different things are established, and they should
not be read as one.

1. *Proved, and tested to machine precision.* Each rule conserves through the proven
   operations (patch merge, shift, mask, regroup, pool): reference suite, float64, toy model.
2. *Tested on the production ports.* Swin and PVT-v2 conserve at every capture, pixels
   included, when the weights are bias-free (backend suite, float32, 7e-6). On those two
   backbones the deviation on trained weights is therefore what the biases absorb.
3. *Measured, not guaranteed.* Everything on trained weights. `sum(R)/logit`, mean ± std
   over the first 20 ImageNet-S images, bench defaults (gamma = 0.25):

| Backbone | Input-adjacent capture | Head-adjacent capture | Pixels |
|---|---|---|---|
| Swin-B | 0.91 ± 0.07 | 1.00 ± 0.01 | 0.74 ± 0.07 |
| PVT-v2-b2 | 0.44 ± 0.04 | 1.01 ± 0.01 | 0.37 ± 0.03 |
| MobileViT-v2 | −0.05 ± 0.05 | 1.05 ± 0.01 | −0.14 ± 0.13 |
| MaxViT-S | 0.44 ± 0.06 | 0.77 ± 0.04 | 0.16 ± 0.03 |
| EfficientViT-B1 | 0.37 ± 0.34 | 4.16 ± 4.09 | 0.18 ± 0.17 |
| EfficientViT-B2 | 0.79 ± 0.36 | 3.80 ± 3.06 | 0.34 ± 0.22 |

The head-adjacent capture is conserved on Swin, PVT-v2 and MobileViT-v2. It is not on MaxViT,
and not on EfficientViT, where the sum also swings in sign from image to image. No backbone
conserves to the pixels.

The EfficientViT ratio needs one caveat. This family's top logit sits near zero (1.8 ± 0.9 on
these images, and negative on some), so dividing by it is ill-conditioned and inflates the
spread. The gap is not only that, though: in absolute terms the head-adjacent relevance sums
to 8.5 ± 6.3 on EfficientViT-B2 against that logit of 1.8. On MaxViT-S the same gap is a
steady −2.2 ± 0.5 on a logit of 9.2. We do not claim end-to-end pixel-level conservation, and on
EfficientViT, MaxViT and MobileViT-v2 we do not claim it below the head either. Reproduce one
backbone per process with `python scripts/hilrp/conservation_grid.py <model> 20`. An earlier
Swin-T run over 20 images gave tokens 1.05 ± 0.11, pixels 0.82 ± 0.08, final logit 1.00.

Conservation is also necessary but not sufficient. A passing conservation check proves no
relevance leaked, but wrong token bookkeeping still conserves numerically while scrambling the
spatial map, which is why the qualitative gates exist alongside the numeric ones.

**Not a universal win.** HiLRP does not top every metric on every backbone. Grad-CAM saturates
the soft bounding-box Pointing metric on Swin and MobileViT (1.00). On those models the HiLRP
case is capability (conservation, stage maps, defined rules), not a pointing score.

**Known failure cases.**

- *MobileViT-v2 robustness.* Max-Sensitivity is 8.51, against 0.44 to 1.12 on Swin, PVT and
  EfficientViT. A gamma sweep showed this is inherent to the deep separable-conv pixel
  gradient, not a tunable parameter.
- *MobileViT-v2 spatial smearing.* Its `GroupNorm1` normalizes across `(C, H, W)`, so every
  pixel mathematically depends on the background. Conserving honestly means passing relevance
  back through that global mean, which visibly smears the map. Naive AttnLRP looks prettier
  there precisely because it fails to patch `GroupNorm1` and destroys conservation (the sum
  flips from +1.00 to -1.94).
- *Flat CLS-pooled ViTs.* The conserving CP-LRP default scores 0.70 Pointing, above the 0.61
  prior but below the non-conserving `attn_mode='attnlrp'` diagnostic (0.90, with pixel
  conservation error rising from 0.39 to 0.85). We report the conserving number. Pointing
  without conserving is the exact failure this work criticizes, so that mode stays a labeled
  diagnostic and never the headline.
- *Equivariance is conditional.* ViTs are not shift-equivariant, and masked Swin has no exact
  input-space translation symmetry. The theorem is conditional: if the forward commutes with a
  token or pixel permutation, HiLRP commutes. Empirically, at equal forward drift, HiLRP
  consistency is about 0.90 against saliency's 0.41 to 0.61.

**Unsupported architectures.** CNNs are out of scope by design and are carried as baselines
only. Real-model ports still outstanding: true cross-attention layers (VQA, LXMERT, DETR
decoder), channel and coordinate attention (SE hybrids), and Deformable DETR. Each is proved
conserving in the toy suite but not yet exercised on trained weights.

**Relation to prior work.** HiLRP inherits AttnLRP's stabilized block-internal rules (softmax
Taylor, bilinear split, LayerNorm) rather than reimplementing attention internals, and adds
the primitive decomposition, the resolution-reduction rules, and the guards that stop silent
failure. On flat ViT-B this is parity with the lineage (0.770 vs 0.770), not a win claim.
