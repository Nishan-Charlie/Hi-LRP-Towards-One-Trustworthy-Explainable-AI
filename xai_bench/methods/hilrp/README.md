# HiLRP — hierarchical LRP for resolution-reducing vision transformers

Conservation-guaranteed attribution for hierarchical / hybrid ViTs (Swin patch
merging, shifted windows, PVT/CvT spatial reduction), formulated as relevance
flow on a multi-resolution token graph.

> **Name is internal.** `HiLRP` is unclaimed in the literature (cf. AttnLRP,
> LRP-QViT, CA-LIG) but signals "hierarchical" rather than "resolution-reduction
> rules". Revisit before it goes in a paper.

## What is the contribution vs. what is inherited

| File | Role |
|---|---|
| `hierarchical.py` | **The contribution.** Flow-conserving rules for patch merging, the shifted-window (cyclic shift + attention mask) transition, and the un-concat regroup. |
| `rules.py` | *Inherited* LRP primitives (linear ε-rule, LayerNorm/softmax identity rule, bilinear split, residual). Self-contained so tests run without LXT. |
| `toy_swin.py` | **Test-only** 2-stage reference model. Uses naive block internals that are numerically unstable with biases — do **not** attribute real models with it. |
| `tests/test_conservation.py` | The guarantee, as dtype-tied numeric ceilings. |

## The unifying proposition

Any resolution-reduction of the form `T = W · φ(concat(neighborhood))` (φ a
conservation-preserving normalization, `concat` an index partition) admits one
ε-rule and conserves relevance. Corollaries, one proof: **Swin patch merging**,
**strided-conv patch embedding**, **PVT/CvT spatial reduction**.

## Two design decisions, both forced by numerics

1. **LayerNorm = identity rule** (detach the whole normalization, pass relevance
   through). The affine-detach alternative makes the LN bias β a relevance sink
   and destroys conservation (relerr ~1.1). *Caveat:* identity-LN is the
   **stronger** simplification — it also discards γ's effect on relevance
   routing. That is a **faithfulness** question, invisible to a conservation
   test, and must be checked at **Gate-2-real** (real Swin-T via LXT: maps land
   on objects, are class-sensitive). **Conservation ≠ faithfulness.**
2. **Inherit LXT for the block internals.** The naive attention rules here blow
   up with biases on (near-zero bilinear denominators amplify). Production drops
   the `hierarchical.py` rules into LXT's stabilized rules rather than
   reimplementing softmax/matmul/linears.

## Verified numbers (float64, `pytest tests/`)

| property | achieved | CI ceiling |
|---|---|---|
| patch-merge pure rule (β=0, ε=0) | ~1e-15 | `< 1e-13` |
| patch-merge rule isolated (β≠0, ε=0) | 3.4e-16 | `< 1e-13` |
| patch-merge in-context (ε=0) | 0.0 | `< 1e-13` |
| un-concat regroup + exact inverse | 0.0 / ~1e-16 | `< 1e-14` |
| LN identity conserves with β≠0 | ~1e-15 | `< 1e-13` |
| forward vs timm `PatchMerging` | 1.1e-15 | `< 1e-14` |
| SW-MSA masked-pair relevance leak | 8e-43 | `< 1e-30` |
| bias-free composition (loose sanity) | 4e-6 | `< 1e-2` |
| **negative:** affine-LN leaks | 1.1 | `> 0.5` |
| **negative:** naive+bias unstable | 6.5e3 | `> 1.0` |

Run: `python -m pytest xai_bench/methods/hilrp/tests/ -v`

## Gate-2-real: pretrained Swin-T (timm), verified 2026-07-04

`swin_lxt.py` + `scripts/gate2_real_swin.py`, on LXT's *efficient* (Gradient x
Input) backend per their vision recipe: CP-LRP inside attention, identity-rule
LN/GELU patches, zennit **Gamma(0.25)** on Linear/Conv2d. Two findings that
matter:

1. **Pure epsilon everywhere is not viable on a real Swin** — forward-exact and
   head-conserving, but relevance amplifies with sign flips descending stages
   (|sum/logit| up to ~10^3 at tokens). Gamma collapses this to ~1.
2. **CP vs AttnLRP attention** (one flag, `attn_mode`): CP conserves better
   (pixels 0.82 vs 0.29 of logit) *and* discriminates classes better on average
   (pos-part corr vs distant class 0.50 vs 0.84). CP is the default; AttnLRP
   mode kept as ablation.

Results over ImageNet-S (cached), swin_tiny, gamma=0.25, CP:

| check | result |
|---|---|
| conservation sum(R)/logit, 20 imgs | tokens 1.05±0.11 · stages 1.04–1.30 · final 1.00±0.01 · **pixels 0.82±0.08** |
| pointing game (pixel-map argmax in GT bbox) | **19/20 = 0.95** |
| class sensitivity (pred vs distant, 5 imgs) | signed corr +0.23, pos-part +0.50; distant classes flip sign on the object |
| stage-localized maps | `results/hilrp_gate2real/stage_localization.png` — texture→parts→object across 56/28/14/7 |
| identity-LN faithfulness cost | none visible: maps trace object structure (lattice/struts), background quiet |

Pixel-level maps come through the strided patch-embed conv (Gamma on Conv2d) —
the fourth corollary of the proposition, working on a real model.
Caveat: `ensure_patched()` is class-level and process-global (nn.GELU,
nn.LayerNorm, timm WindowAttention).
