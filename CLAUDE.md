# CLAUDE.md - HiLRP / xai_bench

Project state for Claude Code. This is the spec, not a diary. Keep it tight.
Read this and `NOTES_next.md` at the start of every session.

## Author preferences (apply to all generated code and text)

- No em dashes anywhere. Use commas, colons, or separate sentences.
- LaTeX for any document deliverable. Concise prose, short sentences, no AI-sounding filler.
- Do not reimplement attention internals naively. Inherit LXT/AttnLRP stabilized block
  rules and drop in only the merge/shift/mask/regroup rules.

## The method (one line)

A conservation-guaranteed attribution framework for hierarchical and hybrid vision
transformers, formulated as relevance flow on a multi-resolution token graph. Extends
AttnLRP/LRP to Swin/PVT/CvT/CNN+ViT, where no conservation-based attribution currently
has defined rules.

## Paper framing (do not drift)

The paper is NOT "HiLRP is the best heatmap for every model and metric." That claim is
false and fragile. The paper is:

Custom ViT-style models are now common in domain applications, and each new mixture of
CNN stems, local attention, patch merging, spatial reduction, shifted windows, pooling,
and normalization creates a new XAI support gap. Existing methods depend on architectural
assumptions: Grad-CAM needs a meaningful terminal spatial feature map, attention rollout
needs usable global attention, Captum LRP needs supported modules, and perturbation
methods are output-level and expensive. HiLRP gives a principled conservation-valid
explanation path for this regime, even when it is not the top visual localizer on every
single metric.

THE AIM (north star, user-stated 2026-07-06): "From the start this is for ALL ViTs, not
one architecture. Many custom ViTs come every year for different domains, but there is no
common trustable XAI for them." The paper must READ as this vision, not as a rules paper.

Paper is now THESIS-FIRST (reframed 2026-07-06, title/abstract/intro/contrib):
Title: "One Trustable Explanation for Any Vision Transformer: Conservation-Valid
Attribution via Attention Primitives."
Two pillars carry everything:
- TRUSTABLE = conservation (the map is a DECOMPOSITION of the prediction, not a heuristic;
  verified to machine precision). We proved the alternatives are NOT trustable (naive
  AttnLRP = zero-sum degenerate maps that still "point"; Grad-CAM = content-free center blob).
- UNIVERSAL = the 4-primitive decomposition (linear / bilinear mix / normalization-gate /
  reindexing). Any ViT, present or FUTURE, is a new arrangement of the same primitives,
  covered by construction. Future-proofing is an explicit selling point: next year's
  backbone = same 4 primitives, no new rule.
The resolution-reduction rules (patch merge / shifted window / spatial reduction) are ONE
INSTANCE of the framework, demoted from headline to supporting result.

Correct high-level claim (updated):
"HiLRP is the first attribution framework that propagates conservation-valid relevance
through ANY attention mechanism and downsampling operation, so it gives one trustable
explanation for the whole, growing space of custom Vision Transformers."

Core value proposition: universal + trustable + future-proof. Use the 10-arch x 13-method
benchmark as evidence that no prior method is both, and the 4-primitive coverage
(attention_primitives.py + tests + CLIP multi-modal) as evidence that HiLRP is.

Do not write:
- "HiLRP beats all baselines on all metrics."
- "HiLRP is the first LRP for ViTs."
- "HiLRP gives exact pixel-level conservation."
- "MobileViT proves Grad-CAM collapse."

Do write:
- "One trustable explanation for any Vision Transformer, present or future."
- "Any ViT is four conserving primitives; a new architecture is a new arrangement, covered by construction."
- "HiLRP is conservation-valid where naive LRP/AttnLRP extensions fail."
- "HiLRP is strongest where CAM assumptions break, especially EfficientViT and PVT."
- "On models where CAM saturates soft pointing, HiLRP remains useful because it provides
  conservation, stage maps, and defined rules."
- "Pixel-level conservation is approximate and quantified."

## Three novelty claims

1. Closed-form flow-conserving rules for the three resolution-reduction mechanisms:
   patch merging, shifted-window partitioning, spatial-reduction attention.
2. First conservation method for hierarchical/hybrid ViTs.
3. Stage-localized attribution maps that output-perturbation methods (ViT-CX, TIS)
   structurally cannot produce.

## Attention as FOUR conserving primitives (novelty upgrade, 2026-07-06)

THE headline reframing to lift novelty from "extension" to "unifying framework"
(the Chefer ICCV'21 move: generalize to ALL attention types). Every attention
mechanism decomposes into four operation classes, each with one conservation rule:

1. Linear map (Q/K/V/out projections, poolings, kernel maps phi, RoPE) -> lrp_linear.
2. Bilinear mix (matmuls: QK^T, attn.V, linear-attn products, deformable sampling)
   -> bilinear_lrp (uniform split) or CP (detach one factor as gating).
3. Normalization/gate (softmax, linear-attn denominator, sigmoid/SE gate, LayerNorm)
   -> detached-denominator / detached-gate identity rule.
4. Reindexing/sparsity (heads, windows, shifts, dilation, groups) -> permutation,
   exact (Lemma 2).

One line: attention = linear -> bilinear mix -> normalize/gate -> aggregate over
some sparsity pattern; HiLRP conserves each, so it conserves through any attention.

Coverage status (15+ types). DONE+verified on real models: self/MHA, windowed/shifted
(Swin), spatial-reduction (PVT), linear (EfficientViT), separable (MobileViT),
multi-axis (MaxViT). PROVEN-CONSERVING in the toy suite (float64, 1e-11,
`attention_primitives.py` + `tests/test_attention_coverage.py`, 6 tests): self, cross
(encoder-decoder), co-attention (bi-modal), linear, channel/coordinate (gated),
dilated/sparse. TRIVIAL: grouped/multi-query (broadcast), RoPE/rel-pos (rotation/
additive bias). ONE SMALL NEW RULE NEEDED: deformable attention (bilinear feature
sampling at learned offsets; sampling is linear so it conserves, needs an explicit
port). Real-model ports still TODO: cross/co (unlocks CLIP/VQA multi-modal, highest
novelty value), channel/coordinate (SE-hybrids), deformable (Def-DETR).

New central claim once ports land: "HiLRP is the first attribution framework that
propagates relevance through ANY attention mechanism, proven and verified across 15
attention types (self, linear, windowed, cross, co, channel, coordinate, dilated,
deformable)." Directly answers the "just an AttnLRP extension" reviewer.

## Naming caveat (resolve before paper)

`hilrp` (hierarchical LRP) is the internal name. Check availability against existing
LRP variants (AttnLRP, AliLRP, DynamicLRP) before it goes in a paper. The name should
signal the contribution (hierarchical / resolution-reduction), not just "LRP but ours".

## Unifying proposition (Gate 0, derived)

Any resolution-reduction of the form `W . phi(concat(neighborhood))` admits one epsilon-rule
and conserves. Corollaries: patch merging, strided-conv patch embed, PVT spatial reduction.
This is what makes the method architecture-agnostic on paper and handles CNN+ViT hybrids
for free. Cyclic shifts and window partitions are permutations/regroupings, so they
conserve exactly (Lemma 2): only ~4 patch points needed for the whole architecture.

## Core design decisions (with data behind them, do not silently change)

- Backend: LXT `efficient` (Gradient x Input). The `explicit` pure-epsilon backend blows
  up on deep vision models (relevance amplified +/-1e3 with sign flips descending stages);
  this is the known epsilon-cancellation instability, not a bug in our rules.
- Attention: CP-LRP, not AttnLRP. CP wins on conservation (0.82 vs 0.29) and class
  discrimination in our ablation. This is a one-flag ablation, keep it that way.
- LayerNorm: identity-LN rule (relevance passes through). The affine-detach LN rule makes
  the bias beta a relevance sink and destroys conservation (relerr 1.1) at trained scale.
- GELU: identity. Linear/Conv2d: zennit Gamma(0.25).
- Unifying insight: LayerNorm and softmax both handled by the same principle, detach the
  normalizing denominator, locally linear, conserves.

## Conservation status (CRITICAL framing constraint)

Conservation is GUARANTEED at the token level and through the proven operations
(merge, shift, mask, regroup, pool). It is APPROXIMATE at the pixel level: the Gamma(0.25)
rule on Linear/Conv2d and the patch-embed conv introduce controlled, monotone, reproducible
deviation.

Measured (Gate 2-real, 20 imgs): tokens 1.05 +/- 0.11, pixels 0.82 +/- 0.08, final 1.00.

NEVER claim end-to-end pixel-level conservation. Anyone can measure 0.82 and call it
overclaiming. Correct claim: "conservation holds through the hierarchical operations
(proven, verified to 1e-7 in the toy); the gamma-rule introduces controlled deviation at
the pixel level in exchange for denoising, quantified as 0.82 +/- 0.08."

## Verified gate results (these are the CI ceilings, encode as thresholds tied to dtype)

- Gate 1 (float64): patch-merge pure rule 2.5e-15; un-concat regroup 2.8e-16;
  decomposed forward vs timm.PatchMerging bit-for-bit 0.0.
- Gate 2 toy (float64): end-to-end 2-stage Swin, bias-free conservation 6.5e-9, holds at
  every stage. Masked SW-MSA pair relevance ~1e-43 (structurally zero). Shifted-window
  mask risk is dead.
- Gate 2-real (timm Swin-T via LXT): pointing game 19/20 = 0.95; maps trace object
  structure not blobs; stage localization figure works (texture->parts->object across
  56/28/14/7); identity-LN faithfulness cost: none visible; conservation on misclassified
  inputs holds too (explaining the model, not the ground truth).

Known instability (documented, resolved by inheriting LXT rules): naive epsilon + naive
bias-absorb is numerically unstable with biases (relerr -> 1e5, near-zero bilinear
denominator amplification). Test our rules in ISOLATION given stable block rules as input;
let LXT's own tests cover block internals.

## The benchmark half (motivation + baseline table, already run)

xai_bench: registry-based runner, Quantus-backed metrics, ImageNet-S (cached, has GT
segmentation masks -> enables pointing_game). Four findings, each a plank under the method:

1. Attention-native methods collapse on ViTs (rollout pointing-game deit 0.48, vit 0.64,
   worse than plain gradients ~0.85, near-flat maps).
2. Grad-CAM breaks on LINEAR-ATTENTION models, not on conv-hybrids generally.
   CORRECTED 2026-07-04: the original "mobilevitv2 0.49" was a preprocessing
   artifact (the bench fed ImageNet-normalized inputs to a model expecting raw
   [0,1]; top-1 was ~0). On the healthy model grad_cam scores 1.0 there. The
   REAL gap is efficientvit_b2 (clean stats): grad_cam 0.551, below the 0.61
   random-point prior, while HiLRP scores 0.960. The gap claim must name
   linear attention (EfficientViT family), never mobilevitv2.
3. Faithfulness-correlation is essentially noise (|mean| < 0.08, std ~0.22 across all
   cells). This is why the PRIMARY yardstick is Shapley-agreement, not deletion AUC.
4. Captum LRP does not even run on the four timm transformers (unsupported Identity
   layers -> 0 rows). The literal infrastructure gap the method fills.
5. Grad-CAM's high pointing where it "wins" (swin/mobilevit 1.0) is the DATASET CENTER
   PRIOR, not localization (confirmed 2026-07-06, paper sec:gradcam-blob). A content-free
   center-Gaussian blob also scores pointing 1.0; Grad-CAM maps correlate 0.72/0.63 with
   that blob (center-of-mass within 11-21% of image center), while HiLRP correlates only
   0.25/0.14 (structure-following). This is WHY pointing saturates and why conservation
   validity + Shapley agreement are the real discriminators.

HiLRP now covers all 8 ViT-family benchmark models (quantus grid, mean pointing 0.90):
vit_base 0.78, deit_base 0.84, swin_base 0.96, pvt_v2_b2 0.98, efficientvit_b1 0.94,
efficientvit_b2 0.96, maxvit_small 0.96, mobilevitv2 0.78. `hilrp` is a registered bench
METHOD (dedicated run: class-patches contaminate other methods in-process).

Full benchmark interpretation (important for paper quality): the existing benchmark is
not a failed attempt to show HiLRP wins everything. It is evidence that XAI behavior is
architecture-dependent. Grad-CAM is strong on some CNN-like or spatial-feature models,
fails on linear-attention EfficientViT, and saturates soft pointing on Swin/MobileViT.
Attention rollout collapses on ViTs. Gradient methods vary by model family. Perturbation
methods are expensive and output-level. Faithfulness metrics are often near zero with
large variance. This motivates a paper framed as benchmark + metric critique + HiLRP as
the conservation-valid resolution for hierarchical/hybrid architectures.

If the user cannot scale beyond 100 images, do not panic. Use a fixed paired protocol:
same 100 images for all compared methods, paired bootstrap confidence intervals, paired
Wilcoxon tests, and claims only for large effects. With n=100, strong claims are allowed
for large gaps such as efficientvit_b2 HiLRP 0.960 vs grad_cam 0.551. Do not claim tiny
differences. Shapley can remain smaller and focused because it is expensive: n=50-100,
key models only, key baselines only.

### 1000-Sample Scaleup Validation (Gate 4)
Executed 2026-07-06 on `ImageNet-S` (n=1000) using a custom robust API download cache (bypassing HF datasets bugs).
Findings definitively prove that while naive AttnLRP extensions might seem to localize visually, they catastrophically fail mathematical conservation on hierarchical/hybrid ViTs:
- **Swin-B**: AttnLRP-naive scores pointing 0.967 but deep conservation is massively broken (+3.33 $\pm$ 0.70), meaning it leaks >3x the actual relevance. HiLRP-CP maintains pointing (0.950) while holding tight deep conservation (+0.93 $\pm$ 0.07).
- **EfficientViT-B2**: AttnLRP-naive pointing degrades to 0.868, with near-total deep relevance collapse (0.00) and volatile head conservation (+0.66 $\pm$ 0.78). HiLRP holds pointing at 0.960.
- **MobileViT-v2**: AttnLRP-naive scores 0.943 but again completely loses deep conservation (0.00). The relevance fails to reach the stem.

This large-scale run cements the core paper claim: visual localization without mathematically sound resolution-reduction rules is just an illusion of correctness. HiLRP is the only conservation-valid method for these architectures.

If scaling to 1000 is possible, scale only cheap metrics and key baselines. Do NOT run
every metric and every slow method blindly. Recommended two-tier design:
1. Large-scale tier: 1000 images, 8-10 models, 5-6 methods, cheap metrics such as
   pointing, sparseness, selected robustness, and faithfulness-correlation only as a
   metric-failure diagnostic.
2. Deep-validation tier: 50-100 images, Swin/PVT/EfficientViT, Shapley agreement,
   conservation validity, qualitative stage maps.

For A* positioning, the contribution set is strongest as:
large cross-architecture benchmark + metric failure diagnosis + conservation-valid HiLRP
rules + Shapley validation + RQ2 pretraining-objective finding. Scale and statistics are
the gate, not more method invention.

## Sensitivity testing note

top1-vs-top2 is the WRONG class-sensitivity test (single-object images share evidence).
Use the distant-class probe: wrong classes should flip sign on the object (correct
signed-evidence semantics). Script already does this.

## Experiment plan (training-free, all frozen timm/released checkpoints)

Models: vit_base, deit_base, swin_t/s, pvt_v2, one hybrid (coatnet/mobilevit),
plus dinov2/mae/clip for the SSL branch. NO fine-tuning anywhere. Medical/BraTS block
DROPPED (no time to train). Localization ground truth comes from ImageNet-S / VOC masks.

Win condition for Gate 3: pointing-game on the LINEAR-ATTENTION models where Grad-CAM
genuinely collapses (efficientvit_b2: grad_cam 0.551 vs HiLRP 0.960, landed 2026-07-04),
NOT deletion AUC (which our own bench proved cannot discriminate). On conv-hybrids with
healthy preprocessing (mobilevitv2) grad_cam saturates the soft pointing metric (1.0) and
HiLRP is competitive with gradient methods (0.86 vs 0.88-0.92): the mobilevit story is
capability (stage maps, conservation), not pointing wins. Pointing on this eval set has a
0.97 center-point prior; lead with energy pointing (HiLRP +0.168 over uniform baseline),
mask localization, and the small-box subset (HiLRP 0.94). Primary evidence overall:
agreement with sampled Shapley (pilot: HiLRP rho +0.58 vs saliency +0.20, n=10, m=32).

### MobileViT and the GroupNorm1 Smearing Effect (Faithfulness vs. Pretty Pictures)
MobileViTv2 uses `GroupNorm1` in its transformer blocks, which normalizes across the entire spatial image `(C, H, W)` simultaneously rather than per-token `(C)` like standard LayerNorm. Because it subtracts a global image mean, every pixel mathematically depends on the background. To maintain strict conservation, HiLRP honestly passes relevance back through this global mean, causing object relevance to smear spatially across the background (making the map look visually "washed out"). AttnLRP (Naive) produces a prettier localized map ONLY because it fails to patch GroupNorm1, effectively detaching the mean and destroying conservation (sum flips from +1.0 to -1.94). This proves HiLRP's visual degradation on MobileViT is a feature (faithful explanation of architectural spatial entanglement) while AttnLRP "cheats" by leaking relevance.
**Note on Gamma Selection**: For deep conv-hybrids like MobileViT, use a split-gamma strategy (`linear_gamma=0.25`, `conv_gamma=0.1`) to prevent exponential relevance explosion in the deep CNN stems.

## Confirmed extension branches (after trunk)

- PVT spatial-reduction rule (third corollary, generalization evidence).
- Equivariance theorem: method is provably equivariant under each architecture's TRUE
  symmetry group (patch-aligned for ViT, window-multiple for Swin), reuses Lemma-1
  sum-exchange. NOT naive pixel shifts (ViTs are not shift-equivariant).
- SSL-scalar attribution: point the same machinery at a label-free scalar (DINO
  view-similarity, MAE reconstruction loss) to explain frozen feature extractors with no
  classifier head. One-line change once the trunk runs.

## Conditional bonus (include only if Gates 1-3 pass on time, drop without regret)

- Min-cut on the relevance graph. Well-posed ONLY because conservation makes relevance a
  genuine flow (max-flow min-cut theorem needs conservation + capacity constraints).
  Delivers "bottleneck tokens": the smallest token set through which all relevance must
  pass. A new explanation type. Raw-attention edges gave garbage here in 2020
  (Attention Flow); conservation-derived edges make it legal.

## Analysis-only tools (real, but NOT the core method, no faithfulness claim)

Demoted here because as a fusion/weighting mechanism each breaks conservation or rewards
shared bias (the C-RAG failure pattern). Legitimate ONLY in clearly-labeled descriptive or
evaluation roles:

- NMF: post-hoc concept discovery across many relevance maps (stack maps, factor, find
  recurring patterns like texture / object-boundary components). Non-negativity matches
  gamma-rule relevance. Descriptive only.
- Spearman / Pearson: EVALUATION agreement score against the Shapley reference. NEVER a
  self-referential fusion weight (agreement != correctness; attention methods correlate
  BECAUSE they share the CLS/outlier-token bias).
- Optimal transport: optional comparison metric between relevance maps across pretraining
  objectives in the RQ2 study, or to measure how relevance transports across a patch-merge
  (moves / concentrates / diffuses). Not an attribution (no completeness statement).
- TTA map-averaging: explicitly-heuristic denoising option that MUST report its
  conservation cost. Outside the core. Averaging aligned maps breaks conservation the same
  way PCA fusion does.

## Validation design (the yardstick, not the method)

- Sampled permutation Shapley over SAM segments, small eval set (200-500 imgs, pilot at
  50). Axiomatic uniqueness (efficiency, symmetry, null-player, linearity), unbiased,
  1/sqrt(m) convergence. This is the PRIMARY evidence because our own bench proved
  faithfulness-correlation cannot discriminate methods. Shapley INTERACTION indices were
  considered and killed (cost explodes); plain sampled Shapley covers the role.
- Sanity battery: parameter randomization (Adebayo, does the map degrade when weights are
  randomized), class sensitivity (distant-class probe, see below), deletion/insertion AUC
  (report with the weak-discrimination caveat), ImageNet-S / VOC mask localization.
- SAM only defines regions; all importance signal comes from the target model, so no
  second-model faithfulness trap.
Current Shapley status (2026-07-05, n=50, m=64, SLIC/two-pass honest protocol):
- swin_tiny:       HiLRP +0.528 | Grad-CAM +0.515 | SmoothGrad +0.312 | Saliency +0.214 | IG +0.192
- pvt_v2_b2:       HiLRP +0.531 | Grad-CAM +0.449 | IG +0.279 | SmoothGrad +0.260 | Saliency +0.137
- efficientvit_b2: HiLRP +0.417 | SmoothGrad +0.321 | IG +0.281 | Saliency +0.240 | Grad-CAM +0.077
Reading: HiLRP is highest on all three Shapley legs. Grad-CAM agrees on Swin/PVT
but collapses on linear attention, matching its pointing collapse on EfficientViT.
Use this as the independent yardstick that supports the architecture-dependent XAI
story. Remaining Shapley work: SAM segments, optional mobilevit wiring, and CIs.

## HiLRP improvement priorities

Improve HiLRP where it is actually weak, not by chasing every metric:

1. MobileViT robustness: max_sensitivity is high (~8.51), an honest limitation (in paper).
   Gamma sweep done (`gamma_selection.py`): gamma does NOT reduce spikiness, it is inherent
   to the deep separable-conv pixel-gradient. The reported "blank heatmap" was a DISPLAY bug
   (heavy-tailed: ~27% of |mass| in top 1% of pixels), NOT a method failure; the map localizes
   (0.86). Fixed via `hilrp/viz.py:normalize_for_display` (percentile-99 clip + optional
   post-hoc smooth). RULE: use that helper for ALL map rendering. gamma=0.25 is the global
   default, justified by the ablation (paper sec:gamma); no per-model tuning.
2. Metric hardening: Pointing Game is soft on ImageNet-S. Add energy-in-mask,
   small-object subset, and full mask localization when masks are cached. Lead with
   these before deletion AUC.
3. Conservation tables: this is the discriminator. Show naive AttnLRP/LRP either inflates
   or zeroes relevance on hierarchical/hybrid models, while HiLRP remains conservation-valid.
4. Optional visualization variants: smoothing or TTA map averaging may be offered only as
   clearly labeled post-hoc visualization. Always report the conservation cost. Never
   fold these into the core method.
5. Per-family attention mode: flat CLS ViTs use AttnLRP mode; windowed/hierarchical/
   hybrid models use CP mode. Present this as a design rule, not an ad hoc fix.

## Competitive positioning (related work + baselines)

- ViT-CX / TIS / C2F-Explainer: output-perturbation, causal but OUTPUT-level. They tell you
  THAT a patch mattered; we tell you WHERE in the network it emerged (stage-localized).
  That is a different axis, not a faithfulness-score arms race. TIS's OOD-token critique is
  why masking baselines use blur/learned tokens, not black patches.
- Chefer et al. (2021) and GenAtt: gradient x relevance for FLAT ViTs. This is the
  GradCAM x LRP marriage already done, and the ViT version of Relevance-CAM. They are
  BASELINES and the lineage we extend, not competition. They have no defined rules for
  hierarchical operations.
- AttnLRP (Achtibat 2024, LXT/Zennit): fixes block-internal ops (softmax Taylor, bilinear
  split, LayerNorm) for FLAT transformers. We inherit these and add the hierarchical rules.
- DynamicLRP (Dec 2025) critique: every new module type needs a hand-derived rule,
  unsustainable. That critique is our opening, stated in someone else's words. Hierarchical
  ViT variants are the "unseen patterns" the current toolchain cannot handle.
- DINO emergent maps: NOT an XAI method, a property of the SSL representation (CLS attention
  over patches, per-head, thresholded to ~60% mass). Doesn't transfer to supervised ViTs
  (evidence for RQ2: pretraining objective, not architecture, drives attention
  explainability). Structurally unavailable for Swin (no CLS, windowed attention). Use as a
  BASELINE and probe for the SSL branch, and to measure what DINOv2 outlier/register tokens
  contribute (a mechanistic side-result no attention viz can give).

## The abstraction (why one method covers all variants)

Every architecture in scope = alternating additive updates (x + f(x)) by differentiable
mixing operators, punctuated by linear resolution changes. Three structural invariants:

1. Additive update wraps every sublayer (attention / windowed / reduced / conv all fit).
   This is what makes exact skip-vs-branch relevance splitting possible = conservation.
2. Tokens on a spatial grid mixed by a differentiable operator. Window partition, cyclic
   shift, pyramid pooling are just sparsity patterns / reindexings, not new objects. A zero
   connection carries zero relevance, so sparsity is handled for free.
3. Resolution changes are linear maps on token neighborhoods (the W.phi(concat) form).
   Prove conservation on the abstraction once; every named architecture (present or future) is
   an instance, not a new derivation. This is the answer to "does it generalize".

## Lemma-1 (the one trick everything rests on)

For a linear layer y_i = sum_j w_ij v_j, contributions z_ij = w_ij v_j, denominators
z_i = sum_j z_ij, the rule R(v_j) = sum_i (z_ij / z_i) R(y_i) conserves because
sum_j R(v_j) = sum_i (R(y_i)/z_i) sum_j z_ij = sum_i R(y_i). Conservation = one swap of
summation order. The epsilon in (z_i + eps) only prevents divide-by-zero and introduces a
measurable leak z_i/(z_i+eps). Holds for ANY linear map (dense/sparse/structured).
Lemma 2: permutations and coordinate embeddings (concat) have one nonzero z_ij per row, so
they conserve EXACTLY (ratio = 1), no leak. This is why shifts/partitions/regroup are free.

## Conservation is necessary, NOT sufficient (the bug conservation can't catch)

A passing conservation check proves no relevance leaked. It does NOT prove the token
bookkeeping (un-concat -> correct spatial position, Swin's interleaved channel order) is
right. Wrong indexing still conserves numerically but scrambles the spatial map. Gate 2
(qualitative sanity on real images) catches this; Gate 1 cannot. Also: conservation !=
faithfulness. A rule can conserve and still smear relevance uselessly. That is why the
gamma-rule and Gate 2/3 matter.

## RQ2 (executed 2026-07-06, the "so what" that lifts toward A*)

Pretraining objective as a determinant of explainability. EXECUTED (`scripts/rq2_pretraining.py`,
paper sec:rq2): fixed ViT-B/16, 4 pretrainings (supervised/DINO/MAE/CLIP), one common
label-free scalar cos(cls(x), cls(flip(x))). n=30 result: cross-objective map agreement 0.33
(vs 1.0 if objective irrelevant), so the objective, not the architecture, shapes the
explanation. DINO localizes best (label-free pointing 0.97, recovering its known object
emergence with NO labels), MAE routes ~0 relevance to pixels (0.07, reconstruction objective).
The method enables this by explaining each model IN ITS OWN OBJECTIVE instead of a linear probe.
Scale-up for A*: n>=100, significance test on the agreement, a second architecture (ViT-S),
optionally a MAE-reconstruction scalar.

## Repo structure target

- `xai_bench/methods/hilrp/` : the method as a proper package.
- `tests/` : pytest conservation suite, thresholds tied to dtype (see CI ceilings above).
- Gate scripts promoted from scratch, not ephemeral.
