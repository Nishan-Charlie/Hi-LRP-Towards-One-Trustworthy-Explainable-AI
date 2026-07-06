# NOTES_next.md - live threads

Delete items as they close. This is "what were we about to do", not durable state
(durable state lives in CLAUDE.md).

## NOVELTY UPGRADE: attention generalization (STARTED 2026-07-06)

Reframe from "LRP for hierarchical ViTs" (extension, below CVPR novelty bar) to
"relevance propagation through ANY attention" (unifying framework, at the bar).
See CLAUDE.md "Attention as FOUR conserving primitives". DONE: attention_primitives.py
(toy fwd+lrp for self/cross/co/linear/channel/dilated) + tests/test_attention_coverage.py
(6 tests, machine-precision conservation, all green; full hilrp suite 22/22). Commit e541594.

Next, in priority order (each = one *_lxt.py real-model port + a conservation row + a
test, like the existing ports):
1. CROSS / CO-ATTENTION port (HIGHEST value): unlocks encoder-decoder + multi-modal
   (CLIP image-text, VQA/LXMERT, DETR). This is the Chefer ICCV'21 differentiator.
   Real model to target: a timm/HF CLIP ViT (cross-modal similarity scalar reuses the
   SSL-scalar machinery) or a DETR decoder. Relevance splits across two streams
   (already proven in the toy: test_cross_attention_conserves_and_splits_streams).
2. CHANNEL / COORDINATE attention port: SE-Net / CoordAttn / conv-attention hybrids.
   The gate is detached, relevance flows through the feature (gate_lrp). Target: a
   timm model with SE blocks (e.g. seresnet, efficientnet) or a CoordAttn hybrid.
3. DILATED / sparse attention: nearly free (Lemma-2 sparsity). Verify on NAT
   (neighborhood attention) or a LongNet-style dilated block.
4. DEFORMABLE attention (the one genuinely new rule): bilinear feature sampling at
   learned offsets. Target: Deformable DETR. Do last; proves even learned-offset
   attention conserves.
Also add an "attention coverage" table to the paper (type x primitive x done/proven/
todo) and cite the 15-type span in the intro/contributions.

## !! BLOCKER for A* scaling (2026-07-06): cache is only 100 images

`data/ImageNetS/cache_validation_100.pt` has 100 images. Lineage scripts
(`lineage_comparison.py`, `lineage_naive_legs.py`) are now set to N_IMAGES=1000
-> they will IndexError at image 100. To scale (the #1 A* lever), BUILD A
500-1000 IMAGE CACHE first: use `cache_imagenets.py` (see [[xai-bench-run-setup]]
for the flaky-HF-streaming gotcha; pre-cache locally). Then re-run pointing /
Shapley / lineage / conservation at n>=500 with 95% CIs + paired Wilcoxon vs
baselines. Until the cache exists, keep N_IMAGES<=100.

## SESSION 2026-07-06 done (paper additions, all committed <= fe604dc)

- GradCAM center-blob CONFIRMED (user's hypothesis): center-Gaussian blob also
  scores pointing 1.0; GradCAM corr 0.72/0.63 with it vs HiLRP 0.25/0.14. Paper
  sec:gradcam-blob.
- Gamma-selection ablation: gamma=0.25 robust no-tuning default. Paper sec:gamma,
  scripts/gamma_selection.py.
- MobileViT "blank map" = display bug (heavy-tailed), fixed hilrp/viz.py.
- 8-model HiLRP coverage complete (mean pointing 0.90); MaxViT ported; per-family
  attn_mode enforced in bench dispatch.
- RQ2 done (n=30): cross-objective agreement 0.33, DINO 0.97, MAE 0.07. Paper
  sec:rq2.
- IN FLIGHT: lineage_naive_legs on all 6 hierarchical/hybrid (bg, N_IMAGES was 100
  at launch) -> full LRP/AttnLRP/HiLRP lineage table. classic LRP=0 rows,
  AttnLRP-naive fails to conserve (swin 3.08x, others zeroed), AttnLRP-proper=flat
  ViT only, HiLRP conserves everywhere.

## 0. THE BIG CORRECTION (2026-07-04, evening): mobilevitv2 gap was an artifact

Healthy-model re-run (results/rerun_mobilevitv2/) shows grad_cam = 1.000 and
grad_cam++ = 1.000 on mobilevitv2_100. The original 0.49 "collapse" was 100%
preprocessing artifact. Gradient methods on the healthy model: 0.88-0.92.
HiLRP there: 0.860, competitive with gradients, below saturated CAM.

The REAL Grad-CAM failure is linear attention: efficientvit_b2 (clean stats)
grad_cam 0.551 (below the 0.61 random-point prior) vs HiLRP 0.960 over 100 imgs.
Gate 3 win condition landed THERE. CLAUDE.md findings and win-condition sections
already corrected. Never quote the mobilevitv2 0.49 again.

Paper framing consequence: the empirical gap claim names linear-attention
architectures. The mobilevit story is capability (stage maps, conservation,
defined rules) not pointing wins. Check efficientvit_b1 (grad_cam 0.70, clean
stats) as a second linear-attention data point.

## 1. Pointing metric is soft on this eval set (audit done)

Center-point prior = 0.97, mean bbox area fraction 0.65. Discriminating
alternatives already measured for the efficientvit leg: energy pointing 0.824 vs
0.656 uniform baseline (+0.168, 84/100 images above), small-box (<0.4 area)
subset pointing 0.94 (n=17). For the paper: lead with energy pointing +
ImageNet-S mask localization + small-box subset. The bench cache stores only
bboxes; masks need re-caching from ImageNet-S proper.

## 2a. Shapley PAPER-GRADE RESULTS (2026-07-05, n=50, m=64, two-pass honest protocol)

Three architectures, results/shapley_pilot/<model>/agreement.csv:

- swin_tiny:       hilrp +0.528 | grad_cam +0.515 | smoothgrad +0.312 | saliency +0.214 | ig +0.192
- pvt_v2_b2:       hilrp +0.531 | grad_cam +0.449 | ig +0.279 | smoothgrad +0.260 | saliency +0.137
- efficientvit_b2: hilrp +0.417 | smoothgrad +0.321 | ig +0.281 | saliency +0.240 | grad_cam +0.077

THE CROSS-YARDSTICK STORY CONFIRMED: HiLRP is highest on all three Shapley legs.
Grad-CAM agreement collapses on linear attention (0.515/0.449 -> 0.077),
mirroring its pointing collapse on efficientvit_b2 (1.0-ish -> 0.55), while
HiLRP remains best. Two independent yardsticks, one story. PVT is now the third
corollary leg: HiLRP +0.531 vs Grad-CAM +0.449, SNR 4.41.
Protocol note: the two-pass fix mattered: contaminated (patched-gradient)
baselines had DEPRESSED saliency/smoothgrad (0.08/0.12 -> honest 0.21/0.31);
HiLRP unchanged (0.528). Remaining scale-up: mobilevit (needs adapt_input
wiring), SAM segments, m=128, 100+ imgs if compute allows.

## 2. Shapley pilot: done, signal strong

10 imgs, m=32 perms, SLIC regions, swin_tiny (results/shapley_pilot/):
HiLRP mean Spearman vs sampled Shapley +0.579, gradient baseline +0.195,
HiLRP ahead on 9/10, SNR 2.5-4.8. Next: SAM segments (needs checkpoint infra),
more baselines (grad_cam, IG, Chefer), m=64-128, 50 imgs, then the 200-500 run.
Caveat noted in code: the saliency baseline runs through the patched forwards.

## 3. mobilevitv2 baseline re-run: DONE (results/rerun_mobilevitv2/summary.csv)

Healthy-model rows: grad_cam 1.00, grad_cam++ 1.00, smoothgrad/occlusion 0.92,
saliency/IG/vargrad/lime/rise 0.90, input_x_gradient 0.88. HiLRP 0.86.
Old vit_suite mobilevitv2 rows stay in combined_summary.csv as artifact
documentation; never quote them as findings.
Small fix needed: gradient_shap errors in the mri-diffuser env
("GradientShap.attribute() got an unexpected keyword argument
'internal_batch_size'", captum version drift). Guard the kwarg in
xai_bench/methods/captum_methods.py.

## 4. Conservation-trace diagnosis: MECHANISM FOUND, interpretation corrected (2026-07-05)

Two kill-tests on pvt_v2_b2 falsified the earlier "conv transitions absorb"
story:

- Gamma sweep (conv_gamma 0 / 0.25 / 1.0): decay identical at gamma=0 (plain
  epsilon). NOT a gamma effect.
- Per-module hooks inside stage3: downsample conv barely moves relevance
  (0.038 -> 0.083). The cliff is the STAGE-FINAL LayerNorm: block2-out +0.000
  vs stage3-out +1.007, separated only by the patched LN.

Mechanism candidate: LXT's layer_norm_forward keeps the mean-subtraction in the
graph; in Gradient x Input the per-token sum through it is
sum_c x_c g_c - gbar \* sum_c x_c, which cancels when features have large
channel means. The "identity rule" in grad-form is sum-preserving only for
near-centered features. Consequence: pre-LN and post-LN cuts disagree, so the
reported "decay below stage3" on PVT/hybrids is at least partly a
MEASUREMENT-CUT artifact, not a leak. Swin's capture points were all post-LN,
which is why they read ~1.0.

UPDATE (2026-07-05, session 2): the LN-cancellation mechanism was ALSO
falsified. The elementwise identity-LN swap left the cliff unchanged, and the
decisive finding is an algebraic impossibility: for the patched LN,
R_in + (beta . g) = R_out holds exactly for ANY g (verified in isolation to 4
decimals), but in the full model at stage3.norm the measured values are
0.000 + 0.008 vs 1.007. Identities cannot fail, so retain_grad measurements
inside the zennit/LXT composite context do not measure what they appear to
(suspect: LXT-patched BasicHook backward divides relevance by module input;
zennit graph rewiring may orphan retained tensors). ALL deep-cut traces
(pvt cliff, mobilevit sign flip, efficientvit inflation) are UNTRUSTED until
the measurement primitive is rebuilt. Trustworthy: head-adjacent cuts
(swin 1.00, mobilevit 1.05, pvt 1.01) and the float64 toy (6.5e-9, no zennit).

RESOLVED (2026-07-05, session 3): ROOT CAUSE = class-subclass patch trap.
timm's PVT norms are timm.layers.norm.LayerNorm, a SUBCLASS whose own forward
ignores the nn.LayerNorm class patch. Every norm in PVT ran UNPATCHED (full
gradient). Unpatched LayerNorm is scale- and shift-invariant, which forces
per-token x . grad == 0 EXACTLY: that is the entire "cliff". Both earlier
conclusions were wrong: never a leak, never a measurement artifact; the graph
simply was not executing the patched forward (the identity verified in
isolation applied to the patched LN; the model ran the unpatched one, whose
Jacobian differs). The elementwise identity-LN swap "changing nothing" is
explained the same way: also a no-op on the subclass.

Fix (pvt_lxt.py): TimmLayerNorm added to the patch map; audit_norm_patches()
guard flags any input-statistic norm whose class forward is not an LRP rule.
Audit across models: pvt 0 unpatched (after fix), mobilevitv2 0,
efficientvit_b2 0, swin 0 (those three were never affected).

Result: PVT traces are now smooth and monotone (stage3 1.01 -> stage2 0.94 ->
stage1 0.63 -> stage0 0.51 -> embed 0.39 -> pixels 0.33: graceful bias/gamma
absorption, no zeros, no flips); 10-img pointing probe 10/10. Full 100-img PVT
leg re-running; update the Gate 3 PVT row when it lands.

Still open from this thread: efficientvit's +11 stage-3 inflation (its norms
audit clean, so different mechanism; suspect the LiteMLA CP detached
normalizer). Re-measure with per-module hooks when it matters.

## 5. Gate 3 CLOSED (2026-07-05): full four-architecture table

PVT leg done (third corollary live, `pvt_lxt.py`, commit 4e396f4) and its
baselines done (`results/baselines_pvt/`; gradient_shap kwarg fix verified).
Final pointing table (HiLRP first, best baselines after):

- swin_base:       HiLRP 0.950 | grad_cam 1.00 | smoothgrad 0.79
- mobilevitv2:     HiLRP 0.860 | grad_cam 1.00 | smoothgrad 0.92 (healthy re-run)
- efficientvit_b2: HiLRP 0.960 | grad_cam 0.551 | smoothgrad 0.92 (outright win)
- pvt_v2_b2:       HiLRP 0.980 | grad_cam 0.840 | smoothgrad 0.92 (outright win;
  0.890 -> 0.980 after the norm-subclass fix, conservation 1.010 +/- 0.011)

Cross-architecture means: HiLRP 0.915 (highest, lowest variance),
smoothgrad 0.888, grad_cam++ 0.861, grad_cam 0.848. HiLRP is the only method
in the top tier on every architecture; CAMs fail on linear attention and are
mediocre on SRA; gradient methods swing 0.62-0.92 by architecture. Headline:
architecture-consistent reliability + the efficientvit win, not "beats CAM
everywhere".

## 5b. Equivariance thread STARTED (2026-07-05): theorem must be conditional

First measurement (scripts/equivariance_test.py, swin_tiny, cyclic shifts,
8 imgs) killed the naive theorem: even window-multiple cyclic shifts are NOT a
symmetry of real Swin (forward drift 7e-2), because the SW-MSA masks anchor to
the canvas. There is no exact input-space translation symmetry for masked Swin.

The honest, provable statement is CONDITIONAL: if the forward commutes with a
permutation of tokens/pixels, HiLRP commutes (Lemma 2 + weight sharing; the
rules use no absolute-position quantities). Empirics then quantify partial
symmetry transfer: at equal forward drift, HiLRP consistency ~0.90 vs saliency
0.41-0.61, and HiLRP degrades exactly when the forward does (unaligned shift:
drift 9.2e-2, rho 0.52). Attribution tracks the model, not the image.

Next: (a) idealized cyclic-Swin (remove masks, cyclic window attention) where
the premise holds exactly: verify rho -> 1.0 numerically, validating the
mechanism; (b) write the conditional theorem in LaTeX (reuse Lemma-1
sum-exchange); (c) extend the consistency table with grad_cam + smoothgrad and
more images.

## 5c. SSL-scalar branch LANDED (2026-07-05): label-free attribution works

vit_lxt.py (plain-ViT port, simplest of the family) + attribute_ssl_similarity:
backward from cos(cls(x), cls(flip(x)).detach()) with the norm denominator
DETACHED (cosine is scale-invariant; an in-graph norm zeroes relevance sums by
the same invariance mechanism as an unpatched LayerNorm: this is now a design
rule: every scalar head needs its own identity treatment for normalizers).

Demo (scripts/ssl_scalar_demo.py, ViT-S/16 DINO vs supervised AugReg, 12 imgs):
- label-free pointing 0.92 for BOTH pretrainings (no labels anywhere)
- conservation to pixels differs by pretraining: supervised 0.91 +/- 0.04 vs
  DINO 0.37 +/- 0.04 of the scalar: a mechanistic RQ2 observation (where the
  view-invariance evidence is absorbed differs by objective)
- map structure differs visibly: DINO traces object geometry, supervised shows
  patch-grid artifacts. Figure: results/ssl_scalar/ssl_similarity_maps.png

RQ2 scale-up next: more images, map-correlation metrics between pretrainings,
MAE reconstruction scalar, DINOv2 (needs 518px or interpolated pos-embed).

## 5d. LINEAGE COMPARISON DONE (2026-07-05): LRP / AttnLRP vs HiLRP

scripts/lineage_comparison.py, 100 imgs, pointing + conservation at deep/head cuts.

Leg A, vit_base (AttnLRP's home turf):
- AttnLRP (inherited, attn_mode flag): pointing 0.770, cons deep +0.15
- HiLRP-CP (our default):              pointing 0.670, cons deep +0.61
- bench: Chefer-proxy 0.740, rollout 0.640, saliency 0.650, grad_cam 0.927
- captum LRP: nonfunctional on timm transformers (0 rows)

Reading: on flat CLS-pooled ViTs the attention path carries localization, so
the attnlrp mode points better and CP conserves 4x more. NEW PER-FAMILY
DEFAULT: attn_mode='attnlrp' for flat CLS ViTs, 'cp' for windowed/hierarchical
(where CP won both pointing AND conservation). Flat ViT is parity-with-lineage
territory (0.77 vs Chefer 0.74), not a win claim; grad_cam still tops soft
pointing there (0.927).

Leg B, swin_base (our turf):
- HiLRP:        pointing 0.950, cons deep +0.95 +/- 0.06
- AttnLRP-naive (flat recipe, vanilla window attention):
                pointing 0.980, cons deep +3.08 +/- 0.78

Reading: pointing does NOT discriminate on this saturated metric (3 pts).
Conservation DOES: the naive adaptation inflates deep relevance 3.1x with high
variance, so its magnitudes are meaningless, stage decompositions invalid, and
no guarantee exists. Plus the norm-subclass trap: a naive user on PVT gets
silently zeroed relevance sums (we measured 0.89 -> 0.98 pointing from our
guard alone). THE LINEAGE CLAIM, sharpened: HiLRP = AttnLRP's rules where they
exist, plus guarantees and guards where they do not.

COMPLETE LINEAGE TABLE (naive legs done 2026-07-05, scripts/lineage_naive_legs.py,
100 imgs, generous naive = canonizer + correct stats):

model            | naive point | naive deep-cons | HiLRP point | HiLRP deep-cons
vit_base (flat)  | 0.770 (=AttnLRP) | +0.15   | 0.770 (attnlrp mode) | +0.61
swin_base        | 0.980       | +3.08 +-0.78    | 0.950       | +0.95
pvt_v2_b2        | 0.800       | 0.00 (zeroed)   | 0.980       | +0.39
efficientvit_b2  | 0.930       | 0.00 (zeroed)   | 0.960       | traceable
mobilevitv2      | 0.950       | 0.00 (zeroed)   | 0.860       | +1.05 head

THE DISCRIMINATOR IS DEEP CONSERVATION, not pointing. On every hierarchical/
hybrid model the naive extension's relevance FAILS TO CONSERVE to the
prediction: either 3x-inflated (Swin) or driven to EXACTLY ZERO at the
input-adjacent cut (pvt/efficientvit/mobilevit, from unpatched scale-invariant
norms). A zero-sum map still has an argmax, so it can "point" (naive mobilevit
0.95 > HiLRP 0.86), but it violates the completeness property that is LRP's
entire theoretical justification: the region contributions do not sum to the
logit. HiLRP is the ONLY conservation-valid method on these architectures.
HONEST FRAMING: superiority = conservation-validity + guards, NOT pointing (which
is mixed and on a soft 0.97-center-prior metric). Never claim pointing wins on
mobilevit. Artifact: paper/lineage_table.md (tracked; results/ is gitignored).

## 5e. hilrp quantus grid: DONE (empty_cache fix held, results/hilrp_grid/summary.csv)

Full grid, 50 imgs, 4 architectures. faith_corr / apf / faith_est / pointing /
max_sens / sparseness:
- pvt_v2_b2       0.053 / 0.028 / -0.226 / 0.98 / 0.44 / 0.62
- efficientvit_b2 -0.023 / -0.020 / 0.027 / 0.96 / 1.12 / 0.66
- swin_base       0.008 / -0.008 / -0.198 / 0.96 / 0.51 / 0.58
- mobilevitv2     0.012 / 0.017 / 0.006 / 0.78 / 8.51 / 0.77

Reading: (1) faithfulness_correlation ~0 for HiLRP on all four - but this is the
METRIC failing (established plank 3: faith_corr is noise for EVERY method,
|mean|<0.08); do not read it as an HiLRP weakness, read it as why Shapley is the
primary yardstick. (2) pointing strong (0.78-0.98, consistent with gate3).
(3) max_sensitivity (robustness, lower better): HiLRP stable on pvt/swin/
efficientvit (0.44-1.12) but UNSTABLE on mobilevitv2 (8.51) - honest robustness
caveat for the conv-hybrid, worth a sentence. (4) HiLRP now sits in the bench
table beside all baselines. CUDA fix = empty_cache + zero_grad per explain call
in hilrp_method (committed ef0bf40).

## 5f. RQ2 DONE (2026-07-05): pretraining objective determines explanation

scripts/rq2_pretraining.py. Fixed ViT-B/16, 4 pretrainings (supervised/DINO/
MAE/CLIP), one common label-free scalar cos(cls(x),cls(flip(x))). n=30:
- cross-objective map agreement (mean pairwise Spearman) = 0.33 (vs 1.0 if
  objective irrelevant). MAE the outlier (0.17-0.18 with others).
- label-free pointing: DINO 0.97 (matches known object-emergence, no labels!),
  supervised 0.60, MAE 0.50, CLIP 0.37.
- pixel-relevance: MAE 0.07 (reconstruction objective, ~no relevance to pixels),
  others 0.39-0.57.
Figure results/rq2_pretraining/rq2_maps.png (DINO traces geometry, MAE patch-
grid, CLIP sparse). In paper as sec:rq2. TO SCALE FOR A*: n>=100, significance
test on agreement, second architecture (ViT-S), maybe MAE-reconstruction scalar.

## 5h. MobileViT "blank heatmap" RESOLVED (2026-07-06): display, not method

User reported near-blank MobileViT maps. Diagnosis: NOT blank - heavy-tailed
(~27% of |mass| in top 1% of pixels, from the deep separable-conv stem), so
symmetric-max normalization washes it out. Map localizes fine (pointing 0.86).
Fix: hilrp/viz.py:normalize_for_display (99pct clip + optional post-hoc smooth),
wired into gate2_real_swin show_relevance (commit b058e52). With it the maps
clearly trace object geometry (yurt lattice, hut silhouette). conv_gamma sweep
does NOT change spikiness (it's inherent to the conv pixel-gradient). USE THIS
HELPER in every viz script; MobileViT is the worst case but all benefit.
Figure: results/hilrp_gate2real/mobilevit_qualitative.png.

## 5g. 10-MODEL HiLRP coverage COMPLETE: MaxViT ported, per-family attn_mode fixed

DONE. HiLRP quantus grid on all 8 ViT-family benchmark models (50 imgs),
pointing_game: vit_base 0.78, deit_base 0.84, swin_base 0.96, pvt_v2_b2 0.98,
efficientvit_b1 0.94, efficientvit_b2 0.96, maxvit_small 0.96, mobilevitv2 0.78.
Mean 0.90. (results/hilrp_grid + hilrp_grid_rest2). CNNs resnet/convnext are
baselines-only (HiLRP-for-ViT not the point). Fold into a full benchmark table.

MaxViT port (maxvit_lxt.py): fwd 4.5e-7, pointing 10/10 smoke, wired into bench
+ gate3 dispatch. CRITICAL FIX: bench hilrp dispatch now binds
attn_mode='attnlrp' for flat vit/deit (was defaulting to cp -> vit_base 0.52;
attnlrp -> 0.78, matching lineage 0.77). rest2 run (vit/deit/efficientvit_b1/
maxvit) in results/hilrp_grid_rest2/. Combined with hilrp_grid (swin/pvt/
efficientvit_b2/mobilevit), HiLRP now covers all 8 ViT-family benchmark models.

## A* PUSH PLAN (reframed 2026-07-05): benchmark + method + metric critique

Contribution is A*-shaped if framed correctly:
benchmark across ViT families + metric-failure diagnosis + HiLRP as the
conservation-valid resolution for custom hierarchical/hybrid ViTs + Shapley
validation + RQ2 pretraining-objective finding.

Do NOT frame as "HiLRP wins every metric." The correct thesis is: XAI behavior
is architecture-dependent, common metrics can fail, and domain/custom ViTs need
a principled fallback that still gives a conservation-valid explanation path.
HiLRP is coverage + conservation + extensibility, not universal visual dominance.

If stuck at n=100:
1. Keep all comparisons paired on the same images.
2. Add paired bootstrap 95% CIs and paired Wilcoxon tests.
3. Claim only large effects: efficientvit_b2 HiLRP 0.960 vs grad_cam 0.551 is
   claimable; 0.96 vs 0.92 is not.
4. Keep Shapley focused: n=50-100, Swin/PVT/EfficientViT, key baselines only.
5. Make the compute limit explicit and methodological, not hidden.

If scaling to n=1000:
1. Scale only the cheap large-tier metrics: pointing, sparseness, selected
   robustness, and faithfulness-correlation as a metric-failure diagnostic.
2. Use 5-6 key methods, not every slow method: HiLRP, Grad-CAM, SmoothGrad,
   Integrated Gradients, Saliency, Attention Rollout where applicable.
3. Keep Shapley at n=50-100 or n=100 max unless there is spare compute.
4. Do not run LIME/RISE/occlusion/GradientShap over 1000 unless a reviewer asks.

Paper-quality priorities:
1. Rewrite intro around custom/domain ViTs creating recurring XAI support gaps.
2. Promote the existing benchmark as evidence of architecture-dependent XAI,
   not as a failed universal leaderboard.
3. Replace placeholder qualitative figures with real stage maps, lineage failure
   maps, RQ2 maps, and Shapley agreement plots.
4. Clean all overclaims: no exact pixel conservation, no mobilevit Grad-CAM
   collapse claim, no universal metric win claim.
5. Add CIs/significance to existing n=50-100 tables before running more models.
6. Improve HiLRP where weak: MobileViT robustness gamma sweep first.

## 6. Then (order per CLAUDE.md)

- Rebuild conservation-trace measurement primitive (thread 4, blocks paper
  conservation table).
- Equivariance theorem + consistency metric.
- SSL-scalar branch (dinov2/mae/clip).
- Register hilrp in the bench METHODS registry so the standard runner computes
  the full quantus grid for it (deletion AUC alongside, with the caveat).
- Shapley scale-up (SAM segments, more baselines, 50 imgs then 200-500).
- ImageNet-S mask re-cache for hard localization.

## Environment

mri-diffuser conda env for everything:
C:\Users\nisha\miniconda3\envs\mri-diffuser\python.exe
(torch 2.6.0+cu124 CUDA, timm 1.0.27, lxt 2.1, zennit, quantus, captum, datasets).
pip install lxt needs PYTHONUTF8=1. Call env python by path, conda run mangles
output. IDE diagnostics point at the wrong interpreter, ignore missing-module
errors for timm/lxt/pytest. Commits: 7b57eeb, d45250f, b6fbdae, c603d91.
