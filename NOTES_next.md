# NOTES_next.md — live threads

Delete items as they close. This is "what were we about to do", not durable state
(durable state lives in CLAUDE.md).

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

## 2. Shapley pilot: done, signal strong

10 imgs, m=32 perms, SLIC regions, swin_tiny (results/shapley_pilot/):
HiLRP mean Spearman vs sampled Shapley +0.579, gradient baseline +0.195,
HiLRP ahead on 9/10, SNR 2.5-4.8. Next: SAM segments (needs checkpoint infra),
more baselines (grad_cam, IG, Chefer), m=64-128, 50 imgs, then the 200-500 run.
Caveat noted in code: the saliency baseline runs through the patched forwards.

## 3. mobilevitv2 baseline re-run: finishing (~420/550 at last check)

Runner writes results.csv + summary.csv on completion. When done: fold the
corrected rows into the money table, mark the old vit_suite mobilevitv2 rows
contaminated (do NOT delete: they document the artifact), commit.

## 4. Conv-stack conservation diagnosis thread (before paper)

Bridge-compatible Gamma on deep conv stacks: sign flip at a signed-input
boundary conv (mobilevit), 12x stage-3 inflation + pixel decay (efficientvit,
last-capture trace +11 +/- 8). Localization unaffected. Options to explore:
bridge-compatible alpha-beta as a custom Function, per-layer gamma schedules,
or scoping pixel maps as visualization-only for conv-heavy hybrids. Token-level
claims at transformer stages are unaffected (swin 1.00, mobilevit 1.05).

## 5. Then (order per CLAUDE.md)

- PVT spatial-reduction rule (third corollary).
- Equivariance theorem + consistency metric.
- SSL-scalar branch (dinov2/mae/clip).
- Register hilrp in the bench METHODS registry so the standard runner computes
  the full quantus grid for it (deletion AUC alongside, with the caveat).

## Environment

mri-diffuser conda env for everything:
C:\Users\nisha\miniconda3\envs\mri-diffuser\python.exe
(torch 2.6.0+cu124 CUDA, timm 1.0.27, lxt 2.1, zennit, quantus, captum, datasets).
pip install lxt needs PYTHONUTF8=1. Call env python by path, conda run mangles
output. IDE diagnostics point at the wrong interpreter, ignore missing-module
errors for timm/lxt/pytest. Commits: 7b57eeb, d45250f, b6fbdae, c603d91.
