# NOTES_next.md — live threads

Delete items as they close. This is "what were we about to do", not durable state
(durable state lives in CLAUDE.md).

## 0. Bench contamination: re-run mobilevitv2 baseline rows (blocks the money table)

Found 2026-07-04: the bench hardcodes ImageNet normalization for ALL models
(xai_bench/datasets/imagenet_s.py, imagenet_val.py, utils/image.py), but
mobilevitv2_100 expects RAW [0,1] inputs (timm default_cfg mean 0, std 1). Its
baseline rows in results/combined_summary.csv ran on out-of-distribution inputs
(top-1 was ~0/20 on the cache). The grad_cam 0.49 "collapse" is contaminated.
efficientvit_b2 expects ImageNet stats, so its 0.55 row is clean and carries the
hybrid-gap finding meanwhile.

Fix: per-model preprocessing (see adapt_input in scripts/gate3_pointing.py),
then re-run the vit_suite mobilevitv2 rows. Until then, any HiLRP-vs-baseline
table on mobilevitv2 must carry the caveat.

## 1. Gate 3 status: pointing legs done, table pending fair baselines

Ran 2026-07-04 (scripts/gate3_pointing.py, 100 cached ImageNet-S imgs, gamma=0.25,
CP, per-model input adaptation, results/hilrp_gate3/):

- swin_base: HiLRP 0.950. Beats 9 of 11 baselines; grad_cam and grad_cam++ are
  saturated at 1.0 on this model (parity leg, as expected).
- mobilevitv2_100: HiLRP 0.860, conservation at last capture 1.051 +/- 0.012.
  Nominally beats all 11 baseline rows including grad_cam 0.49, BUT those rows
  are contaminated (item 0). Honest claim until re-run: HiLRP localizes well
  (0.86) on the healthy model; the comparison table needs clean baselines.

Remaining for Gate 3 close:
- efficientvit_b2 port (clean baseline row 0.55 is the real target now).
  Sized: BN-only (canonizer for efficientvit_mit ConvNormAct pairs), Hardswish
  identity patch, ReLU conserves natively in Gradient x Input, LiteMLA needs the
  CP analog. Comparable effort to the mobilevit port.
- maxvit leg optional after that.
- deletion AUC alongside (reviewers expect it; never as the discriminating signal).
- Register hilrp as a bench METHODS entry so the standard runner computes the
  full quantus grid for it.

## 2. MobileViTv2 port lessons (do not relearn)

- Zennit AlphaBeta / ZPlus are INCOMPATIBLE with the LXT zennit bridge
  (monkey_patch_zennit assumes the mul/div Gradient x Input hook shape;
  multi-term rules explode to 1e18). Bridge-compatible family: Gamma, Epsilon.
- Pure Gamma on the conv stack inverts total sign at the stage3 -> stage4
  boundary conv (signed inputs). BN merge and boundary-epsilon do not fix it.
  With correct preprocessing the maps localize well anyway; pixel-sum residual
  is small (-0.11 +/- 0.09). Documented, not hidden. Token-level conservation at
  transformer stages is the guaranteed quantity (mobilevit last capture 1.05).
- Detaching the whole context_vector in separable attention kills all
  cross-token relevance routing (pointing drops below floor). Detach only
  context_scores; divide_gradient(out, 2) on the value*context mul.

## 3. Conservation-vs-logit diagnostic: done, benign for current metrics

corr(logit, conservation) ~ -0.42 (mild monotone trend). Labels 44/41 sit
~1.2 sigma above the tokens trend line (+0.119 vs -0.040, n=5): mild
class-dependent gamma inflation, order 10%. Our class-sensitivity metrics are
correlation-based (shift and scale invariant), so immune. Rule going forward:
per-map normalization for any cross-class MAGNITUDE claim. Re-check on a larger
sample when the Shapley eval set exists.

## 4. Shapley reference (start in background NOW, it's the slow one)

Unchanged: sampled permutation Shapley over SAM segments, pilot 50 imgs, then
200-500. PRIMARY evidence track. Needs SAM checkpoint + segment cache
infrastructure. Do not let it become the week-4 bottleneck.

## 5. Then (after Gate 3 closes)

- PVT spatial-reduction rule (third corollary).
- Equivariance theorem + consistency metric.
- SSL-scalar branch (dinov2/mae/clip).

## Environment

mri-diffuser conda env for everything:
C:\Users\nisha\miniconda3\envs\mri-diffuser\python.exe
(torch 2.6.0+cu124 CUDA, timm 1.0.27, lxt 2.1, zennit, quantus, captum, datasets).
pip install lxt needs PYTHONUTF8=1 on Windows. Call the env python by path,
conda run mangles output. IDE diagnostics point at the wrong interpreter
(Python312 AppData): ignore its missing-module errors for timm/lxt/pytest.

Checkpoint commit: 7b57eeb (hilrp package, ports, gate scripts, dataset fix,
CLAUDE.md, this file). Gate results in results/ are gitignored, evidence lives
on disk and in CLAUDE.md numbers.
