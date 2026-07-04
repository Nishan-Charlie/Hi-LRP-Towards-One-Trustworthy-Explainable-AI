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
sum_c x_c g_c - gbar * sum_c x_c, which cancels when features have large
channel means. The "identity rule" in grad-form is sum-preserving only for
near-centered features. Consequence: pre-LN and post-LN cuts disagree, so the
reported "decay below stage3" on PVT/hybrids is at least partly a
MEASUREMENT-CUT artifact, not a leak. Swin's capture points were all post-LN,
which is why they read ~1.0.

UPDATE (2026-07-05, later): the elementwise identity-LN swap (custom Function,
grad_x = (y/x)*grad_y) did NOT flatten the trace: stage3 ~1.0, stage2 ~0.05,
identical cliff. So the LN-cancellation hypothesis is falsified or incomplete;
with elementwise pass-through the drop must sit INSIDE the stage3 blocks, which
the probe did not capture. Status: cliff is robust to gamma (0/0.25/1.0), conv
rules, and both LN rules. Two hypotheses proposed and killed by their own tests
(conv absorption, LN mean-subtraction). Remaining candidates: residual-split
semantics in grad x input, CP q-path zeroing interacting with Gamma hooks on
q/kv Linears, GELU-identity on large activations.

SESSION 2 RESULTS (2026-07-05, four more falsifications, then an impossibility):
- LN patch verified applied (not a silent no-op).
- zero_params='bias' on Gamma explodes (1e22): biases in denominators are
  load-bearing stabilizers; bias-share redistribution is not the fix.
- Final-LN beta share of the logit is ~0 (w_t . beta + b_t = -0.005 vs logit 9).
- THE KEY FINDING: an algebraic identity is violated inside the composite
  context. For the patched LN, R_in + (beta . g) = R_out holds EXACTLY for any
  g (verified in isolation to 4 decimals, any input scale). In the full model
  at stage3.norm: R_out = +1.007, beta share = +0.008, but measured
  R_in = 0.0000. Identities cannot fail; therefore retain_grad measurements
  inside the zennit/LXT composite context are not measuring what they appear
  to (suspects: LXT-patched BasicHook backward divides relevance by module
  input; zennit forward-hook graph rewiring may orphan retained tensors).
  Head path verified plain (mean pool + Gamma-hooked Linear), stage source
  verified (LN input IS the last block output).

Consequence: ALL deep-cut conservation traces measured so far (the pvt/hybrid
"cliff", possibly the mobilevit sign flip and efficientvit inflation) are
UNTRUSTED until the measurement mechanics are fixed. The head-adjacent cut
(one plain-autograd hop from the hooked head) remains trustworthy: swin 1.00,
mobilevit 1.05, pvt 1.01. Our own toy (relevance-direct, no zennit) remains
exact at 6.5e-9.

Next action (surgical, next session): minimal repro = stage3.norm +
reshape/permute/contiguous + mean pool + Gamma-hooked Linear on real tensor
values; measure the same three quantities; bisect ops (drop contiguous, drop
hook, swap hook for plain linear) until the identity violation appears/vanishes.
Then rebuild the trace tooling on whatever measurement primitive survives
(likely: capture relevance INSIDE hook backwards, not via retain_grad).

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
