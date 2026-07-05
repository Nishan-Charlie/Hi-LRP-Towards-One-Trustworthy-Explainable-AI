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
