# NOTES_next.md — live threads

Delete items as they close. This is "what were we about to do", not durable state
(durable state lives in CLAUDE.md).

## 0. Commit the checkpoint (do first)

Nothing is committed yet: hilrp package, swin_lxt, gate scripts, the lazy-import fix in
imagenet_s.py. Commit as a green checkpoint before Gate 3. Include the lazy-import fix in
the same commit (load-bearing for the harness to run).

## 1. Pixel-conservation-vs-logit diagnostic (30 min, gates class-sensitivity work)

On the 20 gate2real images, plot pixel-conservation against logit magnitude.
Question: idx 15-19 (labels 44/41) run consistently hotter (tokens 1.09-1.28, stage3 up
to 1.43, pixels up to 0.98) than the label-915 images (tokens ~0.84-1.14). Two classes,
two conservation profiles.

- If clean monotone relationship vs logit -> benign (gamma-rule x activation stats), note
  and move on.
- If 44/41 sit off the trend line -> gamma mass inflation is class-dependent, dig in
  BEFORE building class-sensitivity results on top of it.
  Sanity already confirmed: idx 2 (853->619 wrong) and idx 18 (44->41 wrong) have normal
  conservation profiles, so conservation holds regardless of correctness. Good.

## 2. Gate 3 (after diagnostic)

Runner pulls already-cached baselines from the bench, adds hilrp as a registered method.
Report BOTH:

- deletion AUC vs Chefer / Grad-CAM / rollout (reviewers expect it), AND
- pointing-game on Swin-T AND the hybrid/linear-attention models (mobilevitv2,
  efficientvit_b1/b2, maxvit) where Grad-CAM collapsed to 0.49-0.55.
  The second set is the money comparison. Headline: "HiLRP holds pointing-game on exactly
  the architectures where the CNN method breaks."
  Framing: token-level conservation = guaranteed quantity; pixel-level = controlled
  gamma-deviation. Do not report deletion AUC as the discriminating signal (our bench proved
  it can't discriminate).

## 3. Shapley reference (start in background NOW, it's the slow one)

Generate sampled permutation Shapley on 50 images first (pilot), then scale to 200-500.
Over SAM segments. This is the PRIMARY evidence track (agreement with axiomatic method).
Each image = hundreds-to-thousands of forward passes, so it dominates compute budget.
Do not let it become the week-4 bottleneck.

## 4. Then (after Gate 3 passes)

- PVT spatial-reduction rule (third corollary).
- Equivariance theorem + consistency metric.
- SSL-scalar branch (dinov2/mae/clip).
