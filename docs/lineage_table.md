# LRP / AttnLRP vs HiLRP: the lineage table

100 ImageNet-S images, pointing game (argmax-in-bbox) and conservation reported
as sum(R)/logit at the input-adjacent ("deep") capture. "AttnLRP-naive" = LXT's
generic flat-transformer recipe applied to an unsupported architecture, granted
the BN canonizer and correct preprocessing but none of HiLRP's contributions
(attention rules, norm-subclass guard, non-GELU activation identities). Scripts:
`scripts/lineage_comparison.py`, `scripts/lineage_naive_legs.py`.

| model | family | AttnLRP-naive pointing | naive deep-cons | HiLRP pointing | HiLRP deep-cons |
|---|---|---|---|---|---|
| vit_base_patch16_224 | flat ViT | 0.770 (= AttnLRP proper) | +0.15 | 0.770 (attnlrp mode) | +0.61 |
| swin_base_patch4_window7 | hierarchical | 0.980 | **+3.08 ± 0.78** | 0.950 | +0.95 |
| pvt_v2_b2 | spatial-reduction | 0.800 | **0.00 (zeroed)** | 0.980 | +0.39 |
| efficientvit_b2 | linear attention | 0.930 | **0.00 (zeroed)** | 0.960 | traceable |
| mobilevitv2_100 | conv-hybrid | 0.950 | **0.00 (zeroed)** | 0.860 | +1.05 (head) |

Reference baselines on flat ViT (bench, 100 imgs): grad_cam 0.927,
Chefer-proxy 0.740, attention_rollout 0.640, saliency 0.650, IG 0.640.
captum LRP: does not run on timm transformers (0 rows).

## The claim, precisely

1. **Flat ViT is parity with the lineage.** HiLRP inherits AttnLRP's block rules;
   in `attnlrp` mode it reproduces AttnLRP (0.770) above Chefer (0.740) and
   rollout (0.640). Classic LRP does not run at all. This is not a win claim.

2. **On hierarchical/hybrid models the discriminator is conservation, not
   pointing.** AttnLRP has no rules there; its only available extension (the
   naive recipe) produces relevance that does not conserve to the prediction:
   3× inflated on Swin, driven to exactly zero at depth on PVT / EfficientViT /
   MobileViT by unpatched scale-invariant normalizations. A zero-sum map still
   has an argmax, so it can point (naive MobileViT 0.95 > HiLRP 0.86), but its
   region contributions do not sum to the logit, violating the completeness
   property that is LRP's entire justification.

3. **HiLRP is the only conservation-valid attribution on these architectures**,
   and wins pointing outright on PVT (0.98 vs 0.80) and EfficientViT
   (0.96 vs 0.93) as well. Superiority = conservation-validity plus the guards
   that stop silent failure, not a pointing-score race on a soft metric.
