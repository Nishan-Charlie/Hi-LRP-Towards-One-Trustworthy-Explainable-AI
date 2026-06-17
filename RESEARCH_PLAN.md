# Research Plan — Does Explainability Transfer?

**A Controlled Benchmark of Attribution Methods Across Architectures, Pretraining Objectives, and Scales for Vision Foundation Models**

*Type:* Empirical benchmark / large-scale evaluation study (no new attribution method).
*Target tier:* Q1 journal.
*Status:* Planning.
*Last updated:* 2026-06-17.

---

## 1. One-paragraph summary

Almost every "which attribution method is best" conclusion in explainable AI was established on
small CNNs and never re-validated on the Vision Transformer (ViT) foundation models that now
dominate practice. This work delivers a controlled, statistically rigorous benchmark that measures
attribution **fidelity** across three orthogonal axes — *architecture* (CNN vs ViT), *pretraining
objective* (supervised, self-supervised, masked, contrastive), and *model scale* (small → giant) —
using both **exact/causal ground truth** (synthetic and intervention-based) and **proxy faithfulness
metrics**. The contribution is the benchmark, the released harness, and three new empirical findings
(transfer, pretraining effect, metric disagreement on ViTs) — not a new explanation method.

---

## 2. Motivation and background

### 2.1 Reference point — Brandt, Raatjens & Gaydadjiev (2023)
*"Precise Benchmarking of Explainable AI Attribution Methods"* hand-builds a tiny CNN by **setting
weights manually** (no training), so the exact ground-truth attribution is known by back-tracking
influence through the network. They propose precision/recall-style metrics (Compactness,
Completeness, Correctness), separated by **positive vs negative** contributions, and benchmark 14
attribution methods. Key results: exact GT with no annotation bias; methods are decent on
positively-contributing pixels but **poor on negatively-contributing ones**; Integrated Gradients
performs best overall; Grad-CAM(++) fails on certain concept structures; their metrics are fast.

### 2.2 Why a new benchmark is needed
The reference paper's self-admitted limits are precisely the research gaps: a **toy 36×36 CNN**,
**CNNs only**, **no real models or data**, **a single scale**, and **no statistical/ranking
analysis**. The wider field has split into two camps that never meet:

- **Exact-ground-truth benchmarks** (Brandt; CLEVR-XAI; FunnyBirds; synthetic XAI-Bench) — rigorous
  but only on toy/CNN models.
- **Proxy-faithfulness benchmarks** (Quantus; deletion/insertion; MuFidelity; ROAD) — scale to real
  models but have no ground truth and disagree with one another.

Meanwhile practice has moved to **ViT foundation-model backbones** (DINOv2, CLIP, MAE, BEiT3, EVA),
yet CNN-era attribution rankings have not been re-tested there.

---

## 3. Gap statement

1. **No controlled fidelity benchmark across architectures** (CNN vs ViT) at matched scale/accuracy —
   it is unknown whether CNN-era method rankings transfer to transformers.
2. **The pretraining objective is an unexamined confound.** The same ViT architecture pretrained
   supervised vs self-supervised (DINOv2) vs masked (MAE/BEiT) vs contrastive (CLIP/EVA) may be
   differently explainable; this variable has never been isolated.
3. **No metric meta-evaluation on ViTs.** Which faithfulness metric is chosen may decide the winning
   method; this has not been quantified for transformers.

---

## 4. Research questions

- **RQ1 — Transfer.** Do attribution-method fidelity rankings established on CNNs hold on ViT
  foundation models?
- **RQ2 — Pretraining.** Holding architecture and scale fixed, does the pretraining objective change
  attribution faithfulness?
- **RQ3 — Scaling.** How does fidelity scale with model size? Is there an "explainability scaling law"?
- **RQ4 — Attention vs the rest.** Do attention-native methods (rollout, Chefer/LRP) beat
  gradient/perturbation methods on ViTs, and do **register tokens** fix attention-based attribution?
- **RQ5 — Metric agreement.** How much do faithfulness metrics agree with each other and with
  controllable ground truth?

---

## 5. Benchmark design

### 5.1 Models (backbones)
ViT foundation models from `timm`, spanning pretraining objective and scale:

| Group | Pretraining | Variants (scale) |
|---|---|---|
| DINO / **DINOv2** | Self-supervised (distillation) | small / base / large / giant; **±register tokens** (`reg4`) |
| **MAE** | Masked image modeling | base / large / huge |
| **BEiT3** | Masked + multimodal | base / large |
| **CLIP-ViT / EVA02-CLIP** | Contrastive (image-text) | base / large / enormous |
| Supervised ViT (`*_ft_in1k`) | Supervised | base / large |
| **CNN baselines** | Supervised | ResNet-50, ConvNeXt (matched by params/accuracy) |

*Natural experiments:* (a) DINOv2 ±`reg4` isolates the register-token effect on attention attribution;
(b) supervised vs self-supervised at the same architecture/scale isolates the pretraining effect.

### 5.2 Attribution methods (under test — all existing, no novelty)
- **Gradient:** Saliency, Integrated Gradients, Gradient×Input, SmoothGrad, VarGrad, SquareGrad.
- **CAM:** Grad-CAM, Grad-CAM++ (with ViT activation reshape).
- **Attention-native:** Attention Rollout, Attention Flow, Chefer transformer-attribution / LRP.
- **Perturbation:** Occlusion, RISE, LIME, KernelSHAP.
- **Composite (already prototyped):** Grad-CAM × Integrated Gradients.

### 5.3 Datasets and ground truth
| Tier | Dataset | GT type | Role |
|---|---|---|---|
| 1 | **ImageNet-S** (919/300/50) | Pixel segmentation masks | Primary real-data localization GT (pointing game, IoU, EBPG, RRA) |
| 1 | **ImageNet-1k val + bbox** | Bounding boxes (~50k) | Large-scale localization → statistical power |
| 1 | **FunnyBirds** | Synthetic parts + **causal intervention** protocol | Exact/causal GT arm with ready-made XAI protocol |
| 2 | **PartImageNet** (158 cls) | Part segmentation | Fine-grained "right part" test (echoes concept-parts) |
| 2 | **MS COCO / PASCAL VOC seg** | Instance/semantic masks | Multi-object grid pointing game (stress test) |
| 2 | **BAM** | Object-on-background = known GT | Cheap controllable object-vs-context GT |
| 3 (opt) | CUB-200-2011 | Parts + boxes + attributes | Fine-grained (needs fine-tuning) |
| 3 (opt) | CLEVR-XAI | Exact relevance masks | Reasoning/VQA setting |
| 3 (opt) | ImageNet-C/R/A/Sketch | None | Attribution fidelity under distribution shift |

**Recommended default stack:** ImageNet-S + ImageNet-bbox + FunnyBirds + COCO grid-pointing.

### 5.4 Evaluation metrics (standardized via **Quantus**)
- **Faithfulness:** Deletion AUC, Insertion AUC, Faithfulness Correlation, MuFidelity, **ROAD**
  (distribution-shift-robust alternative to ROAR).
- **Localization (vs masks/boxes):** Pointing Game, Energy-Based Pointing Game (EBPG), Relevance Rank
  Accuracy (RRA), IoU, Precision/Recall/F1.
- **Causal (FunnyBirds):** part-removal controllability, single/multi-target deletion.
- **Robustness:** Max-Sensitivity, SSIM under input noise.
- **Complexity/sparseness:** Gini coefficient, entropy.
- **Sanity checks:** Adebayo model-parameter randomization and label randomization (pass/fail).
- **Cost:** wall-clock ms/explanation (with and without GPU).

> Replace the hand-rolled metric code in `foundation_model_xai.ipynb` with Quantus implementations so
> reviewers trust correctness; keep the notebook as the prototype reference.

### 5.5 Classifier-head handling (required for class-conditioned attribution)
- **Supervised / `*_ft_in1k`** → use directly.
- **DINOv2 / MAE / BEiT3** → frozen-backbone **linear probe** to obtain a class logit.
- **CLIP / EVA-CLIP** → **zero-shot** text-prompt logits (bonus: enables text→image attribution).
- Treat the readout head as an explicit controlled variable (does fidelity depend on head vs backbone?).

---

## 6. Methodology notes

- **Normalization:** per-explanation normalization to a common range (as in the reference paper) so
  unbounded methods (e.g. raw gradients) are comparable; document the procedure and guard against
  outlier-driven distortion.
- **Sign handling:** evaluate positive- and negative-contribution fidelity **separately** (the
  reference paper's most actionable finding) and report whether the CNN "poor on negatives" result
  reproduces on ViTs.
- **Determinism & seeds:** ≥3 seeds per (model × method × dataset × metric) cell; report mean ± std.
- **Fair compute:** fixed input resolution (224), fixed perturbation budgets/steps per metric across
  all methods.

---

## 7. Analysis plan (the Q1 backbone)

This is the core contribution — a benchmark table alone is insufficient.

1. **Ranking-stability analysis** — Spearman/Kendall correlation of method rankings across
   models/datasets/metrics; report **rank-flip rate**. Drives the "does explainability transfer" claim.
2. **Metric meta-evaluation** — inter-metric correlation matrix; quantify how often the chosen metric
   changes the winner; cross-validate proxy metrics against the exact-GT (FunnyBirds/synthetic) arm.
3. **Statistical rigor** — bootstrap confidence intervals; **Friedman test + Nemenyi
   critical-difference diagrams**; pairwise Wilcoxon. (Most XAI benchmarks omit this.)
4. **Scaling regression** — fidelity vs params / FLOPs / accuracy; quantify any "explainability
   scaling law."
5. **Pretraining-objective effect** — grouped comparison controlling for architecture and scale.
6. **Register-token natural experiment** — ±`reg4` for attention-based methods (clean A/B).
7. **Sanity-check pass/fail table** — which methods survive Adebayo randomization on ViTs.
8. **Compute-vs-fidelity Pareto front** — practical "what to use" guidance.

---

## 8. Expected contributions

1. The first **controlled, ground-truth-grounded attribution benchmark on ViT foundation models**,
   spanning architecture, pretraining objective, and scale.
2. **Three empirical findings achieved with zero method novelty:** (a) whether CNN-era rankings
   transfer; (b) the effect of pretraining objective on explainability; (c) the degree of metric
   disagreement on ViTs.
3. A **reusable, Quantus-based open-source harness + leaderboard** for the community.

---

## 9. Experimental setup

- **Frameworks:** PyTorch, `timm` (backbones), `captum` + `pytorch-grad-cam` (methods), **Quantus**
  (metrics).
- **Hardware:** single modern GPU sufficient for gradient/CAM/attention methods; perturbation methods
  (RISE, KernelSHAP, LIME, Occlusion) are the cost bottleneck — budget accordingly and report cost.
- **Reproducibility:** pinned environment, fixed seeds, released configs, per-cell raw outputs, and
  scripts to regenerate every table/figure.

---

## 10. Timeline (indicative, ~5–6 months)

| Phase | Weeks | Output |
|---|---|---|
| 0. Setup & harness | 1–3 | Quantus-based harness; model/method/metric/dataset matrix locked |
| 1. Data + heads | 3–5 | Dataset loaders; linear probes; CLIP zero-shot heads |
| 2. Exact-GT arm | 5–8 | FunnyBirds (+ optional synthetic ViT) pipeline validated |
| 3. Full runs | 8–14 | All (model × method × dataset × metric) cells, ≥3 seeds |
| 4. Analysis | 14–18 | Ranking stability, meta-evaluation, CD diagrams, scaling |
| 5. Writing | 18–24 | Manuscript, figures, leaderboard release, submission |

---

## 11. Deliverables

- Manuscript (Q1 journal).
- Open-source benchmark harness + configs + leaderboard.
- Released result tables (raw + aggregated) and figure-generation scripts.

---

## 12. Target venues (Q1)

IEEE TPAMI · IEEE TNNLS · Pattern Recognition · Information Fusion · Neurocomputing ·
Expert Systems with Applications · IEEE Access.

---

## 13. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Self-supervised backbones lack a class head | Linear probe / CLIP zero-shot; treat head as a variable |
| Perturbation methods too slow at scale | Subsample test set per cell; report cost; parallelize |
| Proxy metrics disagree (no single truth) | Anchor to FunnyBirds causal GT; report disagreement as a *finding* |
| Grad-CAM/attention need ViT-specific reshaping | Use validated `pytorch-grad-cam` ViT reshape; document |
| Scope creep (NLP/multimodal) | Keep vision-only for v1; CLIP text→image as a bonus only |

---

## 14. Open scoping decisions

1. **Exact-GT synthetic arm beyond FunnyBirds?** Recommended: rely on FunnyBirds causal GT; add a
   Brandt-style hand-built small ViT only if reviewers want network-internal exact GT.
2. **Vision-only vs multimodal CLIP text→image attribution?** Recommended: vision-only for v1, CLIP
   text→image as a bonus analysis.
3. **Include distribution-shift axis (ImageNet-C/R/A)?** Optional; adds a robustness story if time allows.

---

## 15. Key references

- Brandt, Raatjens, Gaydadjiev. *Precise Benchmarking of Explainable AI Attribution Methods.* arXiv:2308.03161, 2023.
- Hedström et al. *Quantus: An Explainable AI Toolkit for Responsible Evaluation.* JMLR, 2023.
- Hesse et al. *FunnyBirds: A Synthetic Vision Dataset for a Part-Based Analysis of Explainable AI.* ICCV, 2023.
- Arras, Osman, Samek. *CLEVR-XAI: Ground Truth Evaluation of Neural Network Explanations.* Information Fusion, 2022.
- Rong et al. *A Consistent and Efficient Evaluation Strategy for Attribution Methods (ROAD).* ICML, 2022.
- Adebayo et al. *Sanity Checks for Saliency Maps.* NeurIPS, 2018.
- Chefer, Gur, Wolf. *Transformer Interpretability Beyond Attention Visualization.* CVPR, 2021.
- Abnar & Zuidema. *Quantifying Attention Flow in Transformers (Attention Rollout).* ACL, 2020.
- Oquab et al. *DINOv2.* TMLR, 2024 · Darcet et al. *Vision Transformers Need Registers.* ICLR, 2024.
- He et al. *Masked Autoencoders (MAE).* CVPR, 2022 · Radford et al. *CLIP.* ICML, 2021 · Fang et al. *EVA-02.*
- Sundararajan et al. *Integrated Gradients.* ICML, 2017 · Selvaraju et al. *Grad-CAM.* IJCV, 2019.
