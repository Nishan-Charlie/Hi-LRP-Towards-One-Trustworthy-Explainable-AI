#!/usr/bin/env bash
# Extends the localization runs and the Grad-CAM++ diagnostics from the first 300 to all
# 1,000 evaluation images. loc.py resumes per (method, image, target) and seeds every image
# independently, so the 300-image results are renamed to the 1,000-image file and only
# images 300-999 are computed. IG and the perturbation methods stay at 100 images (cost).
# A file named STOP in this folder ends the queue after the current step.
set -u
OUT=../../results/revision_neucom
run() {
  [ -f STOP ] && { echo "STOPPED before: $*"; exit 0; }
  for t in 1 2 3 4 5; do
    python "$@" && return 0
    echo "RETRY $t: $*"
    sleep 30
  done
  echo "GAVE_UP: $*"
}
extend() {  # extend MODEL METHODS: rename the n300 file, then resume to 1,000 images
  local f="$OUT/loc_${1}_${2//,/_}_n300.csv"
  [ -f "$f" ] && mv "$f" "${f%_n300.csv}_n1000.csv"
  run loc.py "$1" 1000 "$2"
}
MODELS="resnet50 vit_base_patch16_224 swin_base_patch4_window7_224 pvt_v2_b2 maxvit_small_tf_224 mobilevitv2_100 efficientvit_b1 efficientvit_b2"
EXTRA="convnext_tiny convnext_base vit_small_patch16_224 deit_small_patch16_224 deit_base_patch16_224 swin_tiny_patch4_window7_224 maxvit_tiny_tf_224 mobilevitv2_200 efficientvit_b3 resnet18 resnet101 pvt_v2_b0 pvt_v2_b4"
FAST=saliency,input_x_gradient,smoothgrad,vargrad,gradient_shap,grad_cam,grad_cam_plus_plus,attention_rollout

# MaxViT-S: the 20-sample batches of SmoothGrad/VarGrad/GradientSHAP overflow the 8 GB
# GPU into system memory; XAI_GRAD_CHUNK=5 computes their gradients 5 samples at a
# time (same noise, same maps up to rounding; xai_bench/utils/chunked_gradients.py)
chunk_for() { if [ "$1" = maxvit_small_tf_224 ]; then export XAI_GRAD_CHUNK=5; else unset XAI_GRAD_CHUNK; fi; }
for m in $MODELS; do chunk_for $m; extend $m $FAST; done
unset XAI_GRAD_CHUNK
extend vit_base_patch16_224 attnlrp
for m in $EXTRA; do
  extend $m grad_cam,grad_cam_plus_plus,attention_rollout
  run gcpp.py $m 1000
done
# GradientSHAP grid cells with no stored result (MaxViT-S, MobileViT-v2, EffViT-B1/B2);
# the runner resumes from results.json
export XAI_GRAD_CHUNK=5
for t in 1 2 3; do
  [ -f STOP ] && break
  (cd ../.. && python -m xai_bench.runner --config configs/revision_gradshap_fill.yaml) && break
done
unset XAI_GRAD_CHUNK
# Cost table: synchronized GPU timing of all 13 methods on the eight backbones
# (20 images spread over the 1,000-image set; the cost does not depend on content)
ALL=saliency,integrated_gradients,smoothgrad,input_x_gradient,vargrad,gradient_shap,grad_cam,grad_cam_plus_plus,attention_rollout,rise,lime,occlusion
for m in $MODELS; do chunk_for $m; run time_methods.py $m 20 $ALL; done
unset XAI_GRAD_CHUNK
run time_methods.py vit_base_patch16_224 20 attnlrp
run cost_table.py
# Qualitative gallery (map cache was cleared; each stage in its own process)
for stage in baselines attnlrp_vit assemble; do
  [ -f STOP ] && break
  for t in 1 2 3; do
    (cd ../.. && python scripts/bench/qualitative_grid_cat.py $stage) && break
    echo "RETRY $t: gallery $stage"; sleep 30
  done
done
echo EXTEND_1000_DONE
