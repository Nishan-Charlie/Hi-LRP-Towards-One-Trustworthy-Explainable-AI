#!/usr/bin/env bash
# Lean, sequential queue for the NEUCOM revision (~4-5 h of GPU; run it with the GPU to
# itself, see run_after_wildfire.ps1 -- concurrent CUDA jobs trigger driver faults here).
# One process at a time; every step resumes from its CSV and exits on a CUDA fault,
# so a retry continues cleanly. A file named STOP in this folder ends the queue after
# the current step. Run from scripts/revision_neucom.
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
runner() {  # benchmark runner, below-normal priority; resumes from results.json
  [ -f STOP ] && { echo "STOPPED before: runner $1"; exit 0; }
  for t in 1 2 3; do
    (cd ../.. && python -c "import psutil,runpy; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS); runpy.run_module('xai_bench.runner', run_name='__main__')" --config "$1")
  done
}
MODELS="resnet50 vit_base_patch16_224 swin_base_patch4_window7_224 pvt_v2_b2 maxvit_small_tf_224 mobilevitv2_100 efficientvit_b1 efficientvit_b2"
EXTRA="convnext_tiny convnext_base vit_small_patch16_224 deit_small_patch16_224 deit_base_patch16_224 swin_tiny_patch4_window7_224 maxvit_tiny_tf_224 mobilevitv2_200 efficientvit_b3 resnet18 resnet101 pvt_v2_b0 pvt_v2_b4"
FAST=saliency,input_x_gradient,smoothgrad,vargrad,gradient_shap,grad_cam,grad_cam_plus_plus,attention_rollout
CHEAP=saliency,input_x_gradient,grad_cam,grad_cam_plus_plus,grad_cam_plus_plus_stab,attention_rollout

# 1. localisation of the fast methods on the first 300 images
for m in $MODELS; do run loc.py $m 300 $FAST; done
# 2. multi-seed Max-Sensitivity (3 seeds x 10 perturbations x 100 images), incl. bounded-alpha Grad-CAM++
for m in $MODELS; do run ms.py $m 100 0,1,2 10 $CHEAP; done
# 3. IG and RISE localisation (100 images); AttnLRP localisation and MS (own processes)
for m in $MODELS; do run loc.py $m 100 integrated_gradients,rise; done
run loc.py vit_base_patch16_224 300 attnlrp
run ms.py vit_base_patch16_224 100 0,1,2 10 attnlrp
# 4. VarGrad multi-seed MS (the Table 5 outliers): 3 seeds x 10 perturbations x 30 images
for m in $MODELS; do run ms.py $m 30 0,1,2 10 vargrad; done
# 5. ViT-B/16 five-metric row with native input statistics
runner configs/revision_vit_native.yaml
# 6. perturbation-method localisation (100 images) and the faithfulness protocols
for m in $MODELS; do run loc.py $m 100 lime; done
for m in $MODELS; do run loc.py $m 100 occlusion; done
for m in $MODELS; do run ffe.py $m 100 integrated_gradients,input_x_gradient,gradient_shap,occlusion,lime; done
run ffe.py vit_base_patch16_224 100 attnlrp
# 7. capacity study (Grad-CAM / Grad-CAM++ localisation and stability on 13 more backbones)
for m in $EXTRA; do
  run loc.py $m 300 grad_cam,grad_cam_plus_plus,attention_rollout
  run ms.py $m 100 0,1,2 10 grad_cam,grad_cam_plus_plus,grad_cam_plus_plus_stab
done
echo QUEUE_LEAN_DONE
