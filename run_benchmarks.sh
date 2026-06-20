#!/bin/bash
set -e

# Run vit_base
cat <<EOF > configs/temp_vit.yaml
models: ["vit_base_patch16_224"]
methods: ["grad_cam", "integrated_gradients", "attention_rollout", "saliency"]
metrics: ["faithfulness_correlation", "auc", "pointing_game"]
dataset: "imagenets"
dataset_kwargs: {num_samples: 1000, split: "validation"}
num_images: 1000
seeds: [0]
device: "auto"
input_size: 224
output_dir: "results"
run_name: "quant_vit"
save_saliency: false
limit_methods_to_supported: true
EOF
conda run -n mri-diffuser python -m xai_bench.runner --config configs/temp_vit.yaml || true

# Run deit_base
cat <<EOF > configs/temp_deit.yaml
models: ["deit_base_patch16_224"]
methods: ["grad_cam", "integrated_gradients", "attention_rollout", "saliency"]
metrics: ["faithfulness_correlation", "auc", "pointing_game"]
dataset: "imagenets"
dataset_kwargs: {num_samples: 1000, split: "validation"}
num_images: 1000
seeds: [0]
device: "auto"
input_size: 224
output_dir: "results"
run_name: "quant_deit"
save_saliency: false
limit_methods_to_supported: true
EOF
conda run -n mri-diffuser python -m xai_bench.runner --config configs/temp_deit.yaml || true

# Run resnet50
cat <<EOF > configs/temp_resnet.yaml
models: ["resnet50"]
methods: ["grad_cam", "integrated_gradients", "saliency"]
metrics: ["faithfulness_correlation", "auc", "pointing_game"]
dataset: "imagenets"
dataset_kwargs: {num_samples: 1000, split: "validation"}
num_images: 1000
seeds: [0]
device: "auto"
input_size: 224
output_dir: "results"
run_name: "quant_resnet"
save_saliency: false
limit_methods_to_supported: true
EOF
conda run -n mri-diffuser python -m xai_bench.runner --config configs/temp_resnet.yaml || true
