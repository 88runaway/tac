#!/bin/bash
# Stage 2: Train Latent Diffusion Policy (requires trained AT-VAE checkpoint)
# Usage: bash train_ldp.sh <task_name> <at_ckpt_dir> <gpu_id>
# Example: bash train_ldp.sh insert_HDMI /path/to/at/output/checkpoints 0

task_name=${1}
at_ckpt_dir=${2}
gpu_id=${3:-0}

RDP_ROOT="${RDP_ROOT:-/data1/zjb/reactive_diffusion_policy}"
ZARR_DIR="${RDP_ROOT}/data/univtac_${task_name}_zarr"

if [ ! -d "$ZARR_DIR/replay_buffer.zarr" ]; then
    echo "Error: Zarr data not found at $ZARR_DIR/replay_buffer.zarr"
    echo "Please run process_data.sh first."
    exit 1
fi

if [ -z "$at_ckpt_dir" ]; then
    echo "Error: AT checkpoint directory not specified."
    echo "Usage: bash train_ldp.sh <task_name> <at_ckpt_dir> [gpu_id]"
    exit 1
fi

cd "$RDP_ROOT"

echo "Training Latent Diffusion Policy for task: ${task_name}"
echo "  Data: ${ZARR_DIR}"
echo "  AT checkpoint: ${at_ckpt_dir}"
echo "  GPU: ${gpu_id}"

CUDA_VISIBLE_DEVICES=${gpu_id} python train.py \
    --config-name=train_latent_diffusion_unet_real_image_workspace \
    task=univtac_ldp \
    at=at_univtac \
    task.dataset_path="${ZARR_DIR}" \
    at_load_dir="${at_ckpt_dir}" \
    training.device="cuda:0" \
    "exp_name=univtac_${task_name}"
