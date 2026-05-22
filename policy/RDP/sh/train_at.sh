#!/bin/bash
# Stage 1: Train AT-VAE for Latent Diffusion Policy
# Usage: bash train_at.sh <task_name> <gpu_id>
# Example: bash train_at.sh insert_HDMI 0

task_name=${1:-insert_HDMI}
gpu_id=${2:-7}

RDP_ROOT="${RDP_ROOT:-/data1/zjb/reactive_diffusion_policy}"
ZARR_DIR="${RDP_ROOT}/data/univtac_${task_name}_zarr"

if [ ! -d "$ZARR_DIR/replay_buffer.zarr" ]; then
    echo "Error: Zarr data not found at $ZARR_DIR/replay_buffer.zarr"
    echo "Please run process_data.sh first."
    exit 1
fi

cd "$RDP_ROOT"

echo "Training AT-VAE for task: ${task_name}"
echo "  Data: ${ZARR_DIR}"
echo "  GPU: ${gpu_id}"

CUDA_VISIBLE_DEVICES=${gpu_id} python train.py \
    --config-name=train_at_workspace \
    task=univtac_at \
    at=at_univtac \
    task.dataset_path="${ZARR_DIR}" \
    training.device="cuda:0" \
    "exp_name=univtac_${task_name}"
