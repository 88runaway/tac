#!/bin/bash
# Stage 2: Train Latent Diffusion Policy with marker PCA embedding tactile input.
# Requires a trained AT-VAE checkpoint from train_at_marker.sh.
#
# Usage:
#   bash train_ldp_marker.sh <task_name> <at_ckpt_dir> [gpu_id]
#
# Example:
#   bash train_ldp_marker.sh insert_HDMI \
#       /data1/zjb/reactive_diffusion_policy/data/outputs/2024.01.01/12.00.00_train_vae_univtac_at_marker_emb/checkpoints 0

set -e

task_name=${1:?'Usage: train_ldp_marker.sh <task_name> <at_ckpt_dir> [gpu_id]'}
at_ckpt_dir=${2:?'Usage: train_ldp_marker.sh <task_name> <at_ckpt_dir> [gpu_id]'}
gpu_id=${3:-0}

RDP_ROOT="${RDP_ROOT:-/data1/zjb/reactive_diffusion_policy}"
ZARR_DIR="${RDP_ROOT}/data/univtac_${task_name}_marker_zarr"

if [ ! -d "${ZARR_DIR}/replay_buffer.zarr" ]; then
    echo "Error: Marker zarr not found at ${ZARR_DIR}/replay_buffer.zarr"
    echo "Please run process_data_marker.sh first."
    exit 1
fi

if [ ! -e "$at_ckpt_dir" ]; then
    echo "Error: AT checkpoint not found: ${at_ckpt_dir}"
    exit 1
fi

cd "$RDP_ROOT"

echo "=== Latent Diffusion Policy Training (marker emb) ==="
echo "  Task:          ${task_name}  |  GPU: ${gpu_id}"
echo "  Data:          ${ZARR_DIR}"
echo "  AT checkpoint: ${at_ckpt_dir}"

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=${gpu_id} python train.py \
    --config-name=train_latent_diffusion_unet_real_image_workspace \
    task=univtac_ldp_marker_emb \
    at=at_univtac \
    "task.dataset_path='${ZARR_DIR}'" \
    "at_load_dir='${at_ckpt_dir}'" \
    training.device="cuda:0" \
    "exp_name=univtac_marker_${task_name}"
