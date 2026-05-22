#!/bin/bash
# Stage 1: Train AT-VAE with marker PCA embedding tactile input.
# Usage: bash train_at_marker.sh <task_name> <gpu_id> [n_components]
# Example: bash train_at_marker.sh insert_HDMI 0 32
#
# Prerequisites:
#   Run process_data_marker.sh first to build the marker_zarr dataset.

task_name=${1:-insert_HDMI}
gpu_id=${2:-7}
n_components=${3:-15}

RDP_ROOT="${RDP_ROOT:-/data1/zjb/reactive_diffusion_policy}"
ZARR_DIR="${RDP_ROOT}/data/univtac_${task_name}_marker_zarr"

if [ ! -d "$ZARR_DIR/replay_buffer.zarr" ]; then
    echo "Error: Marker zarr data not found at $ZARR_DIR/replay_buffer.zarr"
    echo "Please run process_data_marker.sh first."
    exit 1
fi

cd "$RDP_ROOT"

echo "Training AT-VAE (marker emb) for task: ${task_name}"
echo "  Data:       ${ZARR_DIR}"
echo "  GPU:        ${gpu_id}"
echo "  Components: ${n_components}"

CUDA_VISIBLE_DEVICES=${gpu_id} python train.py \
    --config-name=train_at_workspace \
    task=univtac_at_marker_emb \
    at=at_univtac \
    task.dataset_path="${ZARR_DIR}" \
    training.device="cuda:0" \
    "exp_name=univtac_marker_${task_name}"
# NOTE: n_tac_components is set via YAML anchor in univtac_at_marker_emb.yaml.
# YAML anchors cannot be overridden via CLI. Edit the yaml directly if you
# change --n_components in process_data_marker.sh.
