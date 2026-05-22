#!/bin/bash
# Two-stage LDP pipeline with RGB tactile input.
# Stage 1: Train AT-VAE, Stage 2: Train LDP using the AT checkpoint.
#
# Usage:
#   bash train_ldp_pipeline.sh <task_name> [gpu_id]
#
# Example:
#   bash train_ldp_pipeline.sh grasp_classify 0
#
# Prerequisites:
#   Run process_data.sh first to build the zarr dataset.
#
# Options (environment variables):
#   RDP_ROOT      path to reactive_diffusion_policy repo (default: /data1/zjb/reactive_diffusion_policy)
#   AT_EPOCHS     AT-VAE training epochs, overrides config (default: use config value)
#   LDP_EPOCHS    LDP training epochs, overrides config (default: use config value)

set -e

task_name=${1:?'Usage: train_ldp_pipeline.sh <task_name> [gpu_id]'}
gpu_id=${2:-0}

RDP_ROOT="${RDP_ROOT:-/data1/zjb/reactive_diffusion_policy}"
ZARR_DIR="${RDP_ROOT}/data/univtac_${task_name}_zarr"

# Fixed output directories so we can reliably locate checkpoints after training
DATESTAMP="$(date +%Y.%m.%d)"
TIMESUFFIX="$(date +%H.%M.%S)"
AT_RUN_DIR="data/outputs/${DATESTAMP}/${TIMESUFFIX}_train_vae_univtac_at_${task_name}"
LDP_RUN_DIR="data/outputs/${DATESTAMP}/${TIMESUFFIX}_train_ldp_${task_name}"

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
if [ ! -d "${ZARR_DIR}/replay_buffer.zarr" ]; then
    echo "Error: Zarr not found at ${ZARR_DIR}/replay_buffer.zarr"
    echo "Please run process_data.sh first."
    exit 1
fi

echo "========================================================"
echo "  LDP Pipeline: ${task_name}  GPU: ${gpu_id}"
echo "  Data:    ${ZARR_DIR}"
echo "  AT dir:  ${RDP_ROOT}/${AT_RUN_DIR}"
echo "  LDP dir: ${RDP_ROOT}/${LDP_RUN_DIR}"
echo "========================================================"

cd "${RDP_ROOT}"

# ---------------------------------------------------------------------------
# Stage 1: Train AT-VAE
# ---------------------------------------------------------------------------
echo ""
echo "=== Stage 1/2: Training AT-VAE ==="

AT_EXTRA_ARGS=""
[ -n "${AT_EPOCHS:-}" ] && AT_EXTRA_ARGS="${AT_EXTRA_ARGS} training.num_epochs=${AT_EPOCHS}"

CUDA_VISIBLE_DEVICES=${gpu_id} python train.py \
    --config-name=train_at_workspace \
    task=univtac_at \
    at=at_univtac \
    "task.dataset_path='${ZARR_DIR}'" \
    training.device="cuda:0" \
    "exp_name=univtac_${task_name}" \
    "hydra.run.dir='${AT_RUN_DIR}'" \
    "logging.resume=False" \
    ${AT_EXTRA_ARGS}

# Locate AT checkpoint (latest.ckpt preferred, then last.ckpt, then newest by mtime)
AT_CKPT_DIR="${RDP_ROOT}/${AT_RUN_DIR}/checkpoints"
if [ -f "${AT_CKPT_DIR}/latest.ckpt" ]; then
    AT_CKPT="${AT_CKPT_DIR}/latest.ckpt"
elif [ -f "${AT_CKPT_DIR}/last.ckpt" ]; then
    AT_CKPT="${AT_CKPT_DIR}/last.ckpt"
else
    AT_CKPT="$(ls -t "${AT_CKPT_DIR}"/*.ckpt 2>/dev/null | head -1)"
fi

if [ -z "${AT_CKPT}" ] || [ ! -f "${AT_CKPT}" ]; then
    echo "Error: AT checkpoint not found in ${AT_CKPT_DIR}"
    exit 1
fi

echo ""
echo "AT-VAE training complete."
echo "  Checkpoint: ${AT_CKPT}"

# ---------------------------------------------------------------------------
# Stage 2: Train Latent Diffusion Policy
# ---------------------------------------------------------------------------
echo ""
echo "=== Stage 2/2: Training Latent Diffusion Policy ==="
echo "  AT checkpoint: ${AT_CKPT}"

LDP_EXTRA_ARGS=""
[ -n "${LDP_EPOCHS:-}" ] && LDP_EXTRA_ARGS="${LDP_EXTRA_ARGS} training.num_epochs=${LDP_EPOCHS}"

CUDA_VISIBLE_DEVICES=${gpu_id} python train.py \
    --config-name=train_latent_diffusion_unet_real_image_workspace \
    task=univtac_ldp \
    at=at_univtac \
    "task.dataset_path='${ZARR_DIR}'" \
    "at_load_dir='${AT_CKPT}'" \
    training.device="cuda:0" \
    "exp_name=univtac_${task_name}" \
    "hydra.run.dir='${LDP_RUN_DIR}'" \
    "logging.resume=False" \
    ${LDP_EXTRA_ARGS}

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "========================================================"
echo "  Pipeline complete!"
echo "  Task:            ${task_name}"
echo "  AT checkpoint:   ${AT_CKPT}"
echo "  LDP output dir:  ${RDP_ROOT}/${LDP_RUN_DIR}"
echo "========================================================"
