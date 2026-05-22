#!/usr/bin/env bash
# Run one episode and save a video with tactile modalities split (rgb / rgb_marker / depth / marker).
# Usage:
#   bash vis_tactile.sh <task_name> [model_config] [gpu_id] [seed]
# Example:
#   bash vis_tactile.sh insert_tube univtac 2 1000001

task_name=${1:-put_bottle_in_shelf}
model_config=${2:-univtac}
gpu_id=${3:-0}
seed=${4:-67}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/select_train_config.sh" "${task_name}" "${model_config}"

export CKPT_CONFIG=${model_config}
export CKPT_ROOT=/data1/zjb/ckpt/UniVTAC/checkpoints
export TORCH_HOME=/data1/zjb/ckpt/UniVTAC/torch
export LD_PRELOAD=/data1/zjb/miniconda3/envs/UniVTAC/lib/libstdc++.so.6
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="8.9"
export CUROBO_NO_JIT=1

SEED_ARGS=""
if [ -n "${seed}" ]; then
    SEED_ARGS="--seed ${seed}"
fi

ROOT_DIR="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
cd "$ROOT_DIR"

source /data1/zjb/UniVTAC/IsaacLab/_isaac_sim/setup_conda_env.sh

PYTHONWARNINGS=ignore::UserWarning \
CUDA_VISIBLE_DEVICES=${gpu_id} python scripts/vis_tactile.py \
    "${task_name}" \
    demo \
    ACT/config/deploy \
    ${SEED_ARGS} \
    --headless
