#!/bin/bash
# Evaluate RDP policy on UniVTAC benchmark
# Usage: bash eval.sh <task_name> <model_config> <gpu_id> [save_video]
# Example: bash eval.sh insert_HDMI univtac 0 true
#
# model_config: directory name under CKPT_ROOT/<task_name>/
#   e.g. "univtac" -> CKPT_ROOT/insert_HDMI/univtac/checkpoints/latest.ckpt

task_name=${1:-lift_bottle}
task_config=${2:-demo}
model_config=${2:-univtac}
gpu_id=${3:-0}
save_video=${4:-true}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"

export CKPT_CONFIG=${model_config}
export CKPT_ROOT="${CKPT_ROOT:-/data1/zjb/ckpt/RDP/checkpoints}"
export RDP_ROOT="${RDP_ROOT:-/data1/zjb/reactive_diffusion_policy}"
export TORCH_HOME="${TORCH_HOME:-/data1/zjb/ckpt/UniVTAC/torch}"
export LD_PRELOAD=/data1/zjb/miniconda3/envs/UniVTAC/lib/libstdc++.so.6
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="8.9"
export CUROBO_NO_JIT=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

SAVE_ARGS=""
[ "$save_video" = "true" ] && SAVE_ARGS="$SAVE_ARGS --save_video"

cd "$ROOT_DIR"

source /data1/zjb/UniVTAC/IsaacLab/_isaac_sim/setup_conda_env.sh

PYTHONWARNINGS=ignore::UserWarning \
CUDA_VISIBLE_DEVICES=${gpu_id} python scripts/eval_policy.py \
    ${task_name} \
    ${task_config} \
    RDP/config/deploy \
    ${SAVE_ARGS} \
    --headless
