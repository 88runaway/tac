#!/bin/bash
# ─── UniVTAC Pi0.5 OpenPI (JAX) Full Fine-tuning Script ──────────────────────
#
# Train pi0.5 policy on UniVTAC task data using the OpenPI JAX framework.
#
# Usage:
#   bash policy/Pi05_openpi/sh/train_openpi.sh <task_name> [gpu_id] [config_name] [extra_args...]
#
# Examples:
#   # Single task with default config
#   bash policy/Pi05_openpi/sh/train_openpi.sh lift_bottle 0
#
#   # Full fine-tuning with custom steps
#   bash policy/Pi05_openpi/sh/train_openpi.sh lift_bottle 0,1,2,3 train_full_openpi --wandb
#
#   # Block-wise Diffusion Forcing (Pi0DF) fine-tuning
#   #   --diffusion_forcing 启用逐 block 噪声等级；num_blocks 必须整除 action_horizon(50)
#   bash policy/Pi05_openpi/sh/train_openpi.sh lift_bottle 0 train_full_openpi \
#       --diffusion_forcing --num_blocks 5 --mix_prob 0.5
#   # 评估对应 DF checkpoint：EVAL_CONFIG=eval_df bash policy/Pi05_openpi/sh/eval.sh lift_bottle 0
#
#   # Compute norm stats only
#   bash policy/Pi05_openpi/sh/train_openpi.sh lift_bottle 0 train_full_openpi --compute_norm_stats_only
#
#   # Resume training
#   bash policy/Pi05_openpi/sh/train_openpi.sh lift_bottle 0 train_full_openpi --resume
#
#   # Enable WandB
#   bash policy/Pi05_openpi/sh/train_openpi.sh lift_bottle 0 train_full_openpi --wandb
#
# Prerequisites:
#   1. Convert data: python policy/Pi05_openpi/convert_multitask_openpi.py --task <task>
#   2. Install openpi: cd /data1/zjb/openpi && pip install -e .
#
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

TASK_NAME="${1:?Usage: $0 <task_name> [gpu_id] [config_name] [extra_args...]}"
GPU_ID="${2:-0}"
CONFIG_NAME="${3:-train_full_openpi}"
shift 1
[ $# -ge 1 ] && shift 1
[ $# -ge 1 ] && shift 1

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_DIR="$(dirname "$SCRIPT_DIR")"
UNIVTAC_ROOT="$(dirname "$(dirname "$POLICY_DIR")")"
OPENPI_ROOT="/data1/zjb/openpi"
DATASET_ROOT="/data1/zjb/data_lerobot_openpi"
CONFIG_FILE="${POLICY_DIR}/config/${CONFIG_NAME}.yaml"

# Check dataset (openpi v2.1 format)
DATASET_DIR="${DATASET_ROOT}/${TASK_NAME}"
if [ ! -d "$DATASET_DIR" ]; then
    echo "Dataset not found at $DATASET_DIR"
    echo "Running data conversion (HDF5 → openpi LeRobot)..."
    python "${POLICY_DIR}/convert_multitask_openpi.py" --task "$TASK_NAME" --output_dir "$DATASET_ROOT"
    if [ ! -d "$DATASET_DIR" ]; then
        echo "ERROR: Data conversion failed. Dataset not found at $DATASET_DIR"
        echo "Please run manually:"
        echo "  conda activate openpi"
        echo "  python policy/Pi05_openpi/convert_multitask_openpi.py --task $TASK_NAME"
        exit 1
    fi
fi

# Read config overrides from yaml if available
BATCH_SIZE=4
STEPS=10000
LR="2.5e-5"
WARMUP_STEPS=500
SAVE_FREQ=1000
LOG_FREQ=50
NUM_WORKERS=4
ACTION_HORIZON=50
MODEL_TYPE="vision_only"

if [ -f "$CONFIG_FILE" ]; then
    echo "Loading config from: $CONFIG_FILE"
    _yaml_val() { grep -m1 "^${1}:" "$CONFIG_FILE" 2>/dev/null | awk '{print $2}' | tr -d '"' || echo "$2"; }
    BATCH_SIZE=$(_yaml_val "batch_size" "$BATCH_SIZE")
    STEPS=$(_yaml_val "steps" "$STEPS")
    LR=$(_yaml_val "lr" "$LR")
    WARMUP_STEPS=$(_yaml_val "warmup_steps" "$WARMUP_STEPS")
    SAVE_FREQ=$(_yaml_val "save_freq" "$SAVE_FREQ")
    LOG_FREQ=$(_yaml_val "log_freq" "$LOG_FREQ")
    NUM_WORKERS=$(_yaml_val "num_workers" "$NUM_WORKERS")
    ACTION_HORIZON=$(_yaml_val "action_horizon" "$ACTION_HORIZON")
    MODEL_TYPE=$(_yaml_val "model" "$MODEL_TYPE")
    SAVE_BEST_ONLY=$(_yaml_val "save_best_only" "true")
fi

# Allow CLI override: --model=tactile
FILTERED_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --model=*) MODEL_TYPE="${arg#--model=}" ;;
        *) FILTERED_ARGS+=("$arg") ;;
    esac
done
set -- "${FILTERED_ARGS[@]+"${FILTERED_ARGS[@]}"}"

TIMESTAMP="$(date +%Y-%m-%d_%H:%M:%S)"
OUTPUT_DIR="/data1/zjb/outputs_openpi/pi05_${TASK_NAME}_${CONFIG_NAME}/${TIMESTAMP}"

echo "═══════════════════════════════════════════════════════════════"
echo " UniVTAC Pi0.5 OpenPI (JAX) Training"
echo "═══════════════════════════════════════════════════════════════"
echo "  Task:          $TASK_NAME"
echo "  Model type:    $MODEL_TYPE"
echo "  GPU:           $GPU_ID"
echo "  Config:        $CONFIG_FILE"
echo "  Dataset:       $DATASET_DIR"
echo "  Batch size:    $BATCH_SIZE"
echo "  Steps:         $STEPS"
echo "  LR:            $LR"
echo "  Action horizon:$ACTION_HORIZON"
echo "  Output dir:    $OUTPUT_DIR"
echo "  Extra args:    $@"
echo "═══════════════════════════════════════════════════════════════"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# Redirect caches
export HF_HOME="${HF_HOME:-/data1/zjb/cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/data1/zjb/cache/huggingface/datasets}"
export TMPDIR="${TMPDIR:-/data1/zjb/cache/tmp}"
export WANDB_DIR="${WANDB_DIR:-/data1/zjb/cache/wandb}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-/data1/zjb/cache/wandb/data}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/data1/zjb/cache/wandb/cache}"
mkdir -p "$HF_DATASETS_CACHE" "$TMPDIR" "$WANDB_DIR" "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR"

cd "$UNIVTAC_ROOT"

SAVE_BEST_FLAG=""
if [ "$SAVE_BEST_ONLY" = "true" ]; then
    SAVE_BEST_FLAG="--save_best_only"
fi

python policy/Pi05_openpi/train_pi05_openpi.py \
    --task "$TASK_NAME" \
    --gpu "$GPU_ID" \
    --model_type "$MODEL_TYPE" \
    --batch_size "$BATCH_SIZE" \
    --steps "$STEPS" \
    --lr "$LR" \
    --warmup_steps "$WARMUP_STEPS" \
    --save_freq "$SAVE_FREQ" \
    --log_freq "$LOG_FREQ" \
    --num_workers "$NUM_WORKERS" \
    --action_horizon "$ACTION_HORIZON" \
    --output_dir "$OUTPUT_DIR" \
    $SAVE_BEST_FLAG \
    "$@"

echo ""
echo "Training complete! Model saved to: ${OUTPUT_DIR}"
