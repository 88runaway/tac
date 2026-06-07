#!/bin/bash
# ─── UniVTAC Pi0.5 LoRA Fine-tuning Script ───────────────────────────────────
#
# Train pi0.5 policy on UniVTAC task data using LoRA adapters.
# Delegates to lerobot.scripts.lerobot_train (official entry point).
#
# Usage:
#   bash policy/Pi05/sh/train.sh <task_name> [gpu_id] [config_name] [extra_args...]
#
# Examples:
#   # Single task with default config (LoRA, vision_only)
#   bash policy/Pi05/sh/train.sh lift_can 0
#
#   # Full fine-tuning
#   bash policy/Pi05/sh/train.sh lift_bottle 3,5,6,7 train_full
#
#   # Train with tactile input (override yaml model field via CLI)
#   bash policy/Pi05/sh/train.sh insert_tube 0 train_lora --model=tactile
#
#   # Override steps (CLI args take precedence over yaml)
#   bash policy/Pi05/sh/train.sh lift_can 0 train_lora --steps=10000
#
# Prerequisites:
#   1. Convert data: python scripts/convert_to_lerobot.py --task <task> --output_dir /data1/zjb/lerobot_datasets
#   2. Install deps: cd /data1/zjb/lerobot && pip install -e ".[pi,peft]"
#
# Note on optimizer/lr: use_policy_training_preset=true (default) means lerobot
#   automatically uses PI05Config presets: lr=2.5e-5, AdamW, cosine decay.
#   To override, pass --use_policy_training_preset=false --optimizer.lr=<lr>.
#
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

TASK_NAME="${1:?Usage: $0 <task_name> [gpu_id] [config_name] [extra_args...]}"
GPU_ID="${2:-0}"
CONFIG_NAME="${3:-train_lora}"
# shift exactly the number of positional args we consumed
shift 1
[ $# -ge 1 ] && shift 1
[ $# -ge 1 ] && shift 1

# Detect --resume flag in extra args (remaining "$@")
RESUME=false
NEW_ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--resume" ] || [ "$arg" = "--resume=true" ]; then
        RESUME=true
    else
        NEW_ARGS+=("$arg")
    fi
done
set -- "${NEW_ARGS[@]+"${NEW_ARGS[@]}"}"

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_DIR="$(dirname "$SCRIPT_DIR")"
UNIVTAC_ROOT="$(dirname "$(dirname "$POLICY_DIR")")"
LEROBOT_ROOT="/data1/zjb/lerobot"
DATASET_ROOT="/data1/zjb/UniVTAC/data_lerobot"
CONFIG_FILE="${POLICY_DIR}/config/${CONFIG_NAME}.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    echo "Available configs:"
    ls "${POLICY_DIR}/config/"
    exit 1
fi

DATASET_DIR="${DATASET_ROOT}/${TASK_NAME}"
if [ ! -d "$DATASET_DIR" ]; then
    echo "ERROR: Dataset not found at $DATASET_DIR"
    echo "Run data conversion first:"
    echo "  python scripts/convert_to_lerobot.py --task $TASK_NAME --output_dir $DATASET_ROOT"
    exit 1
fi

# Read 'model' field from config yaml (UniVTAC-specific, not a lerobot field).
# Format: "model: vision_only" or "model: tactile" as a top-level yaml key.
MODEL_TYPE=$(grep -m1 '^model:' "$CONFIG_FILE" 2>/dev/null | awk '{print $2}' || echo "vision_only")
MODEL_TYPE="${MODEL_TYPE:-vision_only}"

# Allow CLI override: --model=tactile
FILTERED_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --model=*) MODEL_TYPE="${arg#--model=}" ;;
        *) FILTERED_ARGS+=("$arg") ;;
    esac
done
set -- "${FILTERED_ARGS[@]+"${FILTERED_ARGS[@]}"}"

# Create a clean config yaml without the 'model' field (draccus rejects unknown fields)
CLEAN_CONFIG=$(mktemp /tmp/univtac_train_config_XXXXXX.yaml)
grep -v '^model:' "$CONFIG_FILE" > "$CLEAN_CONFIG"
trap "rm -f '${CLEAN_CONFIG}'" EXIT

# Read task-specific params from task_settings.json, respecting model type
read -r CAMERA_TYPE EMPTY_CAMERAS RENAME_MAP <<< "$(python3 -c "
import json, sys
with open('${UNIVTAC_ROOT}/policy/task_settings.json') as f:
    settings = json.load(f)
t = settings.get('${TASK_NAME}', {})
model_type = '${MODEL_TYPE}'
camera_type = t.get('camera_type', 'head')
if model_type == 'tactile':
    empty_cameras = t.get('tactile_empty_cameras', 0)
    rename_map = json.dumps(t.get('tactile_rename_map', t.get('rename_map', {})))
else:
    empty_cameras = t.get('empty_cameras', 2 if camera_type == 'head' else 1)
    rename_map = json.dumps(t.get('rename_map', {'observation.images.head': 'observation.images.base_0_rgb'}))
print(camera_type, empty_cameras, rename_map)
")"

TIMESTAMP="$(date +%Y-%m-%d_%H:%M:%S)"
BASE_OUTPUT_DIR="${UNIVTAC_ROOT}/outputs/pi05_${TASK_NAME}_${CONFIG_NAME}"

if [ "$RESUME" = "true" ]; then
    # Resume: find the most recent timestamped run directory
    LAST_RUN=$(ls -td "${BASE_OUTPUT_DIR}/"* 2>/dev/null | head -1)
    if [ -z "$LAST_RUN" ] || [ ! -d "${LAST_RUN}/checkpoints" ]; then
        echo "ERROR: --resume specified but no previous run found under ${BASE_OUTPUT_DIR}/"
        exit 1
    fi
    OUTPUT_DIR="$LAST_RUN"
    LAST_CKPT=$(readlink -f "$OUTPUT_DIR/checkpoints/last" 2>/dev/null || true)
else
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${TIMESTAMP}"
fi

echo "═══════════════════════════════════════════════════════════════"
echo " UniVTAC Pi0.5 Training"
echo "═══════════════════════════════════════════════════════════════"
echo "  Task:          $TASK_NAME"
echo "  Model type:    $MODEL_TYPE"
echo "  GPU:           $GPU_ID"
echo "  Config:        $CONFIG_FILE"
echo "  Camera type:   $CAMERA_TYPE (empty_cameras=$EMPTY_CAMERAS)"
echo "  Dataset:       $DATASET_DIR"
echo "  Optimizer:     PI05Config preset (lr=2.5e-5, AdamW, cosine decay)"
echo "  Resume:        $RESUME"
echo "  Output dir:    $OUTPUT_DIR"
echo "  Extra args:    $@"
echo "═══════════════════════════════════════════════════════════════"
if [ "$RESUME" = "true" ]; then
    echo "  Resuming from ckpt: $LAST_CKPT"
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"

# Redirect caches away from /home and / (may be full) to /data1
export HF_HOME="${HF_HOME:-/data1/zjb/cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/data1/zjb/cache/huggingface/datasets}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/data1/zjb/cache/triton}"
export TORCH_HOME="${TORCH_HOME:-/data1/zjb/cache/torch}"
export TMPDIR="${TMPDIR:-/data1/zjb/cache/tmp}"       # gcc/triton compilation temp files
export WANDB_DIR="${WANDB_DIR:-/data1/zjb/cache/wandb}"          # wandb run logs dir
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-/data1/zjb/cache/wandb/data}"  # artifacts staging (was ~/.local/share/wandb)
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/data1/zjb/cache/wandb/cache}"  # artifact cache (was ~/.cache/wandb)
mkdir -p "$HF_DATASETS_CACHE" "$TRITON_CACHE_DIR" "$TORCH_HOME" "$TMPDIR" "$WANDB_DIR" "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR"

# Count GPUs: single id → 1 GPU, comma-separated → multi-GPU via accelerate
GPU_COUNT=$(echo "$GPU_ID" | tr ',' '\n' | wc -l)

cd "$LEROBOT_ROOT"

# Pass the YAML config via --config_path (draccus supports this).
# CLI args below override yaml values where they conflict.
if [ "$GPU_COUNT" -gt 1 ]; then
    # Use "python -m accelerate.commands.launch" to avoid PATH issues
    LAUNCHER="python -m accelerate.commands.launch --num_processes=$GPU_COUNT"
else
    LAUNCHER="python"
fi

if [ "$RESUME" = "true" ]; then
    # lerobot resume: --config_path must point to the checkpoint's train_config.json,
    # so that policy_dir = checkpoint/pretrained_model (contains policy_preprocessor.json).
    LAST_CKPT_DIR=$(readlink -f "${OUTPUT_DIR}/checkpoints/last")
    RESUME_CONFIG_PATH="${LAST_CKPT_DIR}/pretrained_model/train_config.json"
    if [ ! -f "$RESUME_CONFIG_PATH" ]; then
        echo "ERROR: Cannot find $RESUME_CONFIG_PATH"
        exit 1
    fi
    echo "  Resuming from checkpoint: $LAST_CKPT_DIR"
    echo "  Using config: $RESUME_CONFIG_PATH"

    $LAUNCHER -m lerobot.scripts.lerobot_train \
        --config_path="${RESUME_CONFIG_PATH}" \
        --dataset.repo_id="univtac/${TASK_NAME}" \
        --dataset.root="${DATASET_DIR}" \
        --output_dir="${OUTPUT_DIR}" \
        --job_name="pi05_${TASK_NAME}_${CONFIG_NAME}_${TIMESTAMP}" \
        --policy.empty_cameras="${EMPTY_CAMERAS}" \
        --rename_map="${RENAME_MAP}" \
        --wandb.project="univtac-pi05-${TASK_NAME}" \
        --resume=true \
        "$@"
else
    $LAUNCHER -m lerobot.scripts.lerobot_train \
        --config_path="${CLEAN_CONFIG}" \
        --dataset.repo_id="univtac/${TASK_NAME}" \
        --dataset.root="${DATASET_DIR}" \
        --output_dir="${OUTPUT_DIR}" \
        --job_name="pi05_${TASK_NAME}_${CONFIG_NAME}_${TIMESTAMP}" \
        --policy.empty_cameras="${EMPTY_CAMERAS}" \
        --rename_map="${RENAME_MAP}" \
        --wandb.project="univtac-pi05-${TASK_NAME}" \
        "$@"
fi

echo ""
echo "Training complete! Model saved to: ${OUTPUT_DIR}"
