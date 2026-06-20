#!/bin/bash
# ─── UniVTAC Pi0.5 OpenPI (JAX) Multi-Task Training Script ───────────────────
#
# Complete workflow for multi-task training:
#   1. Convert multi-task data (HDF5 → merged openpi LeRobot dataset)
#   2. Train on merged dataset with FSDP
#
# Usage:
#   bash policy/Pi05_openpi/sh/multi.sh [gpu_ids] [extra_args...]
#
# Examples:
#   # 4-GPU multi-task training with WandB
#   bash policy/Pi05_openpi/sh/multi.sh 0,1,2,3 --wandb
#
#   # Resume training
#   bash policy/Pi05_openpi/sh/multi.sh 0,1,2,3 --wandb --resume
#
#   # Only convert data (no training)
#   bash policy/Pi05_openpi/sh/multi.sh 0 --compute_norm_stats_only
#
# Prerequisites:
#   - conda activate openpi
#   - cd /data1/zjb/UniVTAC
#
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

GPU_ID="${1:-0,1,2,3}"
shift 1 || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_DIR="$(dirname "$SCRIPT_DIR")"
UNIVTAC_ROOT="$(dirname "$(dirname "$POLICY_DIR")")"

cd "$UNIVTAC_ROOT"

# Step 1: Convert multi-task data if not already done
CONFIG_FILE="${POLICY_DIR}/multitask_config.json"
OUTPUT_ROOT=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['output_root'])")
MERGED_DIR="${OUTPUT_ROOT}/multitask"

if [ ! -d "$MERGED_DIR" ]; then
    echo "Merged multi-task dataset not found at ${MERGED_DIR}"
    echo "Running data conversion..."
    python policy/Pi05_openpi/convert_multitask_to_openpi.py
fi

# Step 2: Multi-task training
echo ""
echo "Starting multi-task training on GPU ${GPU_ID}..."
python policy/Pi05_openpi/train_multitask_openpi.py --gpu "$GPU_ID" "$@"
