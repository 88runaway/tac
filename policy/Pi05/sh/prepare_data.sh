#!/bin/bash
# ─── Prepare UniVTAC Data for Pi0.5 Training ─────────────────────────────────
#
# This script:
#   1. Converts UniVTAC HDF5 data to LeRobot format
#   2. (Optional) Computes quantile statistics for QUANTILES normalization
#
# Usage:
#   bash policy/Pi05/sh/prepare_data.sh <task_name|all> [max_episodes]
#
# Examples:
#   bash policy/Pi05/sh/prepare_data.sh lift_can
#   bash policy/Pi05/sh/prepare_data.sh all
#   bash policy/Pi05/sh/prepare_data.sh lift_can 50  # limit to 50 episodes
#
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

TASK="${1:?Usage: $0 <task_name|all> [max_episodes]}"
MAX_EPISODES="${2:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIVTAC_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
LEROBOT_ROOT="/data1/zjb/lerobot"
OUTPUT_DIR="/data1/zjb/lerobot_datasets"

echo "═══════════════════════════════════════════════════════════════"
echo " UniVTAC Data Preparation for Pi0.5"
echo "═══════════════════════════════════════════════════════════════"
echo "  Task:       $TASK"
echo "  Output:     $OUTPUT_DIR"
echo "  Max eps:    ${MAX_EPISODES:-all}"
echo "═══════════════════════════════════════════════════════════════"

# Step 1: Convert to LeRobot format (with 4x subsampling → ~15fps for pi0.5)
echo ""
echo "[Step 1] Converting UniVTAC HDF5 → LeRobot dataset (subsample=4, ~15fps)..."

CONVERT_ARGS="--task $TASK --output_dir $OUTPUT_DIR --subsample 4"
if [ -n "$MAX_EPISODES" ]; then
    CONVERT_ARGS="$CONVERT_ARGS --max_episodes $MAX_EPISODES"
fi

cd "$UNIVTAC_ROOT"
python scripts/convert_to_lerobot.py $CONVERT_ARGS

# Step 2: Compute quantile stats (optional but recommended for pi05)
echo ""
echo "[Step 2] Computing quantile statistics..."

if [ "$TASK" = "all" ]; then
    TASKS=("lift_can" "lift_bottle" "insert_tube" "insert_hole" "insert_HDMI" "pull_out_key" "put_bottle_in_shelf" "grasp_classify")
else
    TASKS=("$TASK")
fi

cd "$LEROBOT_ROOT"
for T in "${TASKS[@]}"; do
    DATASET_DIR="${OUTPUT_DIR}/${T}"
    if [ -d "$DATASET_DIR" ]; then
        echo "  Computing quantile stats for: $T"
        python src/lerobot/scripts/augment_dataset_quantile_stats.py \
            --repo-id "univtac/${T}" \
            --root "$DATASET_DIR" \
            2>/dev/null || echo "  (quantile stats computation skipped or failed for $T, MEAN_STD will be used)"
    fi
done

echo ""
echo "Data preparation complete!"
echo "You can now train with:"
echo "  bash policy/Pi05/sh/train.sh $TASK 0"
