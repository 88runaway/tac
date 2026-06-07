#!/bin/bash
# Convert UniVTAC HDF5 data to RDP zarr format using marker PCA embeddings.
#
# Workflow:
#   Step 1 (run once per task): train PCA on raw HDF5 files.
#   Step 2: convert HDF5 -> zarr with PCA embeddings applied.
#
# Usage:
#   bash process_data_marker.sh <task_name> <num_episodes> [n_components] [downsample] [task_config]
#   bash process_data_marker.sh grasp_classify 100
#   bash process_data_marker.sh grasp_classify 100 15
#   bash process_data_marker.sh grasp_classify 100 15 1 contact
#
# Data is read from:
#   $UNIVTAC_DATA/<task_name>/<task_config>/hdf5/
# where UNIVTAC_DATA defaults to UniVTAC/data (auto-detected from script location)
# Override via: UNIVTAC_DATA=/path/to/data bash process_data_marker.sh ...
#
# Whether to include wrist camera is determined automatically from task_settings.json:
#   camera_type = "all"  → dual-cam mode (--wrist_cam is passed to the converter)
#   otherwise            → single-cam mode

task_name=${1:-insert_HDMI}
num_episodes=${2:-100}
n_components=${3:-15}
downsample=${4:-1}
task_config=${5:-demo}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RDP_ROOT="${RDP_ROOT:-/data1/zjb/reactive_diffusion_policy}"
UNIVTAC_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
UNIVTAC_DATA="${UNIVTAC_DATA:-${UNIVTAC_ROOT}/data}"

# Auto-detect dual_cam from task_settings.json
TASK_SETTINGS="${UNIVTAC_ROOT}/policy/task_settings.json"
dual_cam="false"
if [ -f "$TASK_SETTINGS" ]; then
    cam_type=$(python3 -c "
import json, sys
with open('${TASK_SETTINGS}') as f:
    s = json.load(f)
print(s.get('${task_name}', {}).get('camera_type', 'head'))
" 2>/dev/null)
    [ "$cam_type" = "all" ] && dual_cam="true"
fi
echo "  camera_type: ${cam_type:-unknown}  →  dual_cam: ${dual_cam}"

INPUT_DIR="${UNIVTAC_DATA}/${task_name}/${task_config}/hdf5"
PCA_DIR="${RDP_ROOT}/data/PCA_Transform_UniVTAC_${task_name}"
OUTPUT_DIR="${RDP_ROOT}/data/univtac_${task_name}_marker_zarr"

if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Raw data directory not found: $INPUT_DIR"
    exit 1
fi

echo "=== Step 1: Train PCA on marker offsets ==="
echo "  Input HDF5: $INPUT_DIR"
echo "  PCA dir:    $PCA_DIR"
echo "  Components: $n_components"

python "${RDP_ROOT}/scripts/generate_pca_univtac.py" \
    --hdf5_dir "$INPUT_DIR" \
    --output_dir "$PCA_DIR" \
    --n_components "$n_components"

if [ $? -ne 0 ]; then
    echo "Error: PCA training failed."
    exit 1
fi

echo ""
echo "=== Step 2: Convert HDF5 -> zarr with marker PCA embeddings ==="
echo "  Output: $OUTPUT_DIR"

WRIST_ARGS=""
[ "$dual_cam" = "true" ] && WRIST_ARGS="--wrist_cam"

python "${RDP_ROOT}/scripts/convert_univtac_to_zarr.py" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --num_episodes "$num_episodes" \
    --input_format raw \
    --downsample "$downsample" \
    --tactile_mode marker_emb \
    --pca_dir "$PCA_DIR" \
    ${WRIST_ARGS}

if [ $? -ne 0 ]; then
    echo "Error: Zarr conversion failed."
    exit 1
fi

echo ""
echo "Done. Zarr with marker embeddings saved to: $OUTPUT_DIR"
echo "PCA matrices saved to: $PCA_DIR"
