#!/bin/bash
# Convert UniVTAC HDF5 data to RDP zarr format
#
# Usage (raw format - from UniVTAC/data/<task>/<task_config>/hdf5/):
#   bash process_data.sh <task_name> <num_episodes> [downsample] [input_format] [task_config]
#   bash process_data.sh grasp_classify 100
#   bash process_data.sh grasp_classify 100 1
#   bash process_data.sh grasp_classify 100 1 raw contact
#
# Usage (ACT format - from ACT/sh/process_data.sh output):
#   bash process_data.sh <task_name> <num_episodes> 1 act [task_config]
#   bash process_data.sh insert_HDMI 50 1 act demo
#
# Data is read from:
#   raw: $UNIVTAC_DATA/<task_name>/<task_config>/hdf5/
#   act: $UNIVTAC_ROOT/policy/ACT/data/sim-<task>/<task_config>-<num_episodes>/
# where UNIVTAC_DATA defaults to UniVTAC/data (auto-detected from script location)
# Override via: UNIVTAC_DATA=/path/to/data bash process_data.sh ...

task_name=${1:-insert_HDMI}
num_episodes=${2:-100}
downsample=${3:-1}
input_format=${4:-raw}   # raw | act
task_config=${5:-demo}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RDP_ROOT="${RDP_ROOT:-/data1/zjb/reactive_diffusion_policy}"
UNIVTAC_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
UNIVTAC_DATA="${UNIVTAC_DATA:-${UNIVTAC_ROOT}/data}"

OUTPUT_DIR="${RDP_ROOT}/data/univtac_${task_name}_zarr"

if [ "$input_format" = "raw" ]; then
    INPUT_DIR="${UNIVTAC_DATA}/${task_name}/${task_config}/hdf5"
    if [ ! -d "$INPUT_DIR" ]; then
        echo "Error: Raw data directory not found: $INPUT_DIR"
        exit 1
    fi
else
    INPUT_DIR="${UNIVTAC_ROOT}/policy/ACT/data/sim-${task_name}/${task_config}-${num_episodes}"
    if [ ! -d "$INPUT_DIR" ]; then
        echo "Error: ACT data directory not found: $INPUT_DIR"
        echo "Please run ACT/sh/process_data.sh first."
        exit 1
    fi
fi

echo "Converting UniVTAC data to RDP zarr format..."
echo "  Format:      $input_format"
echo "  Task config: $task_config"
echo "  Input:       $INPUT_DIR"
echo "  Output:      $OUTPUT_DIR"
echo "  Episodes:    $num_episodes"
[ "$input_format" = "raw" ] && echo "  Downsample:  $downsample"

python "${RDP_ROOT}/scripts/convert_univtac_to_zarr.py" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --num_episodes "$num_episodes" \
    --input_format "$input_format" \
    --downsample "$downsample"
