#!/bin/bash
task_name=${1}
task_config=${2}
expert_data_num=${3}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ACT_DIR"

python process_data.py $task_name $task_config $expert_data_num
