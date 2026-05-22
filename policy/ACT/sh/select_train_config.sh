#!/bin/bash
# Usage: source sh/select_train_config.sh <task_name> <model_config> [vis]
# Sets TRAIN_CONFIG based on camera_type x model_config x vis override:
#
#   vis=0 (default): use camera_type from task_settings.json
#   vis=1:           force head-only  (cam_high only)
#   vis=2:           force all-camera (cam_high + cam_wrist)
#
# Resulting TRAIN_CONFIG matrix:
#   camera=head + univtac      -> train_config
#   camera=head + vision_only  -> train_config_vision
#   camera=all  + univtac      -> train_config_all
#   camera=all  + vision_only  -> train_config_vision_all

_task_name=${1}
_model_config=${2}
_vis=${3:-0}   # 0=default, 1=force head, 2=force all

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_TASK_SETTINGS="$(dirname "$(dirname "$_SCRIPT_DIR")")/task_settings.json"

# Determine effective camera_type
if [ "$_vis" = "1" ]; then
    _camera_type="head"
elif [ "$_vis" = "2" ]; then
    _camera_type="all"
else
    _camera_type=$(python3 -c "
import json
d = json.load(open('$_TASK_SETTINGS'))
print(d.get('$_task_name', {}).get('camera_type', 'head'))
")
fi

if [ "$_model_config" = "vision_only" ]; then
    [ "$_camera_type" = "all" ] && TRAIN_CONFIG=train_config_vision_all || TRAIN_CONFIG=train_config_vision
else
    [ "$_camera_type" = "all" ] && TRAIN_CONFIG=train_config_all || TRAIN_CONFIG=train_config
fi

export TRAIN_CONFIG
_vis_desc="default"
[ "$_vis" = "1" ] && _vis_desc="forced head"
[ "$_vis" = "2" ] && _vis_desc="forced all"
echo -e "\033[36mTask: ${_task_name}, camera_type: ${_camera_type} (vis=${_vis}, ${_vis_desc}) -> TRAIN_CONFIG: ${TRAIN_CONFIG}\033[0m"
