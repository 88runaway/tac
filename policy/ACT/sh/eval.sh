task_name=${1:-grasp_classify}
model_config=${2:-univtac}   # univtac | vision_only
gpu_id=${3:-5}
save_image=${4:-false}       # true | false
save_video=${5:-true}        # true | false
seed=${6:-}                  # optional: run single seed
vis=${7:-0}                  # 0=task default, 1=force head only, 2=force head+wrist
temporal_agg=${8:-false}          # true | false | (empty=use config default)
task_config=demo

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/select_train_config.sh" ${task_name} ${model_config} ${vis}

# Export camera override so deploy_policy.py can honour --vis
# vis=0: no override (deploy_policy reads task_settings.json as usual)
# vis=1: override to head-only
# vis=2: override to head+wrist
if [ "$vis" = "1" ]; then
    export CAMERA_TYPE_OVERRIDE=head
elif [ "$vis" = "2" ]; then
    export CAMERA_TYPE_OVERRIDE=all
else
    unset CAMERA_TYPE_OVERRIDE
fi

# temporal_agg: "true"/"false" override, empty=use config default
if [ "$temporal_agg" = "true" ]; then
    export TEMPORAL_AGG_OVERRIDE=true
elif [ "$temporal_agg" = "false" ]; then
    export TEMPORAL_AGG_OVERRIDE=false
else
    unset TEMPORAL_AGG_OVERRIDE
fi

export CKPT_CONFIG=${model_config}
export CKPT_ROOT=/data1/zjb/ckpt/UniVTAC/checkpoints
export TORCH_HOME=/data1/zjb/ckpt/UniVTAC/torch
export LD_PRELOAD=/data1/zjb/miniconda3/envs/UniVTAC/lib/libstdc++.so.6
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="8.9"
export CUROBO_NO_JIT=1

SAVE_ARGS=""
[ "$save_image" = "true" ] && SAVE_ARGS="$SAVE_ARGS --save_image"
[ "$save_video" = "true" ] && SAVE_ARGS="$SAVE_ARGS --save_video"

SEED_ARGS=""
if [ -n "$seed" ]; then
    SEED_ARGS="--start_seed ${seed} --max_seed ${seed} --total_num 1"
fi

ROOT_DIR="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
cd "$ROOT_DIR"

source /data1/zjb/UniVTAC/IsaacLab/_isaac_sim/setup_conda_env.sh

PYTHONWARNINGS=ignore::UserWarning \
CUDA_VISIBLE_DEVICES=${gpu_id} python scripts/eval_policy.py \
    ${task_name} \
    ${task_config} \
    ACT/config/deploy \
    ${SAVE_ARGS} \
    ${SEED_ARGS} \
    --headless
