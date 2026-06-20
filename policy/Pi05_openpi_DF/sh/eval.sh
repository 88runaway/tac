#!/bin/bash
# Pi0.5-DF (Diffusion Forcing) evaluation on UniVTAC benchmark
#
# 所有默认参数从 config/eval_df.yaml 读取，CLI 参数按位置覆盖。
#
# Usage:
#   bash policy/Pi05_openpi_DF/sh/eval.sh [task_name] [gpu_id] [save_video] [seed] [total_num]
#
# Examples:
#   bash policy/Pi05_openpi_DF/sh/eval.sh insert_HDMI 1
#   bash policy/Pi05_openpi_DF/sh/eval.sh insert_HDMI 1,2   # 2卡并行，seed 自动拆分
#   CKPT_DIR=/data1/zjb/UniVTAC/outputs_openpi_df/.../5000 \
#     bash policy/Pi05_openpi_DF/sh/eval.sh insert_HDMI 1

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

CONFIG_FILE="${POLICY_DIR}/config/eval_df.yaml"
BASE_DEPLOY="${POLICY_DIR}/config/deploy_df.yml"

# ─── 解析 yaml helper ─────────────────────────────────────────────────────────
_yaml_get() {
    python3 -c "
import yaml
with open('${CONFIG_FILE}') as f:
    cfg = yaml.safe_load(f)
val = cfg.get('$1', '')
if val is None:
    print('')
elif isinstance(val, bool):
    print('true' if val else 'false')
else:
    print(val)
" 2>/dev/null
}

# ─── 读取默认值；CLI 参数按位置覆盖 ──────────────────────────────────────────
task_name=${1:-$(_yaml_get task_name)}
gpu_id=${2:-$(_yaml_get gpu_id)}
save_video=${3:-$(_yaml_get save_video)}
total_num=${5:-$(_yaml_get total_num)}

task_config=$(_yaml_get task_config)
start_seed=$(_yaml_get start_seed)
max_seed=$(_yaml_get max_seed)
instruction_type=$(_yaml_get instruction_type)
n_action_steps=$(_yaml_get n_action_steps)
num_inference_steps=$(_yaml_get num_inference_steps)
infer_time_schedule=$(_yaml_get infer_time_schedule)
block_size=$(_yaml_get block_size)
decimation=$(_yaml_get decimation)
save_image=$(_yaml_get save_image)
expert_check=$(_yaml_get expert_check)

# CLI 第 4 位传入 seed → 单 seed 评估
if [ -n "${4:-}" ]; then
    start_seed=${4}
    max_seed=${4}
    total_num=1
fi

# CKPT_DIR 环境变量优先，否则读 yaml
ckpt_dir="${CKPT_DIR:-$(_yaml_get ckpt_dir)}"

# ─── 环境变量 ─────────────────────────────────────────────────────────────────
[ -n "${ckpt_dir}" ] && export CKPT_DIR="${ckpt_dir}"

export LD_PRELOAD=/data1/zjb/miniconda3/envs/UniVTAC/lib/libstdc++.so.6
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="8.9"
export CUROBO_NO_JIT=1

# ─── 构造额外参数 ─────────────────────────────────────────────────────────────
EXTRA_ARGS=""
[ "${save_video}"    = "true" ] && EXTRA_ARGS="${EXTRA_ARGS} --save_video"
[ "${save_image}"    = "true" ] && EXTRA_ARGS="${EXTRA_ARGS} --save_image"
[ "${expert_check}"  = "true" ] && EXTRA_ARGS="${EXTRA_ARGS} --expert_check"

# ─── 解析 GPU 列表 ────────────────────────────────────────────────────────────
IFS=',' read -ra GPU_LIST <<< "${gpu_id}"
NUM_GPUS=${#GPU_LIST[@]}

# ─── 打印配置摘要 ─────────────────────────────────────────────────────────────
echo "========================================="
echo " Pi0.5-DF (Diffusion Forcing) Eval"
echo "========================================="
echo "  Config:        ${CONFIG_FILE}"
echo "  Task:          ${task_name}"
echo "  Task config:   ${task_config}"
echo "  Ckpt dir:      ${ckpt_dir}"
echo "  GPU(s):        ${gpu_id}  (${NUM_GPUS} 卡)"
echo "  Total num:     ${total_num}"
echo "  Instruction:   ${instruction_type}"
echo "  n_action_steps:       ${n_action_steps}"
echo "  num_inference_steps:  ${num_inference_steps}"
echo "  infer_time_schedule:  ${infer_time_schedule}"
echo "  block_size:           ${block_size:-null}"
echo "  decimation:           ${decimation}"
echo "  Save video:    ${save_video}"
echo "  Expert check:  ${expert_check}"
echo "========================================="

cd "${ROOT_DIR}"

# ─── 生成临时 deploy 配置（基于 deploy_df.yml，叠加 eval 参数）────────────────
TMP_DEPLOY=$(mktemp /tmp/pi05df_deploy_XXXXXX.yml)
python3 -c "
import yaml
with open('${BASE_DEPLOY}') as f:
    cfg = yaml.safe_load(f)

cfg['ckpt_dir']          = '${ckpt_dir}'
cfg['instruction_type']  = '${instruction_type}'

for key, val in [
    ('n_action_steps',      '${n_action_steps}'),
    ('num_inference_steps', '${num_inference_steps}'),
    ('infer_time_schedule', '${infer_time_schedule}'),
    ('block_size',          '${block_size}'),
]:
    v = val.strip()
    if v and v not in ('null', 'None', ''):
        try:
            cfg[key] = int(v)
        except ValueError:
            cfg[key] = v

with open('${TMP_DEPLOY}', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
"

# ─── 生成临时 task_config ─────────────────────────────────────────────────────
TMP_TASK_CONFIG=$(mktemp /tmp/pi05df_task_config_XXXXXX.yml)
python3 -c "
import yaml
with open('task_config/${task_config}.yml') as f:
    cfg = yaml.safe_load(f)
d = '${decimation}'.strip()
if d and d not in ('null', 'None', ''):
    cfg['decimation'] = int(d)
with open('${TMP_TASK_CONFIG}', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
"

trap "rm -f ${TMP_DEPLOY} ${TMP_TASK_CONFIG}" EXIT

source /data1/zjb/UniVTAC/IsaacLab/_isaac_sim/setup_conda_env.sh

# ─── 单卡评估 ─────────────────────────────────────────────────────────────────
if [ "${NUM_GPUS}" -eq 1 ]; then
    SEED_ARGS=""
    if [ -n "${4:-}" ]; then
        SEED_ARGS="--start_seed ${start_seed} --max_seed ${max_seed} --total_num 1"
    else
        [ "${start_seed}" != "-1" ] && [ -n "${start_seed}" ] && SEED_ARGS="${SEED_ARGS} --start_seed ${start_seed}"
        [ "${max_seed}"   != "-1" ] && [ -n "${max_seed}"   ] && SEED_ARGS="${SEED_ARGS} --max_seed ${max_seed}"
        SEED_ARGS="${SEED_ARGS} --total_num ${total_num}"
    fi

    PYTHONWARNINGS=ignore::UserWarning \
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    python scripts/eval_policy.py \
        "${task_name}" \
        "${TMP_TASK_CONFIG}" \
        "${TMP_DEPLOY}" \
        ${SEED_ARGS} \
        ${EXTRA_ARGS} \
        --headless

# ─── 多卡并行评估（seed 范围自动拆分）────────────────────────────────────────
else
    if [ "${start_seed}" = "-1" ] || [ -z "${start_seed}" ]; then
        BASE_SEED=1000000
    else
        BASE_SEED=${start_seed}
    fi

    PER_GPU=$(( (total_num + NUM_GPUS - 1) / NUM_GPUS ))

    echo ""
    echo "多卡并行评估（${NUM_GPUS} 卡），seed 分配："

    PIDS=()
    for i in "${!GPU_LIST[@]}"; do
        g="${GPU_LIST[$i]}"
        gpu_start_seed=$(( BASE_SEED + i * PER_GPU ))
        remaining=$(( total_num - i * PER_GPU ))
        gpu_total=$(( remaining < PER_GPU ? remaining : PER_GPU ))
        gpu_max_seed=$(( gpu_start_seed + gpu_total - 1 ))

        if [ "${max_seed}" != "-1" ] && [ -n "${max_seed}" ]; then
            [ "${gpu_start_seed}" -gt "${max_seed}" ] && { echo "  GPU ${g}: 超出 max_seed，跳过"; continue; }
            [ "${gpu_max_seed}"   -gt "${max_seed}" ] && { gpu_max_seed=${max_seed}; gpu_total=$(( gpu_max_seed - gpu_start_seed + 1 )); }
        fi

        echo "  GPU ${g}: start_seed=${gpu_start_seed}  max_seed=${gpu_max_seed}  total=${gpu_total}"

        PYTHONWARNINGS=ignore::UserWarning \
        CUDA_VISIBLE_DEVICES="${g}" \
        python scripts/eval_policy.py \
            "${task_name}" \
            "${TMP_TASK_CONFIG}" \
            "${TMP_DEPLOY}" \
            --start_seed "${gpu_start_seed}" \
            --max_seed   "${gpu_max_seed}" \
            --total_num  "${gpu_total}" \
            ${EXTRA_ARGS} \
            --headless &

        PIDS+=($!)
    done

    echo ""
    echo "等待所有 ${#PIDS[@]} 个进程完成..."
    FAILED=0
    for pid in "${PIDS[@]}"; do
        wait "$pid" || FAILED=$(( FAILED + 1 ))
    done

    echo "========================================="
    [ "${FAILED}" -gt 0 ] \
        && echo "警告：${FAILED} 个子进程异常退出，请检查各卡日志。" \
        || echo "所有 GPU 评估完成。"
    echo "========================================="
fi
