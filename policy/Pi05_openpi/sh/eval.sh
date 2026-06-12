#!/bin/bash
# Pi0.5 JAX (openpi) evaluation on UniVTAC benchmark
#
# 所有默认参数从 config/eval.yaml 读取，CLI 参数可按位置覆盖：
#
# Usage:
#   bash policy/Pi05_openpi/sh/eval.sh [task_name] [gpu_id] [save_video] [seed] [total_num]
#
# Examples:
#   bash policy/Pi05_openpi/sh/eval.sh lift_bottle 0
#   CKPT_DIR=/data1/zjb/ckpt/lerobot/pi05_jax/lift_bottle/9k bash policy/Pi05_openpi/sh/eval.sh
#   EVAL_CONFIG=eval bash policy/Pi05_openpi/sh/eval.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/../config/${EVAL_CONFIG:-eval}.yaml"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# ─── 解析 eval.yaml ─────────────────────────────────────────────────────────
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

# 读取 yaml 默认值，CLI 参数按位置覆盖
task_name=${1:-$(_yaml_get task_name)}
gpu_id=${2:-$(_yaml_get gpu_id)}
save_video=${3:-$(_yaml_get save_video)}
seed=${4:-}
total_num=${5:-$(_yaml_get total_num)}

task_config=$(_yaml_get task_config)
start_seed=$(_yaml_get start_seed)
max_seed=$(_yaml_get max_seed)
instruction_type=$(_yaml_get instruction_type)
n_action_steps=$(_yaml_get n_action_steps)
num_inference_steps=$(_yaml_get num_inference_steps)
decimation=$(_yaml_get decimation)
config_name=$(_yaml_get config_name)
config_name="${config_name:-pi05_univtac}"

# CLI 传入的 seed（第4位）覆盖 yaml start_seed/max_seed
if [ -n "${4:-}" ]; then
    start_seed=${4}
    max_seed=${4}
    total_num=1
fi

# 环境变量可覆盖 ckpt_dir
ckpt_dir="${CKPT_DIR:-$(_yaml_get ckpt_dir)}"

expert_check=$(_yaml_get expert_check)
save_image=$(_yaml_get save_image)

# ─── 导出评估所需环境变量 ─────────────────────────────────────────────────────
[ -n "${ckpt_dir}" ] && export CKPT_DIR="${ckpt_dir}"
export CKPT_CONFIG="${config_name}"

export LD_PRELOAD=/data1/zjb/miniconda3/envs/UniVTAC/lib/libstdc++.so.6
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="8.9"
export CUROBO_NO_JIT=1

# ─── 构造 eval_policy.py 参数 ─────────────────────────────────────────────────
EXTRA_ARGS=""
[ "${save_video}" = "true"  ] && EXTRA_ARGS="${EXTRA_ARGS} --save_video"
[ "${save_image}" = "true"  ] && EXTRA_ARGS="${EXTRA_ARGS} --save_image"
[ "${expert_check}" = "true" ] && EXTRA_ARGS="${EXTRA_ARGS} --expert_check"

SEED_ARGS=""
if [ -n "${4:-}" ]; then
    SEED_ARGS="--start_seed ${start_seed} --max_seed ${max_seed} --total_num 1"
else
    if [ "${start_seed}" != "-1" ] && [ -n "${start_seed}" ]; then
        SEED_ARGS="${SEED_ARGS} --start_seed ${start_seed}"
    fi
    if [ "${max_seed}" != "-1" ] && [ -n "${max_seed}" ]; then
        SEED_ARGS="${SEED_ARGS} --max_seed ${max_seed}"
    fi
    SEED_ARGS="${SEED_ARGS} --total_num ${total_num}"
fi

# ─── 打印配置摘要 ─────────────────────────────────────────────────────────────
echo "========================================="
echo " Pi0.5 JAX (openpi) Eval on UniVTAC"
echo "========================================="
echo "  Config file:   ${CONFIG_FILE}"
echo "  Task:          ${task_name}"
echo "  Task config:   ${task_config}"
echo "  Ckpt dir:      ${ckpt_dir}"
echo "  Config name:   ${config_name}"
echo "  GPU:           ${gpu_id}"
echo "  Total num:     ${total_num}"
echo "  Instruction:   ${instruction_type}"
  echo "  n_action_steps:       ${n_action_steps:-（使用默认值）}"
  echo "  num_inference_steps:  ${num_inference_steps:-（使用模型默认值 10）}"
  echo "  decimation:           ${decimation:-（使用 task_config 默认值）}"
echo "  Save video:    ${save_video}"
echo "  Save image:    ${save_image}"
echo "  Expert check:  ${expert_check}"
echo "========================================="

# ─── 启动评估 ─────────────────────────────────────────────────────────────────
cd "${ROOT_DIR}"

# 生成临时 deploy 配置
TMP_DEPLOY=$(mktemp /tmp/pi05_jax_deploy_XXXXXX.yml)
python3 -c "
import yaml
with open('policy/Pi05_openpi/config/deploy.yml') as f:
    cfg = yaml.safe_load(f)
cfg['instruction_type'] = '${instruction_type}'
cfg['config_name'] = '${config_name}'
n = '${n_action_steps}'.strip()
if n and n not in ('null', 'None', ''):
    cfg['n_action_steps'] = int(n)
ni = '${num_inference_steps}'.strip()
if ni and ni not in ('null', 'None', ''):
    cfg['num_inference_steps'] = int(ni)
with open('${TMP_DEPLOY}', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
"

# 生成临时 task_config
TMP_TASK_CONFIG=$(mktemp /tmp/pi05_jax_task_config_XXXXXX.yml)
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

PYTHONWARNINGS=ignore::UserWarning \
CUDA_VISIBLE_DEVICES="${gpu_id}" \
python scripts/eval_policy.py \
    "${task_name}" \
    "${TMP_TASK_CONFIG}" \
    "${TMP_DEPLOY}" \
    ${SEED_ARGS} \
    ${EXTRA_ARGS} \
    --headless
