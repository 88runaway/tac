#!/bin/bash
# Pi0.5 evaluation on UniVTAC benchmark
#
# 所有默认参数从 config/eval.yaml 读取，CLI 参数可按位置覆盖：
#
# Usage:
#   bash policy/Pi05/sh/eval.sh [task_name] [ckpt_config] [gpu_id] [save_video] [seed] [total_num]
#
# Examples:
#   CKPT_TIMESTAMP="2026-06-06_18:15:18" bash policy/Pi05/sh/eval.sh lift_bottle  
#   bash policy/Pi05/sh/eval.sh lift_bottle                  # 覆盖 task_name
#   bash policy/Pi05/sh/eval.sh lift_bottle train_lora 0     # 覆盖 task/config/gpu
#   CKPT_DIR=/path/to/ckpt bash policy/Pi05/sh/eval.sh       # 直接指定 checkpoint
#   EVAL_CONFIG=eval3 bash policy/Pi05/sh/eval.sh            # 使用 config/eval1.yaml

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/../config/${EVAL_CONFIG:-eval}.yaml"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# ─── 解析 eval.yaml（使用 python 读取，避免依赖额外工具）─────────────────────
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
ckpt_config=${2:-$(_yaml_get ckpt_config)}
gpu_id=${3:-$(_yaml_get gpu_id)}
save_video=${4:-$(_yaml_get save_video)}
seed=${5:-}
total_num=${6:-$(_yaml_get total_num)}

model=$(_yaml_get model)
model="${model:-vision_only}"

task_config=$(_yaml_get task_config)
start_seed=$(_yaml_get start_seed)
max_seed=$(_yaml_get max_seed)
instruction_type=$(_yaml_get instruction_type)
lerobot_src=$(_yaml_get lerobot_src)
torch_home=$(_yaml_get torch_home)

# CLI 传入的 seed（第5位）覆盖 yaml start_seed/max_seed
if [ -n "${5:-}" ]; then
    start_seed=${5}
    max_seed=${5}
    total_num=1
fi

# 允许剩余参数中的 --model=<value> 覆盖 yaml 中的 model 字段
_remaining_args=()
for _arg in "${@}"; do
    case "$_arg" in
        --model=*) model="${_arg#--model=}" ;;
        *) _remaining_args+=("$_arg") ;;
    esac
done
set -- "${_remaining_args[@]+"${_remaining_args[@]}"}"

# 环境变量可覆盖 ckpt_root / ckpt_dir
ckpt_root="${CKPT_ROOT:-$(_yaml_get ckpt_root)}"
ckpt_dir="${CKPT_DIR:-$(_yaml_get ckpt_dir)}"

expert_check=$(_yaml_get expert_check)
save_image=$(_yaml_get save_image)
n_action_steps=$(_yaml_get n_action_steps)
num_inference_steps=$(_yaml_get num_inference_steps)
ckpt_step=$(_yaml_get ckpt_step)
ckpt_timestamp=$(_yaml_get ckpt_timestamp)
decimation=$(_yaml_get decimation)
rtc_enabled=$(_yaml_get rtc_enabled)
rtc_execution_horizon=$(_yaml_get rtc_execution_horizon)

# ─── 导出评估所需环境变量 ─────────────────────────────────────────────────────
export CKPT_CONFIG="${ckpt_config}"
[ -n "${ckpt_root}"      ] && export CKPT_ROOT="${ckpt_root}"
[ -n "${ckpt_dir}"       ] && export CKPT_DIR="${ckpt_dir}"
[ -n "${ckpt_timestamp}" ] && export CKPT_TIMESTAMP="${ckpt_timestamp}"

export TORCH_HOME="${torch_home}"
export PYTHONPATH="${lerobot_src}:${PYTHONPATH}"
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
if [ -n "${5:-}" ]; then
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
echo " Pi0.5 Evaluation on UniVTAC Benchmark"
echo "========================================="
echo "  Config file:   ${CONFIG_FILE}"
echo "  Model type:    ${model}"
echo "  Task:          ${task_name}"
echo "  Task config:   ${task_config}"
echo "  Ckpt config:   ${ckpt_config}"
[ -n "${ckpt_dir}"  ] && echo "  Ckpt dir:      ${ckpt_dir}"
[ -n "${ckpt_root}" ] && echo "  Ckpt root:     ${ckpt_root}"
echo "  GPU:           ${gpu_id}"
echo "  Total num:     ${total_num}"
echo "  Instruction:   ${instruction_type}"
echo "  n_action_steps:       ${n_action_steps:-（使用 checkpoint 值）}"
echo "  num_inference_steps:  ${num_inference_steps:-（使用 checkpoint 值）}"
echo "  decimation:           ${decimation:-（使用 task_config 默认值）}"
  echo "  ckpt_timestamp: ${ckpt_timestamp:-（自动选最新）}"
  echo "  ckpt_step:      ${ckpt_step:-last（最新）}"
echo "  RTC enabled:    ${rtc_enabled:-false}"
echo "  RTC exec horiz: ${rtc_execution_horizon:-25}"
echo "  Save video:    ${save_video}"
echo "  Save image:    ${save_image}"
echo "  Expert check:  ${expert_check}"
echo "========================================="

# ─── 启动评估 ─────────────────────────────────────────────────────────────────
cd "${ROOT_DIR}"

# 生成带有运行时 instruction_type 的临时 deploy 配置（覆盖 deploy.yml 静态默认值）
TMP_DEPLOY=$(mktemp /tmp/pi05_deploy_XXXXXX.yml)
python3 -c "
import yaml
with open('policy/Pi05/config/deploy.yml') as f:
    cfg = yaml.safe_load(f)
cfg['instruction_type'] = '${instruction_type}'
cfg['model_type'] = '${model}'
n = '${n_action_steps}'.strip()
if n and n not in ('null', 'None', ''):
    cfg['n_action_steps'] = int(n)
ni = '${num_inference_steps}'.strip()
if ni and ni not in ('null', 'None', ''):
    cfg['num_inference_steps'] = int(ni)
s = '${ckpt_step}'.strip()
if s and s not in ('null', 'None', ''):
    cfg['ckpt_step'] = int(s)
rtc = '${rtc_enabled}'.strip().lower()
if rtc in ('true', '1', 'yes'):
    cfg['rtc_enabled'] = True
elif rtc in ('false', '0', 'no', ''):
    cfg['rtc_enabled'] = False
rh = '${rtc_execution_horizon}'.strip()
if rh and rh not in ('null', 'None', ''):
    cfg['rtc_execution_horizon'] = int(rh)
with open('${TMP_DEPLOY}', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
"

# 生成临时 task_config（在原始 task_config 基础上覆盖 decimation）
TMP_TASK_CONFIG=$(mktemp /tmp/pi05_task_config_XXXXXX.yml)
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
