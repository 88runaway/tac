#!/bin/bash
# Pi0.5 JAX (openpi) evaluation on UniVTAC benchmark
#
# 所有默认参数从 config/eval.yaml 读取，CLI 参数可按位置覆盖：
#
# Usage:
#   bash policy/Pi05_openpi/sh/eval.sh [task_name] [gpu_id] [save_video] [seed] [total_num]
#
# Examples:
#   bash policy/Pi05_openpi/sh/eval.sh lift_can 1
#   bash policy/Pi05_openpi/sh/eval.sh lift_bottle 0,1,2,3   # 4卡并行，seed自动拆分
#   CKPT_DIR=/data1/zjb/ckpt/lerobot/pi05_jax/all/64_5k bash policy/Pi05_openpi/sh/eval.sh lift_bottle 4
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

# Diffusion Forcing 推理参数（仅对 Pi0DF / *_df checkpoint 生效）
infer_time_schedule=$(_yaml_get infer_time_schedule)
block_size=$(_yaml_get block_size)

# 若启用 blockwise（逐 block 自回归）去噪，必须使用 DF 模型配置（Pi0DFConfig），
# 否则推理服务端会因模型非 DF 而报错。若 config_name 尚未指向 *_df 配置则自动切换。
if [ "${infer_time_schedule}" = "blockwise" ]; then
    case "${config_name}" in
        *_df) ;;
        *)
            echo "[eval] infer_time_schedule=blockwise 需要 DF 模型配置；config_name 自动从 '${config_name}' 切换为 '${config_name}_df'"
            config_name="${config_name}_df"
            ;;
    esac
fi

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

# ─── 构造 eval_policy.py 公共参数 ─────────────────────────────────────────────
EXTRA_ARGS=""
[ "${save_video}" = "true"  ] && EXTRA_ARGS="${EXTRA_ARGS} --save_video"
[ "${save_image}" = "true"  ] && EXTRA_ARGS="${EXTRA_ARGS} --save_image"
[ "${expert_check}" = "true" ] && EXTRA_ARGS="${EXTRA_ARGS} --expert_check"

# ─── 解析 GPU 列表 ────────────────────────────────────────────────────────────
IFS=',' read -ra GPU_LIST <<< "${gpu_id}"
NUM_GPUS=${#GPU_LIST[@]}

# ─── 打印配置摘要 ─────────────────────────────────────────────────────────────
echo "========================================="
echo " Pi0.5 JAX (openpi) Eval on UniVTAC"
echo "========================================="
echo "  Config file:   ${CONFIG_FILE}"
echo "  Task:          ${task_name}"
echo "  Task config:   ${task_config}"
echo "  Ckpt dir:      ${ckpt_dir}"
echo "  Config name:   ${config_name}"
echo "  GPU(s):        ${gpu_id}  (${NUM_GPUS} 卡)"
echo "  Total num:     ${total_num}"
echo "  Instruction:   ${instruction_type}"
  echo "  n_action_steps:       ${n_action_steps:-（使用默认值）}"
  echo "  num_inference_steps:  ${num_inference_steps:-（使用模型默认值 10）}"
  echo "  infer_time_schedule:  ${infer_time_schedule:-const}"
  echo "  block_size:           ${block_size:-（null=使用训练 config 的 num_blocks）}"
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
its = '${infer_time_schedule}'.strip()
if its and its not in ('null', 'None', ''):
    cfg['infer_time_schedule'] = its
bs = '${block_size}'.strip()
if bs and bs not in ('null', 'None', ''):
    cfg['block_size'] = int(bs)
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

# ─── 单卡：原有逻辑 ────────────────────────────────────────────────────────────
if [ "${NUM_GPUS}" -eq 1 ]; then
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

    PYTHONWARNINGS=ignore::UserWarning \
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    python scripts/eval_policy.py \
        "${task_name}" \
        "${TMP_TASK_CONFIG}" \
        "${TMP_DEPLOY}" \
        ${SEED_ARGS} \
        ${EXTRA_ARGS} \
        --headless

# ─── 多卡：按 seed 范围拆分并行 ───────────────────────────────────────────────
else
    # 确定 base seed（与 eval_policy.py 保持一致：start_seed=-1 时取 1000000）
    if [ "${start_seed}" = "-1" ] || [ -z "${start_seed}" ]; then
        BASE_SEED=1000000
    else
        BASE_SEED=${start_seed}
    fi

    # 每卡分配 episode 数（向上取整）
    PER_GPU=$(( (total_num + NUM_GPUS - 1) / NUM_GPUS ))

    echo ""
    echo "多卡并行评估（${NUM_GPUS} 卡），seed 分配如下："

    PIDS=()
    for i in "${!GPU_LIST[@]}"; do
        g="${GPU_LIST[$i]}"
        gpu_start_seed=$(( BASE_SEED + i * PER_GPU ))
        remaining=$(( total_num - i * PER_GPU ))
        gpu_total=$(( remaining < PER_GPU ? remaining : PER_GPU ))
        gpu_max_seed=$(( gpu_start_seed + gpu_total - 1 ))

        # 若用户额外指定了全局 max_seed，则截断
        if [ "${max_seed}" != "-1" ] && [ -n "${max_seed}" ]; then
            if [ "${gpu_start_seed}" -gt "${max_seed}" ]; then
                echo "  GPU ${g}: seed 范围超出 max_seed，跳过"
                continue
            fi
            if [ "${gpu_max_seed}" -gt "${max_seed}" ]; then
                gpu_max_seed=${max_seed}
                gpu_total=$(( gpu_max_seed - gpu_start_seed + 1 ))
            fi
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
    if [ "${FAILED}" -gt 0 ]; then
        echo "警告：${FAILED} 个子进程异常退出，请检查各卡日志。"
    else
        echo "所有 GPU 评估完成。"
    fi
    echo "结果分别保存在各卡对应的 eval_result 子目录中。"
    echo "========================================="
fi
