#!/bin/bash
# UniVTAC + RDP 端到端 Pipeline
# 一次完成：数据处理 -> 训练 -> 评测
#
# 支持两种触觉模式：
#   rgb        : RGB 图像触觉，走 DP 或 AT+LDP 两条路径
#   marker_emb : marker PCA 嵌入触觉，走 AT_marker+LDP_marker 路径
#
# 支持三种策略：
#   dp         : 单阶段 Diffusion Policy
#   ldp        : 两阶段 Latent Diffusion Policy（AT-VAE + LDP，rgb 触觉）
#   ldp_marker : 两阶段 LDP，marker PCA 触觉
#
# Usage:
#   bash UniVTAC/policy/RDP/sh/run_pipeline.sh <task_name> [policy] [gpu_id] [num_episodes]
#
# Arguments:
#   task_name     UniVTAC 任务名（如 insert_HDMI, grasp_classify）
#   policy        dp | ldp | ldp_marker   (default: ldp)
#   gpu_id        GPU 编号                 (default: 0)
#   num_episodes  数据集 episode 数量      (default: 100)
#
# Options（环境变量）:
#   RDP_ROOT       reactive_diffusion_policy 仓库路径（默认 /data1/zjb/reactive_diffusion_policy）
#   UNIVTAC_ROOT   UniVTAC 仓库路径（默认 /data1/zjb/UniVTAC）
#   UNIVTAC_DATA   UniVTAC 数据根目录（默认 $UNIVTAC_ROOT/data，即 UniVTAC/data/）
#   CKPT_ROOT      评测用 checkpoint 根目录（默认 /data1/zjb/ckpt/RDP/checkpoints）
#   DOWNSAMPLE     时序下采样倍率（默认 1）
#   N_COMPONENTS   marker PCA 维度（默认 15，ldp_marker 模式）
#   SKIP_DATA      =1 跳过数据处理步骤（复用已有 zarr）
#   SKIP_TRAIN     =1 跳过训练步骤（直接评测）
#   SKIP_EVAL      =1 跳过评测步骤
#   SAVE_VIDEO     =true 评测时保存视频（默认 true）
#   INPUT_FORMAT   raw | act（默认 raw，仅 rgb/dp 模式）
#   HEADLESS       =1 无头模式评测（默认 1）
#   CONDA_BASE     conda 安装路径（默认 /data1/zjb/miniconda3）
#   RDP_CONDA_ENV  数据处理 + 训练环境（默认 rdp）
#   UNIVTAC_CONDA_ENV  评测环境（默认 UniVTAC）
#
# Examples:
#   # LDP 完整流程（rgb 触觉）
# #   # lift_can（双相机，自动使用 dual_cam 配置）
# cd /data1/zjb/UniVTAC
# bash policy/RDP/sh/run_pipeline.sh lift_can ldp_marker 1 100
# bash policy/RDP/sh/run_pipeline.sh insert_tube ldp_marker 2 100
#   # LDP marker 完整流程
#   bash UniVTAC/policy/RDP/sh/run_pipeline.sh insert_HDMI ldp_marker 0 100
#
#   # 单阶段 DP
#   bash UniVTAC/policy/RDP/sh/run_pipeline.sh insert_HDMI dp 0 100
#
#   # 跳过数据处理（数据已存在）
#   SKIP_DATA=1 bash UniVTAC/policy/RDP/sh/run_pipeline.sh insert_HDMI ldp 0
#
#   # 只做评测（训练已完成，手动指定 checkpoint 根目录）
#   SKIP_DATA=1 SKIP_TRAIN=1 CKPT_ROOT=/my/ckpt bash UniVTAC/policy/RDP/sh/run_pipeline.sh insert_HDMI ldp 0

set -euo pipefail

# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------
task_name=${1:?'Usage: run_pipeline.sh <task_name> [policy] [gpu_id] [num_episodes]'}
policy=${2:-ldp}              # dp | ldp | ldp_marker
gpu_id=${3:-0}
num_episodes=${4:-100}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# SCRIPT_DIR = UniVTAC/policy/RDP/sh/
UNIVTAC_POLICY_RDP_DIR="$(dirname "$SCRIPT_DIR")"          # UniVTAC/policy/RDP/
UNIVTAC_ROOT="${UNIVTAC_ROOT:-$(dirname "$(dirname "$UNIVTAC_POLICY_RDP_DIR")")}"   # UniVTAC/

RDP_ROOT="${RDP_ROOT:-/data1/zjb/reactive_diffusion_policy}"
UNIVTAC_DATA="${UNIVTAC_DATA:-${UNIVTAC_ROOT}/data}"
CKPT_ROOT="${CKPT_ROOT:-/data1/zjb/ckpt/RDP/checkpoints}"
DOWNSAMPLE="${DOWNSAMPLE:-1}"
N_COMPONENTS="${N_COMPONENTS:-15}"
SKIP_DATA="${SKIP_DATA:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SAVE_VIDEO="${SAVE_VIDEO:-true}"
INPUT_FORMAT="${INPUT_FORMAT:-raw}"
TASK_CONFIG="${TASK_CONFIG:-demo}"
HEADLESS="${HEADLESS:-1}"
CONDA_BASE="${CONDA_BASE:-/data1/zjb/miniconda3}"
RDP_CONDA_ENV="${RDP_CONDA_ENV:-rdp}"
UNIVTAC_CONDA_ENV="${UNIVTAC_CONDA_ENV:-UniVTAC}"

# 校验 policy 参数
if [[ "$policy" != "dp" && "$policy" != "ldp" && "$policy" != "ldp_marker" ]]; then
    echo "Error: policy must be one of: dp, ldp, ldp_marker (got: $policy)"
    exit 1
fi

# ---------------------------------------------------------------------------
# 路径推导
# ---------------------------------------------------------------------------
if [ "$policy" = "ldp_marker" ]; then
    ZARR_DIR="${RDP_ROOT}/data/univtac_${task_name}_marker_zarr"
    PCA_DIR="${RDP_ROOT}/data/PCA_Transform_UniVTAC_${task_name}"
    TACTILE_MODE="marker_emb"
    EXP_SUFFIX="marker_${task_name}"
else
    ZARR_DIR="${RDP_ROOT}/data/univtac_${task_name}_zarr"
    TACTILE_MODE="rgb"
    EXP_SUFFIX="${task_name}"
    PCA_DIR=""
fi

# sh/ scripts reside alongside this script
SH_DIR="$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
banner() {
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  $1"
    echo "════════════════════════════════════════════════════════"
}

check_step() {
    if [ $? -ne 0 ]; then
        echo ""
        echo "ERROR: Step failed. Aborting pipeline."
        exit 1
    fi
}

find_latest_output_dir() {
    local keyword=$1
    find "${RDP_ROOT}/data/outputs" -maxdepth 2 -type d -name "*${keyword}*" 2>/dev/null \
        | sort | tail -1
}

# 在指定 conda 环境中执行命令
_with_conda() {
    local env_name="$1"
    shift
    if [ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
        echo "Error: conda.sh not found at ${CONDA_BASE}/etc/profile.d/conda.sh"
        echo "  Set CONDA_BASE to your miniconda/anaconda root."
        exit 1
    fi
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${env_name}"
    "$@"
}

run_rdp() {
    _with_conda "${RDP_CONDA_ENV}" "$@"
}

run_univtac_eval() {
    (
        set -euo pipefail
        # shellcheck disable=SC1091
        source "${CONDA_BASE}/etc/profile.d/conda.sh"
        conda activate "${UNIVTAC_CONDA_ENV}"
        cd "${UNIVTAC_ROOT}"
        if [ -f "IsaacLab/_isaac_sim/setup_conda_env.sh" ]; then
            # shellcheck disable=SC1091
            source IsaacLab/_isaac_sim/setup_conda_env.sh
        fi
        "$@"
    )
}

# ---------------------------------------------------------------------------
# 打印配置摘要
# ---------------------------------------------------------------------------
banner "UniVTAC RDP Pipeline"
echo "  Task:         ${task_name}"
echo "  Policy:       ${policy}"
echo "  GPU:          ${gpu_id}"
echo "  Episodes:     ${num_episodes}"
echo "  Tactile mode: ${TACTILE_MODE}"
echo "  RDP_ROOT:     ${RDP_ROOT}"
echo "  UNIVTAC_ROOT: ${UNIVTAC_ROOT}"
echo "  Zarr dir:     ${ZARR_DIR}"
[ "$policy" = "ldp_marker" ] && echo "  PCA dir:      ${PCA_DIR}"
echo "  Skip data:    ${SKIP_DATA}  |  Skip train: ${SKIP_TRAIN}  |  Skip eval: ${SKIP_EVAL}"
echo "  Conda (data/train): ${RDP_CONDA_ENV}"
echo "  Conda (eval):       ${UNIVTAC_CONDA_ENV}"
echo ""

# ===========================================================================
# Step 1: 数据处理
# ===========================================================================
if [ "$SKIP_DATA" = "0" ]; then
    if [ "$policy" = "ldp_marker" ]; then
        banner "Step 1/3  数据处理 (marker PCA)  [conda: ${RDP_CONDA_ENV}]"
        echo "  Input:       ${UNIVTAC_DATA}/${task_name}/${TASK_CONFIG}/hdf5"
        echo "  PCA dir:     ${PCA_DIR}"
        echo "  Output zarr: ${ZARR_DIR}"
        echo "  Components:  ${N_COMPONENTS}"
        run_rdp bash "${SH_DIR}/process_data_marker.sh" \
            "${task_name}" "${num_episodes}" "${N_COMPONENTS}" "${DOWNSAMPLE}" "${TASK_CONFIG}"
        check_step
    else
        banner "Step 1/3  数据处理 (rgb)  [conda: ${RDP_CONDA_ENV}]"
        echo "  Format:      ${INPUT_FORMAT}"
        echo "  Input:       ${UNIVTAC_DATA}/${task_name}/${TASK_CONFIG}/hdf5"
        echo "  Output zarr: ${ZARR_DIR}"
        run_rdp bash "${SH_DIR}/process_data.sh" \
            "${task_name}" "${num_episodes}" "${DOWNSAMPLE}" "${INPUT_FORMAT}" "${TASK_CONFIG}"
        check_step
    fi
else
    banner "Step 1/3  数据处理 [跳过]"
    if [ ! -d "${ZARR_DIR}/replay_buffer.zarr" ]; then
        echo "Warning: Zarr not found at ${ZARR_DIR}/replay_buffer.zarr"
        echo "  If training fails, remove SKIP_DATA=1 and rerun."
    else
        echo "  已有 zarr: ${ZARR_DIR}"
    fi
fi

# ===========================================================================
# Step 2: 训练
# ===========================================================================
if [ "$SKIP_TRAIN" = "0" ]; then

    # ---------- DP（单阶段）-------------------------------------------------
    if [ "$policy" = "dp" ]; then
        banner "Step 2/3  训练 Diffusion Policy  [conda: ${RDP_CONDA_ENV}]"
        echo "  Data: ${ZARR_DIR}  |  GPU: ${gpu_id}"

        run_rdp bash -c "
            cd '${RDP_ROOT}'
            CUDA_VISIBLE_DEVICES=${gpu_id} python train.py \
            --config-name=train_diffusion_unet_real_image_workspace \
            task=univtac_dp \
            task.dataset_path='${ZARR_DIR}' \
            training.device='cuda:0' \
            exp_name=univtac_${EXP_SUFFIX}
        "
        check_step

        TRAIN_OUTPUT_DIR="$(find_latest_output_dir "train_diffusion_unet_image_univtac_dp")"

    # ---------- LDP（两阶段 rgb）--------------------------------------------
    elif [ "$policy" = "ldp" ]; then
        banner "Step 2/3  训练 AT-VAE (Stage 1 / 2)  [conda: ${RDP_CONDA_ENV}]"
        echo "  Data: ${ZARR_DIR}  |  GPU: ${gpu_id}"

        run_rdp bash "${SH_DIR}/train_at.sh" "${task_name}" "${gpu_id}"
        check_step

        AT_OUTPUT_DIR="$(find_latest_output_dir "train_vae_univtac_at")"
        AT_CKPT_DIR="${AT_OUTPUT_DIR}/checkpoints"

        banner "Step 2/3  训练 Latent Diffusion Policy (Stage 2 / 2)  [conda: ${RDP_CONDA_ENV}]"
        echo "  AT checkpoint: ${AT_CKPT_DIR}"

        run_rdp bash "${SH_DIR}/train_ldp.sh" "${task_name}" "${AT_CKPT_DIR}" "${gpu_id}"
        check_step

        TRAIN_OUTPUT_DIR="$(find_latest_output_dir "train_latent_diffusion_unet_image_univtac_ldp")"

    # ---------- LDP marker（两阶段 marker PCA）------------------------------
    elif [ "$policy" = "ldp_marker" ]; then
        banner "Step 2/3  训练 AT-VAE marker (Stage 1 / 2)  [conda: ${RDP_CONDA_ENV}]"
        echo "  Data: ${ZARR_DIR}  |  GPU: ${gpu_id}  |  PCA_K: ${N_COMPONENTS}"

        run_rdp bash "${SH_DIR}/train_at_marker.sh" "${task_name}" "${gpu_id}"
        check_step

        AT_OUTPUT_DIR="$(find_latest_output_dir "train_vae_univtac_at_marker_emb")"
        AT_CKPT_DIR="${AT_OUTPUT_DIR}/checkpoints"

        banner "Step 2/3  训练 Latent Diffusion Policy marker (Stage 2 / 2)  [conda: ${RDP_CONDA_ENV}]"
        echo "  AT checkpoint: ${AT_CKPT_DIR}"

        run_rdp bash "${SH_DIR}/train_ldp_marker.sh" "${task_name}" "${AT_CKPT_DIR}" "${gpu_id}"
        check_step

        TRAIN_OUTPUT_DIR="$(find_latest_output_dir "train_latent_diffusion_unet_image_univtac_ldp_marker_emb")"
    fi

    # 将最终 checkpoint 软链到统一评测目录
    EVAL_CKPT_DIR="${CKPT_ROOT}/${task_name}/univtac_${policy}"
    mkdir -p "$(dirname "$EVAL_CKPT_DIR")"
    if [ -L "$EVAL_CKPT_DIR" ]; then rm "$EVAL_CKPT_DIR"; fi
    ln -s "${TRAIN_OUTPUT_DIR}" "${EVAL_CKPT_DIR}"
    echo ""
    echo "Checkpoint linked: ${EVAL_CKPT_DIR} -> ${TRAIN_OUTPUT_DIR}"

else
    banner "Step 2/3  训练 [跳过]"
    EVAL_CKPT_DIR="${CKPT_ROOT}/${task_name}/univtac_${policy}"
    echo "  使用已有 checkpoint: ${EVAL_CKPT_DIR}"
fi

# ===========================================================================
# Step 3: 评测（UniVTAC conda + IsaacLab）
# ===========================================================================
if [ "$SKIP_EVAL" = "0" ]; then
    banner "Step 3/3  评测  [conda: ${UNIVTAC_CONDA_ENV}]"

    if [ ! -d "$UNIVTAC_ROOT" ]; then
        echo "Error: UNIVTAC_ROOT not found: ${UNIVTAC_ROOT}"
        echo "  Set UNIVTAC_ROOT env var to the UniVTAC repository path."
        exit 1
    fi

    # deploy 配置：policy/RDP/config/deploy（UniVTAC 标准路径）
    DEPLOY_CONFIG="${UNIVTAC_POLICY_RDP_DIR}/config/deploy"

    export CKPT_CONFIG="univtac_${policy}"
    export CKPT_ROOT="${CKPT_ROOT}"
    export RDP_ROOT="${RDP_ROOT}"
    export TORCH_HOME="${TORCH_HOME:-/data1/zjb/ckpt/UniVTAC/torch}"
    export LD_PRELOAD="${LD_PRELOAD:-/data1/zjb/miniconda3/envs/UniVTAC/lib/libstdc++.so.6}"
    export PATH="/usr/local/cuda/bin:${PATH}"
    export CUDA_HOME="/usr/local/cuda"
    export TORCH_CUDA_ARCH_LIST="8.9"
    export CUROBO_NO_JIT=1

    if [ "$policy" = "ldp_marker" ]; then
        export PCA_DIR="${PCA_DIR}"
    fi

    SAVE_ARGS=""
    [ "$SAVE_VIDEO" = "true" ] && SAVE_ARGS="--save_video"
    HEADLESS_ARGS=""
    [ "$HEADLESS" = "1" ] && HEADLESS_ARGS="--headless"

    echo "  Task:         ${task_name}"
    echo "  CKPT_CONFIG:  ${CKPT_CONFIG}"
    echo "  CKPT_ROOT:    ${CKPT_ROOT}"
    echo "  Deploy cfg:   ${DEPLOY_CONFIG}"
    [ "$policy" = "ldp_marker" ] && echo "  PCA_DIR:      ${PCA_DIR}"

    run_univtac_eval \
        env CKPT_CONFIG="${CKPT_CONFIG}" \
            CKPT_ROOT="${CKPT_ROOT}" \
            RDP_ROOT="${RDP_ROOT}" \
            TORCH_HOME="${TORCH_HOME}" \
            LD_PRELOAD="${LD_PRELOAD}" \
            PATH="${PATH}" \
            CUDA_HOME="${CUDA_HOME}" \
            TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
            CUROBO_NO_JIT="${CUROBO_NO_JIT}" \
            ${PCA_DIR:+PCA_DIR="${PCA_DIR}"} \
            PYTHONWARNINGS=ignore::UserWarning \
            CUDA_VISIBLE_DEVICES="${gpu_id}" \
            python scripts/eval_policy.py \
                "${task_name}" \
                univtac \
                "${DEPLOY_CONFIG}" \
                ${SAVE_ARGS} \
                ${HEADLESS_ARGS}
    check_step

else
    banner "Step 3/3  评测 [跳过]"
fi

# ===========================================================================
# 完成
# ===========================================================================
banner "Pipeline 完成"
echo "  Task:   ${task_name}"
echo "  Policy: ${policy}"
[ "$SKIP_DATA"  = "0" ] && echo "  Zarr:   ${ZARR_DIR}"
[ "$SKIP_TRAIN" = "0" ] && echo "  Ckpt:   ${EVAL_CKPT_DIR}"
echo ""
