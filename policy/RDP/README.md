# RDP in UniVTAC — 完整流程指南

本文档介绍在 UniVTAC 仿真基准中使用 **Reactive Diffusion Policy (RDP)** 的完整流程，包括数据采集、数据处理、模型训练和策略评估。

---

## 目录结构

```
UniVTAC/policy/RDP/
├── config/
│   └── deploy.yml          # 评测配置（checkpoint 路径、触觉模式等）
├── sh/
│   ├── env.sh              # 环境变量设置
│   ├── process_data.sh     # HDF5 -> zarr 转换（rgb 触觉）
│   ├── process_data_marker.sh  # HDF5 -> zarr 转换（marker PCA 触觉）
│   ├── train_at.sh         # Stage 1: 训练 AT-VAE（rgb 模式）
│   ├── train_at_marker.sh  # Stage 1: 训练 AT-VAE（marker 模式）
│   ├── train_ldp.sh        # Stage 2: 训练 Latent Diffusion Policy（rgb 模式）
│   ├── train_ldp_marker.sh # Stage 2: 训练 Latent Diffusion Policy（marker 模式）
│   ├── train_dp.sh         # 单阶段 Diffusion Policy 训练
│   ├── eval.sh             # 策略评测
│   └── run_pipeline.sh     # 一键端到端 Pipeline
└── deploy_policy.py        # RDP 策略推理适配器（UniVTAC 接口）
```

---

## 策略类型

RDP 支持三种策略配置：

| 策略 | 描述 | 触觉模式 | 阶段数 |
|------|------|----------|--------|
| `dp` | Diffusion Policy | RGB 图像 | 1 |
| `ldp` | Latent Diffusion Policy | RGB 图像 | 2（AT-VAE + LDP）|
| `ldp_marker` | Latent Diffusion Policy | Marker PCA 嵌入 | 2（AT-VAE + LDP）|

> **推荐使用 `ldp` 或 `ldp_marker`**：两阶段模型中的 fast policy 支持实时触觉反馈，推理效果最好。

---

## 环境要求

| 环境 | 用途 |
|------|------|
| `rdp` conda 环境 | 数据处理、模型训练（在 `reactive_diffusion_policy` 仓库中运行）|
| `UniVTAC` conda 环境 | 数据采集、策略评测（在 `UniVTAC` 仓库中运行）|

关键路径默认值（可通过环境变量覆盖）：

```bash
RDP_ROOT=/data1/zjb/reactive_diffusion_policy
UNIVTAC_ROOT=/data1/zjb/UniVTAC
UNIVTAC_CKPT=/data1/zjb/ckpt/UniVTAC   # 原始 HDF5 数据根目录
CKPT_ROOT=/data1/zjb/ckpt/RDP/checkpoints  # 评测用 checkpoint 根目录
```

---

## 完整流程

### 步骤 0：数据采集

使用 UniVTAC 的数据采集脚本收集演示数据，输出原始 HDF5 文件：

```bash
# 在 UniVTAC 环境中
cd /data1/zjb/UniVTAC
bash sh/collect_data.sh <task_name> <num_episodes>
# 输出: /data1/zjb/ckpt/UniVTAC/<task_name>/clean/*.hdf5
```

每个 HDF5 文件包含以下数据：
- `observation/head/rgb`：头部相机 RGB 图像
- `tactile/left_tactile/rgb_marker`、`tactile/right_tactile/rgb_marker`：触觉 RGB 图像
- `tactile/left_tactile/marker`、`tactile/right_tactile/marker`：触觉 marker 坐标（用于 PCA 模式）
- `embodiment/joint`：关节位置（8 维 qpos）
- `actor/joint`：动作序列

---

### 步骤 1：数据处理（HDF5 → zarr）

根据所选触觉模式，选择以下其中一种方式处理数据。

#### 方式 A：RGB 触觉（`dp` / `ldp` 策略）

```bash
conda activate rdp
cd /data1/zjb/UniVTAC

bash policy/RDP/sh/process_data.sh <task_name> <num_episodes> [downsample]
# 示例：
bash policy/RDP/sh/process_data.sh insert_HDMI 100 1

# 输出: $RDP_ROOT/data/univtac_<task_name>_zarr/replay_buffer.zarr
```

#### 方式 B：Marker PCA 触觉（`ldp_marker` 策略）

该脚本自动完成两步：（1）在 marker 数据上训练 PCA；（2）转换为 zarr 并应用 PCA 嵌入。

```bash
conda activate rdp
cd /data1/zjb/UniVTAC

bash policy/RDP/sh/process_data_marker.sh <task_name> <num_episodes> [n_components] [downsample]
# 示例：
bash policy/RDP/sh/process_data_marker.sh insert_HDMI 100 15 1

# 输出:
#   PCA 矩阵: $RDP_ROOT/data/PCA_Transform_UniVTAC_<task_name>/
#   Zarr 数据: $RDP_ROOT/data/univtac_<task_name>_marker_zarr/replay_buffer.zarr
```

---

### 步骤 2：模型训练

#### 路径 A：单阶段 Diffusion Policy（`dp`）

```bash
conda activate rdp
cd /data1/zjb/UniVTAC

bash policy/RDP/sh/train_dp.sh <task_name> [gpu_id]
# 示例：
bash policy/RDP/sh/train_dp.sh insert_HDMI 0

# 输出: $RDP_ROOT/data/outputs/<date>/...train_diffusion_unet_image.../checkpoints/
```

#### 路径 B：两阶段 LDP（`ldp`，RGB 触觉）

**Stage 1：训练 AT-VAE**

```bash
conda activate rdp
cd /data1/zjb/UniVTAC

bash policy/RDP/sh/train_at.sh <task_name> [gpu_id]
# 示例：
bash policy/RDP/sh/train_at.sh insert_HDMI 0

# 输出: $RDP_ROOT/data/outputs/<date>/...train_vae_univtac_at.../checkpoints/
```

**Stage 2：训练 Latent Diffusion Policy**

```bash
bash policy/RDP/sh/train_ldp.sh <task_name> <at_ckpt_dir> [gpu_id]
# 示例：
bash policy/RDP/sh/train_ldp.sh insert_HDMI \
    /data1/zjb/reactive_diffusion_policy/data/outputs/2024.01.01/12.00.00_train_vae_univtac_at/checkpoints \
    0

# 输出: $RDP_ROOT/data/outputs/<date>/...train_latent_diffusion_unet_image.../checkpoints/
```

#### 路径 C：两阶段 LDP（`ldp_marker`，Marker PCA 触觉）

**Stage 1：训练 AT-VAE（marker 模式）**

```bash
conda activate rdp
cd /data1/zjb/UniVTAC

bash policy/RDP/sh/train_at_marker.sh <task_name> [gpu_id]
# 示例：
bash policy/RDP/sh/train_at_marker.sh insert_HDMI 0
```

> 注意：`n_tac_components` 需与数据处理时的 `n_components` 一致（默认 15）。若需修改，请同步更新 `reactive_diffusion_policy/univtac/config/at/at_univtac.yaml` 及 `config/task/univtac_at_marker_emb.yaml` 中的 `n_tac_components`。

**Stage 2：训练 Latent Diffusion Policy（marker 模式）**

```bash
bash policy/RDP/sh/train_ldp_marker.sh <task_name> <at_ckpt_dir> [gpu_id]
# 示例：
bash policy/RDP/sh/train_ldp_marker.sh insert_HDMI \
    /data1/zjb/reactive_diffusion_policy/data/outputs/2024.01.01/12.00.00_train_vae_univtac_at_marker_emb/checkpoints \
    0
```

---

### 步骤 3：策略评测

#### 配置 deploy.yml

在 `policy/RDP/config/deploy.yml` 中确认以下关键参数与训练一致：

```yaml
policy_type: ldp          # dp / ldp / ldp_marker 对应 dp / ldp / ldp_marker
n_obs_steps: 2            # 必须与训练 at_univtac.yaml 的 n_obs_steps 一致
dataset_obs_temporal_downsample_ratio: 1  # 必须与训练配置一致
tactile_mode: rgb         # rgb 或 marker_emb（ldp_marker 使用 marker_emb）
pca_dir: null             # marker_emb 模式需要指定，或通过 PCA_DIR 环境变量设置
```

#### 运行评测

```bash
conda activate UniVTAC
cd /data1/zjb/UniVTAC

source IsaacLab/_isaac_sim/setup_conda_env.sh

# 设置环境变量
export CKPT_CONFIG=univtac          # checkpoint 子目录名，对应 CKPT_ROOT/<task>/<CKPT_CONFIG>/
export CKPT_ROOT=/data1/zjb/ckpt/RDP/checkpoints
export RDP_ROOT=/data1/zjb/reactive_diffusion_policy
export PCA_DIR=/data1/zjb/reactive_diffusion_policy/data/PCA_Transform_UniVTAC_<task_name>  # 仅 marker 模式

CUDA_VISIBLE_DEVICES=0 python scripts/eval_policy.py \
    <task_name> \
    demo \
    RDP/config/deploy \
    --save_video \
    --headless
```

或使用封装好的脚本（在 UniVTAC 环境中运行）：

```bash
bash policy/RDP/sh/eval.sh <task_name> <ckpt_config> [gpu_id] [save_video]
# 示例：
bash policy/RDP/sh/eval.sh insert_HDMI univtac 0 true
```

---

## 一键端到端 Pipeline

`run_pipeline.sh` 脚本封装了数据处理、训练、评测三个步骤，适合从头开始的完整实验：

```bash
cd /data1/zjb/UniVTAC

# LDP（RGB 触觉，推荐）
bash policy/RDP/sh/run_pipeline.sh <task_name> ldp <gpu_id> <num_episodes>

# LDP（Marker PCA 触觉）
bash policy/RDP/sh/run_pipeline.sh <task_name> ldp_marker <gpu_id> <num_episodes>

# 单阶段 DP
bash policy/RDP/sh/run_pipeline.sh <task_name> dp <gpu_id> <num_episodes>

# 示例：
bash policy/RDP/sh/run_pipeline.sh insert_HDMI ldp 0 100
```

支持跳过已完成的步骤：

```bash
# 跳过数据处理（zarr 已存在）
SKIP_DATA=1 bash policy/RDP/sh/run_pipeline.sh insert_HDMI ldp 0 100

# 跳过训练（仅评测）
SKIP_DATA=1 SKIP_TRAIN=1 bash policy/RDP/sh/run_pipeline.sh insert_HDMI ldp 0

# 跳过评测（仅训练）
SKIP_EVAL=1 bash policy/RDP/sh/run_pipeline.sh insert_HDMI ldp 0 100
```

---

## Checkpoint 路径解析规则

`deploy_policy.py` 按以下优先级查找 checkpoint：

1. `deploy.yml` 中的 `ckpt_dir`（显式指定）
2. 环境变量 `CKPT_DIR`
3. `$CKPT_ROOT/<task_name>/$CKPT_CONFIG/checkpoints/latest.ckpt`
4. 兜底：`$RDP_ROOT/ckpt/<task_name>/univtac/`

---

## 推理机制说明

### LDP Reactive Fast 推理（与原 RDP 仓库一致）

当策略为 `ldp` 且使用 RNN 解码器时，每个 action chunk 的执行过程如下：

```
Slow Policy（每个 chunk 推理一次）
  输入：n_obs_steps 帧视觉 + 本体感知历史
  输出：latent_action

Fast Policy（每控制步推理一次）
  输入：latent_action + 当前时刻为止的触觉序列（长度随步数递增）
  输出：完整 action chunk → 取最后一步执行
  → 下一步触觉更新 → 重新解码 → 取最后一步执行 → ...
```

每步取 `action_pred[-1]` 的原因：RNN 解码器按时间步顺序处理触觉历史，最后一步的输出整合了截至当前的所有触觉信息，是最准确的动作预测。

### 数据频率

| 组件 | 数据来源 | 说明 |
|------|----------|------|
| Slow Policy 输入 | 视觉 + qpos 历史（n_obs_steps 帧）| 每个 chunk 更新一次 |
| Fast Policy 输入 | 触觉（extended_obs）| 每个控制步更新一次 |
| 训练数据 | 同一高频数据流 | `dataset_obs_temporal_downsample_ratio` 控制采样比例 |

---

## 常见问题

**Q: 评测时提示找不到 checkpoint？**

检查 `CKPT_ROOT`、`CKPT_CONFIG` 环境变量和 `deploy.yml` 中的 `ckpt_dir`，确保路径下存在 `checkpoints/latest.ckpt` 或其他 `.ckpt` 文件。

**Q: `marker_emb` 模式找不到 PCA 文件？**

确保运行过 `process_data_marker.sh`，并通过 `PCA_DIR` 环境变量或 `deploy.yml` 的 `pca_dir` 字段指定 PCA 矩阵目录（包含 `pca_transform_matrix.npy` 和 `pca_mean_matrix.npy`）。

**Q: `n_tac_components` 不一致导致维度错误？**

`process_data_marker.sh` 的 `n_components` 参数、`train_at_marker.sh` 使用的 YAML 配置（`univtac_at_marker_emb.yaml` 中的 `n_tac_components`）必须保持一致，默认均为 15。

**Q: Stage 2 训练时提示找不到 AT checkpoint？**

`train_ldp.sh` 和 `train_ldp_marker.sh` 的第二个参数需要指向 Stage 1 训练输出的 `checkpoints/` 目录，而不是上层目录。
