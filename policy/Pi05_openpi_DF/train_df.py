#!/usr/bin/env python3
"""
UniVTAC Pi0.5 Diffusion Forcing — Training (Single-Task & Multi-Task).

  --task <name> [<name2> ...]   单任务 / 多任务顺序训练
  --task all                    多任务联合训练（使用 multitask_config.json 合并数据集）

Usage:
    # 单任务
    python policy/Pi05_openpi_DF/train_df.py --task lift_bottle --gpu 0

    # 多任务联合（--task all → multitask 合并数据集）
    python policy/Pi05_openpi_DF/train_df.py --task all --gpu 0,1,2,3

    # 覆盖关键参数
    python policy/Pi05_openpi_DF/train_df.py --task lift_bottle --gpu 0 \
        --block_size 5 --mix_prob 0.5 --steps 10000 --lr 2.5e-5

    # 仅计算 norm stats
    python policy/Pi05_openpi_DF/train_df.py --task lift_bottle --compute_norm_stats_only
    python policy/Pi05_openpi_DF/train_df.py --task all --compute_norm_stats_only

Prerequisites:
    单任务: python policy/Pi05_openpi/convert_multitask_openpi.py --task <task>
    多任务: python policy/Pi05_openpi/convert_multitask_to_openpi.py
    安装:   cd /data1/zjb/openpi && pip install -e .

Checkpoint retention:
    - orbax CheckpointManager 硬编码 max_to_keep=1，只保留最新 1 个 ckpt
    - keep_period: 额外永久保留满足 step % keep_period == 0 的 ckpt
    - 若只想保留最新一个，设 keep_period: null（不额外保留任何里程碑）

WandB:
    - 仅记录训练损失等标量指标，不上传模型参数或 artifact
    - 若不使用 WandB，不传 --wandb 即可（默认 disabled）
"""

import argparse
import io
import json
import os
import sys
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

SCRIPT_DIR     = Path(__file__).parent
UNIVTAC_ROOT   = SCRIPT_DIR.parent.parent
OPENPI_ROOT    = Path("/data/zjb/UniVTAC/openpi")
_DATASET_ROOT_DEFAULT = Path("/data/zjb/data/UniVTAC")
DATASET_ROOT   = _DATASET_ROOT_DEFAULT  # overridden by --dataset_root / yaml dataset_root
BASE_FPS       = 60
CKPT_PATH      = Path("/data/zjb/ckpts/pi05_jax")
# 固定 norm_stats 来源：多任务训练集统计，适用于所有单/多任务场景。
DEFAULT_NORM_CKPT = Path("/data/zjb/ckpts/pi05_all_128_20k")
TASK_SETTINGS  = UNIVTAC_ROOT / "policy" / "task_settings.json"
DEFAULT_CFG    = SCRIPT_DIR / "config" / "train_df.yaml"
DEFAULT_MT_CFG = SCRIPT_DIR.parent / "Pi05_openpi" / "multitask_config.json"
OPENPI_CONFIG  = "pi05_univtac_df"


class _TeeStream(io.TextIOBase):
    """Write to both the original stream and a log file simultaneously."""

    def __init__(self, original: io.TextIOBase, log_path: Path):
        self._orig = original
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(log_path, "a", buffering=1, encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, s: str) -> int:
        with self._lock:
            self._orig.write(s)
            self._orig.flush()
            self._file.write(s)
            self._file.flush()
        return len(s)

    def flush(self):
        self._orig.flush()
        self._file.flush()

    @property
    def encoding(self):
        return self._orig.encoding

    @property
    def errors(self):
        return self._orig.errors

    def isatty(self):
        return False

    def close_log(self):
        try:
            self._file.close()
        except Exception:
            pass

TASK_INSTRUCTIONS = {
    "lift_can":            "Pick up the can and place it in the basket.",
    "lift_bottle":         "Pick up the bottle and place it upright.",
    "insert_tube":         "Insert the tube into the connector.",
    "insert_hole":         "Insert the peg into the hole.",
    "insert_HDMI":         "Insert the HDMI cable into the port.",
    "pull_out_key":        "Pull the key out of the lock.",
    "put_bottle_in_shelf": "Place the bottle on the shelf.",
    "grasp_classify":      "Grasp the object and classify its texture.",
    "insert_card":         "Insert the card into the slot.",
    "insert_lean":         "Insert the peg at an angle.",
}

MULTITASK_REPACK_MAP = {
    "observation/image":       "observation.images.head",
    "observation/wrist_image": "observation.images.wrist",
    "observation/state":       "observation.state",
    "actions":                 "action",
    "prompt":                  "task",
}

def _build_multitask_repack_map(use_tactile: bool = False) -> dict:
    """Build repack map for multitask dataset (tac_all format).

    tac_all column names match individual task datasets:
      observation.images.head / wrist / tactile_left / tactile_right
    """
    m = dict(MULTITASK_REPACK_MAP)
    if use_tactile:
        m["observation/tactile_left"]  = "observation.images.tactile_left"
        m["observation/tactile_right"] = "observation.images.tactile_right"
    return m


# ─── helpers ──────────────────────────────────────────────────────────────────

def _load_cfg(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _get_task_settings(task_name: str) -> dict:
    if TASK_SETTINGS.exists():
        with open(TASK_SETTINGS) as f:
            return json.load(f).get(task_name, {})
    return {}


def _build_repack_map(task_name: str) -> dict:
    settings = _get_task_settings(task_name)
    repack = {
        "observation/image":      "observation.images.head",
        "observation/state":      "observation.state",
        "actions":                "action",
        "prompt":                 "task",
    }
    for orig, mapped in settings.get("rename_map", {}).items():
        if mapped == "observation.images.base_0_rgb":
            repack["observation/image"] = orig
        elif mapped == "observation.images.left_wrist_0_rgb":
            repack["observation/wrist_image"] = orig
        elif mapped == "observation.images.right_wrist_0_rgb":
            repack["observation/right_wrist_image"] = orig
    return repack


def _openpi_imports():
    sys.path.insert(0, str(OPENPI_ROOT / "src"))
    import dataclasses
    import importlib.util
    import openpi.training.config as opi_config
    import openpi.training.optimizer as _optimizer
    import openpi.training.weight_loaders as weight_loaders
    import openpi.models.pi0_config as pi0_config
    from openpi import transforms as _transforms
    return dataclasses, importlib, opi_config, _optimizer, weight_loaders, pi0_config, _transforms


def _run_openpi_train(train_config):
    import importlib.util
    spec = importlib.util.spec_from_file_location("openpi_train", OPENPI_ROOT / "scripts" / "train.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.main(train_config)


def _str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def _build_freeze_filter(args):
    """Build an nnx freeze filter (params to *freeze*) from the per-module flags.

    Param-tree conventions (see openpi pi0_config.get_freeze_filter):
      - ".*llm.*"        : all LLM params (PaliGemma prefix expert + action expert)
      - ".*llm.*_1.*"    : the action expert (2nd stacked expert, gemma_300m)
      - ".*img.*"        : the SigLIP image encoder
      - projections      : action_in_proj / action_out_proj / time_mlp_in / time_mlp_out / state_proj
    """
    sys.path.insert(0, str(OPENPI_ROOT / "src"))
    from flax import nnx
    import openpi.shared.nnx_utils as nnx_utils

    llm_all = nnx_utils.PathRegex(".*llm.*")
    action_expert = nnx_utils.PathRegex(".*llm.*_1.*")

    parts = []
    if args.freeze_img:
        parts.append(nnx_utils.PathRegex(".*img.*"))
    if args.freeze_paligemma:
        # PaliGemma prefix expert = all llm params EXCEPT the action expert (_1).
        parts.append(nnx.All(llm_all, nnx.Not(action_expert)))
    if args.freeze_action_expert:
        parts.append(action_expert)
    if args.freeze_projections:
        parts.append(
            nnx_utils.PathRegex(
                ".*(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out|state_proj).*"
            )
        )
    if getattr(args, "freeze_tactile_expert", False):
        # Freeze both the projection layers and the third expert's gemma params
        parts.append(nnx_utils.PathRegex(".*tac_expert.*"))
        parts.append(nnx_utils.PathRegex(".*_2.*"))
    if getattr(args, "sparsh_freeze_backbone", False):
        # Freeze Sparsh ViT backbone; AttentionPool + proj remain trainable.
        # NNX param path: tactile_encoder/backbone/<block_i>/...
        parts.append(nnx_utils.PathRegex(".*tactile_encoder/backbone/.*"))
    if getattr(args, "univtac_freeze_backbone", False):
        # Freeze UniVTAC ResNet-18 backbone conv + BN layers (+ fc when stack_fc=True).
        # proj, spatial_emb, finger_emb remain trainable.
        # NNX param paths: tactile_encoder/stem_conv/*, tactile_encoder/stem_bn/*,
        #                  tactile_encoder/layer{1-4}/*, tactile_encoder/fc/*
        parts.append(nnx_utils.PathRegex(".*tactile_encoder/stem_conv/.*"))
        parts.append(nnx_utils.PathRegex(".*tactile_encoder/stem_bn/.*"))
        parts.append(nnx_utils.PathRegex(".*tactile_encoder/layer[1-4]/.*"))
        parts.append(nnx_utils.PathRegex(".*tactile_encoder/fc/.*"))
    if not parts:
        return nnx.Nothing
    return nnx.Any(*parts)


def _resolve_params_path(params_path: str) -> str:
    """Resolve the params path for warm-start loading.

    Accepts two conventions:
      (a) Direct params dir:  .../params          (contains _METADATA)
      (b) Step dir:           .../16000            (contains params/_METADATA)

    If the given path doesn't have _METADATA but has a 'params' subdirectory
    that does, automatically appends '/params'.
    """
    p = Path(params_path)
    if (p / "_METADATA").exists():
        return str(p)
    params_sub = p / "params"
    if params_sub.is_dir() and (params_sub / "_METADATA").exists():
        print(f"[warm_start] Auto-resolved params path: {params_path} → {params_sub}")
        return str(params_sub)
    return str(p)  # fallback, let downstream handle the error


def _make_weight_loader(params_path: str, use_tactile: bool, use_tactile_expert: bool = False):
    """Build the warm-start weight loader.

    The default ``CheckpointWeightLoader`` only back-fills missing params matching
    ``.*lora.*`` from the freshly-initialized model; everything else must exist in
    the checkpoint or ``check_pytree_equality`` fails. When tactile is enabled the
    ``tactile_encoder`` subtree is absent from a (non-tactile) warm-start ckpt, so we
    widen the missing regex to also let it initialize from scratch. Similarly for
    ``tac_expert_*`` params when the tactile expert is newly added.
    """
    params_path = _resolve_params_path(params_path)
    sys.path.insert(0, str(OPENPI_ROOT / "src"))
    import openpi.training.weight_loaders as weight_loaders

    if not use_tactile and not use_tactile_expert:
        return weight_loaders.CheckpointWeightLoader(params_path)

    import dataclasses as _dc
    import numpy as _np
    import openpi.models.model as _model
    import openpi.shared.download as _download

    # Build missing regex: always allow lora; add tactile modules as needed
    missing_parts = [r".*lora.*"]
    if use_tactile:
        missing_parts.append(r".*tactile_encoder.*")
    if use_tactile_expert:
        # tac_expert_* projection layers
        missing_parts.append(r".*tac_expert.*")
        # Third expert's gemma layers (named with _2 suffix by gemma._name).
        # Keys look like: PaliGemma/llm/layers/pre_attention_norm_2/scale
        # or: PaliGemma/llm/final_norm_2/scale
        # Use .*_2.* to fullmatch any key containing the _2 suffix segment.
        missing_parts.append(r".*_2.*")
    missing_regex = "|".join(missing_parts)

    @_dc.dataclass(frozen=True)
    class _WarmStartWithTactile(weight_loaders.CheckpointWeightLoader):
        def load(self, params):
            loaded = _model.restore_params(
                _download.maybe_download(self.params_path), restore_type=_np.ndarray
            )
            return weight_loaders._merge_params(
                loaded, params, missing_regex=missing_regex
            )

    return _WarmStartWithTactile(params_path)


def _make_optimizer(args):
    import sys
    sys.path.insert(0, str(OPENPI_ROOT / "src"))
    import openpi.training.optimizer as _optimizer
    return _optimizer.AdamW(
        b1=args.adamw_b1,
        b2=args.adamw_b2,
        eps=args.adamw_eps,
        weight_decay=args.adamw_weight_decay,
        clip_gradient_norm=args.adamw_clip_grad_norm,
    )


def _make_model_config(args, num_blocks: int):
    sys.path.insert(0, str(OPENPI_ROOT / "src"))
    import openpi.models.pi0_config as pi0_config
    return pi0_config.Pi0DFConfig(
        pi05=True,
        action_dim=32,
        action_horizon=args.action_horizon,
        num_blocks=num_blocks,
        mix_prob=args.mix_prob,
        block_time_sampling=args.block_time_sampling,
        reweight_gamma=getattr(args, "reweight_gamma", 1.0),
        phase_alpha=getattr(args, "phase_alpha", 1.0),
        use_tactile=getattr(args, "use_tactile", False),
        tactile_tokens_per_finger=getattr(args, "tactile_tokens_per_finger", 16),
        tactile_encoder_type=getattr(args, "tactile_encoder_type", "resnet"),
        sparsh_npz_path=getattr(args, "sparsh_npz_path",
                                "/data/zjb/ckpts/sparsh/dino/sparsh_dino_small_jax.npz"),
        sparsh_freeze_backbone=getattr(args, "sparsh_freeze_backbone", False),
        univtac_encoder_path=getattr(args, "univtac_encoder_path",
                                     "/data/zjb/ckpts/univtac_encoder/univtac_resnet18_jax.npz"),
        univtac_freeze_backbone=getattr(args, "univtac_freeze_backbone", False),
        univtac_stack_fc=getattr(args, "univtac_stack_fc", False),
        use_tactile_expert=getattr(args, "use_tactile_expert", False),
        tactile_expert_variant=getattr(args, "tactile_expert_variant", "gemma_300m"),
        tactile_expert_num_tokens=getattr(args, "tactile_expert_num_tokens", 32),
        tactile_expert_loss_weight=getattr(args, "tactile_expert_loss_weight", 0.5),
        tactile_attend_prefix=getattr(args, "tactile_attend_prefix", True),
        tactile_attend_self=getattr(args, "tactile_attend_self", True),
        use_tactile_register_token=getattr(args, "use_tactile_register_token", False),
        tactile_use_pos_emb=getattr(args, "tactile_use_pos_emb", True),
    )


def _collate(items):
    out = {}
    for k in items[0]:
        vals = [i[k] for i in items]
        if isinstance(vals[0], np.ndarray) and np.issubdtype(vals[0].dtype, np.number):
            out[k] = np.stack(vals)
        elif isinstance(vals[0], (int, float)):
            out[k] = np.array(vals)
    return out


# ─── norm stats ───────────────────────────────────────────────────────────────

def _resolve_norm_stats(args, default_dir: Path, task_name: str, dataset_dir: str | None = None) -> Path:
    """Resolve norm stats directory.

    固定使用 DEFAULT_NORM_CKPT/assets/ 下的 norm_stats（多任务训练集统计，适用于所有单/多任务场景）。
    若该目录不存在则回落到从 parquet 计算。
    """
    _fallback_assets = DEFAULT_NORM_CKPT / "assets"
    if _fallback_assets.exists():
        for sub in sorted(_fallback_assets.iterdir()):
            if sub.is_dir() and (sub / "norm_stats.json").exists():
                print(f"[norm_stats] Using DEFAULT_NORM_CKPT: {sub}")
                return sub

    print(f"[norm_stats] DEFAULT_NORM_CKPT not found, computing for {task_name} ...")
    return _compute_norm_stats_from_parquet(task_name, dataset_dir=dataset_dir)


def _compute_norm_stats_from_parquet(task_name: str, dataset_dir: str | None = None) -> Path:
    """Compute norm stats directly from parquet files, bypassing LeRobotDataset.

    Args:
        task_name: Task name used for output path naming.
        dataset_dir: Root directory of the LeRobot dataset for this task. When None,
            falls back to DATASET_ROOT / task_name (non-tactile default).
    """
    import pyarrow.parquet as pq
    import tqdm as _tqdm
    sys.path.insert(0, str(OPENPI_ROOT / "src"))
    import openpi.shared.normalize as normalize

    data_root = Path(dataset_dir) if dataset_dir else DATASET_ROOT / task_name
    parquet_dir = data_root / "data"
    parquet_files = sorted(parquet_dir.glob("**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir}")

    stats = {k: normalize.RunningStats() for k in ["state", "actions"]}
    for pf in _tqdm.tqdm(parquet_files, desc=f"norm_stats {task_name}"):
        table = pq.read_table(pf)
        df = table.to_pandas()
        for col, key in [("observation.state", "state"), ("action", "actions")]:
            if col in df.columns:
                vals = np.stack(df[col].values)
                stats[key].update(vals)

    norm_stats = {k: s.get_statistics() for k, s in stats.items()}
    out_dir = CKPT_PATH / "assets" / "univtac" / f"univtac_{task_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    normalize.save(out_dir, norm_stats)
    print(f"[norm_stats] Saved → {out_dir}")
    return out_dir


def _resolve_norm_stats_multitask(args, default_dir: Path, dataset_dir: str) -> Path:
    """Resolve norm stats for multi-task.

    固定使用 DEFAULT_NORM_CKPT/assets/ 下的 norm_stats（多任务训练集统计）。
    若该目录不存在则回落到从 parquet 计算。
    """
    _fallback_assets = DEFAULT_NORM_CKPT / "assets"
    if _fallback_assets.exists():
        for sub in sorted(_fallback_assets.iterdir()):
            if sub.is_dir() and (sub / "norm_stats.json").exists():
                print(f"[norm_stats] Using DEFAULT_NORM_CKPT: {sub}")
                return sub
    print(f"[norm_stats] DEFAULT_NORM_CKPT not found, computing for multitask ...")
    return _compute_norm_stats_multitask_from_parquet(dataset_dir)


def _compute_norm_stats_multitask_from_parquet(dataset_dir: str) -> Path:
    """Compute norm stats for multi-task dataset directly from parquet files.

    Supports two layouts:
      (a) Merged dataset:    dataset_dir/data/**/*.parquet
      (b) Per-task layout:   dataset_dir/<task>/data/**/*.parquet  (tac_all style)
    """
    import pyarrow.parquet as pq
    import tqdm as _tqdm
    sys.path.insert(0, str(OPENPI_ROOT / "src"))
    import openpi.shared.normalize as normalize

    root = Path(dataset_dir)
    # Try merged layout first
    merged_data_dir = root / "data"
    if merged_data_dir.is_dir():
        parquet_files = sorted(merged_data_dir.glob("**/*.parquet"))
    else:
        # Per-task layout: <task>/data/**/*.parquet
        parquet_files = sorted(root.glob("*/data/**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found under {dataset_dir} "
            "(checked both merged layout '<root>/data/' and per-task '<root>/<task>/data/')"
        )

    stats = {k: normalize.RunningStats() for k in ["state", "actions"]}
    for pf in _tqdm.tqdm(parquet_files, desc="norm_stats multitask-df"):
        table = pq.read_table(pf)
        df = table.to_pandas()
        for col, key in [("observation.state", "state"), ("action", "actions")]:
            if col in df.columns:
                vals = np.stack(df[col].values)
                stats[key].update(vals)

    norm_stats = {k: s.get_statistics() for k, s in stats.items()}
    out_dir = CKPT_PATH / "assets" / "univtac" / "univtac_multitask_df"
    out_dir.mkdir(parents=True, exist_ok=True)
    normalize.save(out_dir, norm_stats)
    print(f"[norm_stats] Saved → {out_dir}")
    return out_dir


# ─── train ────────────────────────────────────────────────────────────────────

def train_singletask(args, task_name: str):
    global DATASET_ROOT
    DATASET_ROOT = Path(getattr(args, "dataset_root", _DATASET_ROOT_DEFAULT))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    sys.path.insert(0, str(OPENPI_ROOT / "src"))

    import dataclasses
    import openpi.training.config as opi_config
    import openpi.training.optimizer as _optimizer
    import openpi.training.weight_loaders as weight_loaders
    import openpi.models.pi0_config as pi0_config
    from openpi import transforms as _transforms

    num_blocks = args.action_horizon // args.block_size
    assert args.action_horizon % args.block_size == 0, (
        f"action_horizon ({args.action_horizon}) must be divisible by block_size ({args.block_size})"
    )

    prompt = TASK_INSTRUCTIONS.get(task_name, "Perform the manipulation task.")
    repack_map = _build_repack_map(task_name)

    use_tactile = getattr(args, "use_tactile", False)
    # 无论是否启用触觉，均使用 tactile_dataset_root 下的数据集（包含 head/wrist/tactile 全部模态）。
    # use_tactile=False 时只是不把 tactile 列加入 repack_map，不加载触觉图像。
    tac_root = Path(getattr(args, "tactile_dataset_root",
                            "/data/zjb/data/UniVTAC/data_lerobot_openpi_df_tactile"))
    dataset_dir = str(tac_root / task_name)
    repo_id = f"univtac_df_tac/{task_name}"
    if use_tactile:
        repack_map["observation/tactile_left"] = "observation.images.tactile_left"
        repack_map["observation/tactile_right"] = "observation.images.tactile_right"

    norm_asset_id = f"univtac_{task_name}"
    norm_asset_dir = CKPT_PATH / "assets" / "univtac" / norm_asset_id
    actual_norm_dir = _resolve_norm_stats(args, norm_asset_dir, task_name, dataset_dir=dataset_dir)

    # Compute tactile delta_timestamps if needed
    extra_dt = None
    if use_tactile:
        _meta_path = Path(dataset_dir) / "meta" / "info.json"
        if _meta_path.exists():
            with open(_meta_path) as _f:
                _fps = json.load(_f).get("fps", BASE_FPS)
        else:
            _fps = BASE_FPS
        # Each entry loads num_blocks tactile frames at block-boundary timestamps:
        #   t[b] = b * block_size / fps  (b = 0 … num_blocks-1)
        # _select_tactile_for_training picks frame c (current) and c-1 (prev),
        # giving a temporal stride of block_size/fps per adjacent frame pair.
        # With block_size=5, fps=60: stride ≈ 83ms — matches Sparsh DINO's ~80ms
        # pre-training stride.  With block_size=10: stride ≈ 167ms (out-of-dist.).
        extra_dt = {
            "observation.images.tactile_left":  [b * args.block_size / _fps for b in range(num_blocks)],
            "observation.images.tactile_right": [b * args.block_size / _fps for b in range(num_blocks)],
        }

    base = opi_config.get_config(OPENPI_CONFIG)
    data_factory = dataclasses.replace(
        base.data,
        repo_id=repo_id,
        local_root=dataset_dir,
        default_prompt=prompt,
        assets=opi_config.AssetsConfig(
            assets_dir=str(actual_norm_dir.parent),
            asset_id=actual_norm_dir.name,
        ),
        repack_transforms=_transforms.Group(inputs=[_transforms.RepackTransform(repack_map)]),
        extra_delta_timestamps=extra_dt,
    )

    model_config = _make_model_config(args, num_blocks)

    output_base = Path(args.output_dir) if args.output_dir else (UNIVTAC_ROOT / "outputs_openpi_df")

    train_config = dataclasses.replace(
        base,
        exp_name=f"pi05_df_{task_name}",
        project_name=args.wandb_project if args.wandb else "openpi",
        model=model_config,
        data=data_factory,
        freeze_filter=_build_freeze_filter(args),
        weight_loader=_make_weight_loader(
            args.warm_start_ckpt if args.warm_start_ckpt else str(CKPT_PATH / "params"),
            use_tactile,
            use_tactile_expert=getattr(args, "use_tactile_expert", False),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=args.warmup_steps,
            peak_lr=args.lr,
            decay_steps=args.steps,
            decay_lr=args.lr / 10,
        ),
        optimizer=_optimizer.AdamW(
            b1=args.adamw_b1, b2=args.adamw_b2, eps=args.adamw_eps,
            weight_decay=args.adamw_weight_decay,
            clip_gradient_norm=args.adamw_clip_grad_norm,
        ),
        ema_decay=0.99 if args.ema else None,
        fsdp_devices=args.fsdp_devices,
        num_train_steps=args.steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=getattr(args, "prefetch_factor", None),
        seed=args.seed,
        log_interval=args.log_freq,
        save_interval=args.save_freq,
        keep_period=args.keep_period,   # None → 只保留最新 1 个 ckpt
        overwrite=args.overwrite,
        resume=args.resume,
        wandb_enabled=args.wandb,
        checkpoint_base_dir=str(output_base / "checkpoints"),
        assets_base_dir=str(output_base / "assets"),
    )

    # ── tee stdout → log file ──────────────────────────────────────────────────
    log_dir = output_base / "checkpoints" / "pi05_univtac_df" / f"pi05_df_{task_name}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"train_{timestamp}.log"
    tee = _TeeStream(sys.stdout, log_path)
    sys.stdout = tee
    # ──────────────────────────────────────────────────────────────────────────

    try:
        sep = "=" * 60
        print(f"\n{sep}")
        print(f" UniVTAC Pi0.5-DF Single-Task Training")
        print(sep)
        print(f"  Task:           {task_name}")
        print(f"  Prompt:         {prompt}")
        print(f"  GPU:            {args.gpu}")
        print(f"  Batch size:     {args.batch_size}")
        print(f"  Steps:          {args.steps}")
        print(f"  LR:             {args.lr}")
        print(f"  Action horizon: {args.action_horizon}")
        print(f"  Block size:     {args.block_size}  →  num_blocks={num_blocks}")
        print(f"  Mix prob:       {args.mix_prob}")
        print(f"  Block time:     {args.block_time_sampling}")
        print(f"  Freeze:         img={args.freeze_img} paligemma={args.freeze_paligemma} "
              f"action_expert={args.freeze_action_expert} projections={args.freeze_projections}")
        print(f"  Use tactile:    {use_tactile}")
        if use_tactile and getattr(args, "use_tactile_expert", False):
            print(f"  Tactile expert: tokens={args.tactile_expert_num_tokens} "
                  f"loss_weight={args.tactile_expert_loss_weight}")
        print(f"  EMA:            {0.99 if args.ema else 'disabled'}")
        print(f"  keep_period:    {args.keep_period} (None = 只保留最新 ckpt)")
        print(f"  Warm start:     {args.warm_start_ckpt or str(CKPT_PATH / 'params') + ' (pi05_base)'}")
        print(f"  Output:         {output_base}")
        print(f"  Log file:       {log_path}")
        print(sep + "\n")

        _run_openpi_train(train_config)
        print(f"\nTask '{task_name}' training complete!")
    finally:
        sys.stdout = tee._orig
        tee.close_log()


def train_multitask(args):
    global DATASET_ROOT
    DATASET_ROOT = Path(getattr(args, "dataset_root", _DATASET_ROOT_DEFAULT))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    sys.path.insert(0, str(OPENPI_ROOT / "src"))

    import dataclasses
    import openpi.training.config as opi_config
    import openpi.training.optimizer as _optimizer
    import openpi.training.weight_loaders as weight_loaders
    import openpi.models.pi0_config as pi0_config
    from openpi import transforms as _transforms

    with open(args.config) as f:
        mt_config = json.load(f)

    # dataset_dir: 优先使用 --multitask_data_dir（或 yaml multitask_data_dir）；
    # 若未指定则回落到 multitask_config.json 的 output_root/multitask（旧行为）。
    _default_mt_data = getattr(args, "multitask_data_dir", None)
    if _default_mt_data:
        dataset_dir = str(Path(_default_mt_data))
    else:
        output_root = Path(mt_config["output_root"])
        dataset_dir = str(output_root / "multitask")
    if not Path(dataset_dir).exists():
        raise FileNotFoundError(
            f"Multi-task dataset not found at {dataset_dir}.\n"
            "  指定路径: --multitask_data_dir <path>\n"
            "  或先运行 convert_df_tactile.py 生成合并数据集。"
        )

    num_blocks = args.action_horizon // args.block_size
    assert args.action_horizon % args.block_size == 0, (
        f"action_horizon ({args.action_horizon}) must be divisible by block_size ({args.block_size})"
    )

    norm_asset_id = "univtac_multitask_df"
    norm_asset_dir = CKPT_PATH / "assets" / "univtac" / norm_asset_id
    actual_norm_dir = _resolve_norm_stats_multitask(args, norm_asset_dir, dataset_dir)

    _use_tactile_mt = getattr(args, "use_tactile", False)
    _mt_repack = _build_multitask_repack_map(use_tactile=_use_tactile_mt)
    base = opi_config.get_config(OPENPI_CONFIG)
    data_factory = dataclasses.replace(
        base.data,
        repo_id="univtac_df_tac/multitask",
        local_root=dataset_dir,
        default_prompt="Perform the manipulation task.",
        assets=opi_config.AssetsConfig(
            assets_dir=str(actual_norm_dir.parent),
            asset_id=actual_norm_dir.name,
        ),
        repack_transforms=_transforms.Group(inputs=[_transforms.RepackTransform(_mt_repack)]),
    )

    model_config = _make_model_config(args, num_blocks)

    output_base = Path(args.output_dir) if args.output_dir else (UNIVTAC_ROOT / "outputs_openpi_df")
    task_names = list(mt_config["tasks"].keys())
    total_eps = sum(v["num_episodes"] for v in mt_config["tasks"].values())

    train_config = dataclasses.replace(
        base,
        exp_name="pi05_df_multitask",
        project_name=args.wandb_project if args.wandb else "openpi",
        model=model_config,
        data=data_factory,
        freeze_filter=_build_freeze_filter(args),
        weight_loader=_make_weight_loader(
            args.warm_start_ckpt if args.warm_start_ckpt else str(CKPT_PATH / "params"),
            getattr(args, "use_tactile", False),
            use_tactile_expert=getattr(args, "use_tactile_expert", False),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=args.warmup_steps,
            peak_lr=args.lr,
            decay_steps=args.steps,
            decay_lr=args.lr / 10,
        ),
        optimizer=_optimizer.AdamW(
            b1=args.adamw_b1, b2=args.adamw_b2, eps=args.adamw_eps,
            weight_decay=args.adamw_weight_decay,
            clip_gradient_norm=args.adamw_clip_grad_norm,
        ),
        ema_decay=0.99 if args.ema else None,
        fsdp_devices=args.fsdp_devices,
        num_train_steps=args.steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=getattr(args, "prefetch_factor", None),
        seed=args.seed,
        log_interval=args.log_freq,
        save_interval=args.save_freq,
        keep_period=args.keep_period,   # None → 只保留最新 1 个 ckpt
        overwrite=args.overwrite,
        resume=args.resume,
        wandb_enabled=args.wandb,
        checkpoint_base_dir=str(output_base / "checkpoints"),
        assets_base_dir=str(output_base / "assets"),
    )

    sep = "=" * 60
    print(f"\n{sep}")
    print(f" UniVTAC Pi0.5-DF Multi-Task Training")
    print(sep)
    print(f"  Tasks ({len(task_names)}):")
    for t in task_names:
        print(f"    - {t:25s}: {mt_config['tasks'][t]['num_episodes']} eps")
    print(f"  Total episodes:  {total_eps}")
    print(f"  GPU:             {args.gpu}")
    print(f"  Batch size:      {args.batch_size}")
    print(f"  Steps:           {args.steps}")
    print(f"  LR:              {args.lr}")
    print(f"  Action horizon:  {args.action_horizon}")
    print(f"  Block size:      {args.block_size}  →  num_blocks={num_blocks}")
    print(f"  Mix prob:        {args.mix_prob}")
    print(f"  Block time:      {args.block_time_sampling}")
    print(f"  Freeze:          img={args.freeze_img} paligemma={args.freeze_paligemma} "
          f"action_expert={args.freeze_action_expert} projections={args.freeze_projections}")
    print(f"  EMA:             {0.99 if args.ema else 'disabled'}")
    print(f"  FSDP devices:    {args.fsdp_devices}")
    print(f"  keep_period:     {args.keep_period} (None = 只保留最新 ckpt)")
    print(f"  Warm start:      {args.warm_start_ckpt or str(CKPT_PATH / 'params') + ' (pi05_base)'}")
    print(f"  Output:          {output_base}")
    print(sep + "\n")

    _run_openpi_train(train_config)
    print("\nMulti-task DF training complete!")


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    # ── Pre-parse --train_config before building the full parser so that all
    #    subsequent argument defaults are sourced from the user-specified yaml.
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--train_config", default=str(DEFAULT_CFG),
                      help="Path to training yaml config (default: config/train_df.yaml)")
    _pre_args, _ = _pre.parse_known_args()
    cfg_path = Path(_pre_args.train_config)
    cfg = _load_cfg(cfg_path)
    opt = cfg.get("optimizer", {})

    parser = argparse.ArgumentParser(
        description="UniVTAC Pi0.5-DF training (single-task & multi-task)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--task", nargs="+", required=True,
        help=(
            "Task name(s) for single-task training. "
            "Use 'all' to run multi-task joint training on the merged dataset."
        ),
    )
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--train_config", default=str(DEFAULT_CFG),
                        help="Path to training yaml config (default: config/train_df.yaml)")
    parser.add_argument("--dataset_root", type=str,
                        default=cfg.get("dataset_root", str(_DATASET_ROOT_DEFAULT)),
                        help="非触觉数据集根目录，任务数据位于 {dataset_root}/{task_name}（默认 /data/zjb/data/UniVTAC）")
    parser.add_argument("--config", default=str(DEFAULT_MT_CFG),
                        help="Path to multitask_config.json (only used when --task all)")
    parser.add_argument("--multitask_data_dir", type=str,
                        default=cfg.get("multitask_data_dir",
                                        "/data/zjb/data/UniVTAC/tac_all"),
                        help=(
                            "--task all 时使用的多任务合并数据集路径（默认 /data/zjb/data/UniVTAC/tac_all）。"
                            "若未设置则回落到 multitask_config.json 的 output_root/multitask。"
                        ))

    # ── Diffusion Forcing params ──────────────────────────────────────────────
    parser.add_argument("--block_size", type=int, default=cfg.get("block_size", 5),
                        help="Tokens per block; num_blocks = action_horizon // block_size")
    parser.add_argument("--mix_prob", type=float, default=cfg.get("mix_prob", 0.5),
                        help="Prob of block-wise DF noise; 1.0 = full DF, 0.0 = standard flow matching")
    parser.add_argument(
        "--block_time_sampling", type=str,
        default=cfg.get("block_time_sampling", "independent"),
        choices=["independent", "monotone"],
        help=(
            "DF 噪声采样方式: "
            "independent = 每个 block 独立随机噪声等级(原始行为); "
            "monotone = 固定的单调递增噪声等级(靠前 block 更早 clean), "
            "与推理 blockwise 金字塔调度同分布, 推荐用于消除抖动。"
        ),
    )
    parser.add_argument(
        "--reweight_gamma", type=float,
        default=cfg.get("reweight_gamma", 1.0),
        help=(
            "monotone 调度下逆频率 loss 加权的指数: w_k = (nb/(k+1))^gamma, 归一化后 mean=1。"
            "gamma=0 禁用加权；gamma=0.5 缓和；gamma=1.0 完全逆频率（默认）。"
            "仅在 block_time_sampling==monotone 时生效。"
        ),
    )
    parser.add_argument(
        "--phase_alpha", type=float,
        default=cfg.get("phase_alpha", 1.0),
        help=(
            "monotone 调度中 phase 的 Beta(alpha,1) 分布参数。"
            "alpha=1.0 = Uniform(0,1)（默认，无改变）；"
            "alpha<1 使 phase 偏向 0（高噪声区域），增加早期 block 的训练频率和 t>0.5 占比。"
            "推荐: alpha=0.7（温和）或 alpha=0.5（中等）。"
            "仅在 block_time_sampling==monotone 时生效。"
        ),
    )

    # ── Tactile injection ───────────────────────────────────────────────────
    parser.add_argument("--use_tactile", type=_str2bool, default=cfg.get("use_tactile", False),
                        help="启用触觉注入（需要 convert_df_tactile.py 转出的数据集）")
    parser.add_argument("--tactile_tokens_per_finger", type=int,
                        default=cfg.get("tactile_tokens_per_finger", 16),
                        help="每个手指触觉图编码的 token 数（默认 16，左右共 32）")
    parser.add_argument("--tactile_dataset_root", type=str,
                        default=cfg.get("tactile_dataset_root", "/data/zjb/data/UniVTAC/data_lerobot_openpi_df_tactile"),
                        help="触觉数据集根路径")
    parser.add_argument("--tactile_encoder_type", type=str,
                        default=cfg.get("tactile_encoder_type", "resnet"),
                        choices=["resnet", "sparsh", "univtac"],
                        help=(
                            "触觉编码器后端: "
                            "resnet=ResNet-18从零训练(单帧3ch); "
                            "sparsh=Sparsh DINO ViT双帧; "
                            "univtac=UniVTAC预训练ResNet-18(BatchNorm,单帧3ch)"
                        ))
    parser.add_argument("--sparsh_npz_path", type=str,
                        default=cfg.get("sparsh_npz_path",
                                        "/data/zjb/ckpts/sparsh/dino/sparsh_dino_small_jax.npz"),
                        help="Sparsh 预训练权重路径（仅 tactile_encoder_type==sparsh 时使用）")
    parser.add_argument("--sparsh_freeze_backbone", type=_str2bool,
                        default=cfg.get("sparsh_freeze_backbone", False),
                        help="冻结 Sparsh ViT backbone（proj+spatial_emb+finger_emb 始终可训）")
    parser.add_argument("--univtac_encoder_path", type=str,
                        default=cfg.get("univtac_encoder_path",
                                        "/data/zjb/ckpts/univtac_encoder/univtac_resnet18_jax.npz"),
                        help="UniVTAC 预训练 ResNet-18 权重路径（仅 tactile_encoder_type==univtac 时使用）"
                             "由 convert_univtac_encoder_weights.py convert 生成")
    parser.add_argument("--univtac_freeze_backbone", type=_str2bool,
                        default=cfg.get("univtac_freeze_backbone", False),
                        help="冻结 UniVTAC ResNet-18 backbone（proj+spatial_emb+finger_emb 始终可训）")
    parser.add_argument("--univtac_stack_fc", type=_str2bool,
                        default=cfg.get("univtac_stack_fc", False),
                        help=(
                            "true: GlobalAvgPool→预训练fc(512→512)→新proj(512→1024)，输出1 token/指；"
                            "false(默认): 双线性resize→新proj，输出16 token/指（空间模式）。"
                            "需与 tactile_tokens_per_finger=1 配合使用。"
                        ))

    # ── Tactile expert (future tactile prediction) ────────────────────────
    parser.add_argument("--use_tactile_expert", type=_str2bool,
                        default=cfg.get("use_tactile_expert", False),
                        help="启用 tactile expert 预测未来触觉（与 action 同步去噪）")
    parser.add_argument("--tactile_expert_num_tokens", type=int,
                        default=cfg.get("tactile_expert_num_tokens", 32),
                        help="Tactile expert 预测的 token 数")
    parser.add_argument("--tactile_expert_loss_weight", type=float,
                        default=cfg.get("tactile_expert_loss_weight", 0.5),
                        help="触觉预测 velocity loss 权重（相对 action loss）")
    parser.add_argument("--tactile_expert_variant", type=str,
                        default=cfg.get("tactile_expert_variant", "gemma_300m"),
                        help="Tactile expert Transformer 架构 (gemma_300m / gemma_2b)")
    parser.add_argument("--tactile_attend_prefix", type=_str2bool,
                        default=cfg.get("tactile_attend_prefix", True),
                        help=(
                            "true(默认): suffix 中的触觉 token 可以 attend prefix（图像/语言条件）; "
                            "false: 触觉 token 不 attend prefix"
                        ))
    parser.add_argument("--tactile_attend_self", type=_str2bool,
                        default=cfg.get("tactile_attend_self", True),
                        help=(
                            "true(默认): 触觉 token 之间可以互相 attend（双向 self-attention）; "
                            "false: 禁止触觉 token 互相 attend。"
                            "与 tactile_attend_prefix=false 同时设置时，触觉 token 不 attend 任何 token，"
                            "仅被 action token 被动 attend（纯被动 KV 注入模式）。"
                        ))
    parser.add_argument("--use_tactile_register_token", type=_str2bool,
                        default=cfg.get("use_tactile_register_token", False),
                        help=(
                            "仅对 tactile_encoder_type==sparsh 有效。"
                            "true: 将 Sparsh ViT 的 register token（全局 DINO 摘要）作为额外 token 加入 suffix，"
                            "每个手指输出 tactile_tokens_per_finger+1 个 token（共 34 个）; "
                            "false(默认): 丢弃 register token，仅使用 16 个空间 patch token（共 32 个）"
                        ))
    parser.add_argument("--tactile_use_pos_emb", type=_str2bool,
                        default=cfg.get("tactile_use_pos_emb", True),
                        help=(
                            "是否对触觉 token 添加可学习位置编码（spatial_emb + finger_emb）。"
                            "true（默认）: 对每个 token 叠加空间位置编码（4×4 grid）和手指身份编码（左/右）。"
                            "false: 不添加位置编码（消融实验用），两个编码参数不创建。"
                            "此参数为模型架构参数，随 checkpoint 保存，评估时自动继承，不可在推理时切换。"
                        ))

    # ── Module freezing (true = freeze) ───────────────────────────────────────
    fz = cfg.get("freeze", {}) or {}
    parser.add_argument("--freeze_img", type=_str2bool, default=fz.get("img", False),
                        help="冻结 SigLIP 图像编码器 (true=freeze)")
    parser.add_argument("--freeze_paligemma", type=_str2bool, default=fz.get("paligemma", False),
                        help="冻结 PaliGemma 主干 gemma_2b / prefix LLM (true=freeze)")
    parser.add_argument("--freeze_action_expert", type=_str2bool, default=fz.get("action_expert", False),
                        help="冻结 Action Expert gemma_300m (true=freeze)")
    parser.add_argument("--freeze_projections", type=_str2bool, default=fz.get("projections", False),
                        help="冻结 action_in_proj/action_out_proj/time_mlp/state_proj (true=freeze)")
    parser.add_argument("--freeze_tactile_expert", type=_str2bool, default=fz.get("tactile_expert_freeze", False),
                        help="冻结 tactile expert 的 tac_expert_* 层 (true=freeze)")

    # ── Training hyperparams ─────────────────────────────────────────────────
    parser.add_argument("--batch_size",    type=int,   default=cfg.get("batch_size", 4))
    parser.add_argument("--steps",         type=int,   default=cfg.get("steps", 10000))
    parser.add_argument("--lr",            type=float, default=cfg.get("lr", 2.5e-5))
    parser.add_argument("--warmup_steps",  type=int,   default=cfg.get("warmup_steps", 500))
    parser.add_argument("--save_freq",     type=int,   default=cfg.get("save_freq", 1000))
    parser.add_argument("--log_freq",      type=int,   default=cfg.get("log_freq", 50))
    parser.add_argument(
        "--keep_period", type=lambda x: None if x in ("none", "null", "None") else int(x),
        default=cfg.get("keep_period", None),
        help=(
            "Permanently keep checkpoints where step %% keep_period == 0. "
            "Set to null/none to keep only the latest checkpoint (recommended to save disk)."
        ),
    )
    parser.add_argument("--num_workers",    type=int,  default=cfg.get("num_workers", 4))
    parser.add_argument("--prefetch_factor", type=int,  default=cfg.get("prefetch_factor", None),
                        help="DataLoader prefetch_factor per worker (None = PyTorch default 2). "
                             "Values of 4-8 reduce worker stall for large tactile batches.")
    parser.add_argument("--seed",           type=int,  default=cfg.get("seed", 42))
    parser.add_argument("--action_horizon", type=int,  default=cfg.get("action_horizon", 50))
    parser.add_argument("--ema",   action="store_true", default=cfg.get("ema", True))
    parser.add_argument("--no_ema", dest="ema", action="store_false")
    parser.add_argument("--fsdp_devices",  type=int,  default=cfg.get("fsdp_devices", 1),
                        help="FSDP sharding devices (must be power of 2 and divide GPU count)")
    parser.add_argument("--output_dir",    type=str,  default=cfg.get("output_dir", None))
    parser.add_argument(
        "--warm_start_ckpt", type=str, default=None,
        help=(
            "可选：用于 warm start 的 params/ 目录路径，替代默认的 pi05_base。"
            "支持已有的非 DF 微调 ckpt（参数形状兼容）。"
            "例：/data1/zjb/UniVTAC/ckpt/lerobot/pi05_jax/all/128_20k/params"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume",    action="store_true")
    parser.add_argument("--wandb",     action="store_true", default=cfg.get("wandb", False),
                        help="Enable WandB logging (metrics only, no model artifacts)")
    parser.add_argument("--wandb_project", default=cfg.get("wandb_project", "univtac-pi05-df"))
    parser.add_argument("--adamw_b1",           type=float, default=opt.get("b1", 0.9))
    parser.add_argument("--adamw_b2",           type=float, default=opt.get("b2", 0.95))
    parser.add_argument("--adamw_eps",          type=float, default=opt.get("eps", 1e-6))
    parser.add_argument("--adamw_weight_decay", type=float, default=opt.get("weight_decay", 1e-10))
    parser.add_argument("--adamw_clip_grad_norm", type=float, default=opt.get("clip_gradient_norm", 1.0))
    parser.add_argument("--compute_norm_stats_only", action="store_true")

    args = parser.parse_args()

    if args.task == ["all"]:
        # ── multi-task joint training ──────────────────────────────────────
        if args.compute_norm_stats_only:
            with open(args.config) as f:
                mt = json.load(f)
            _compute_norm_stats_multitask_from_parquet(str(Path(mt["output_root"]) / "multitask"))
        else:
            train_multitask(args)
    else:
        # ── single-task training (sequential if multiple tasks given) ──────
        for task in args.task:
            if task not in TASK_INSTRUCTIONS:
                print(f"WARNING: Unknown task '{task}', skipping. "
                      f"Known: {list(TASK_INSTRUCTIONS)}")
                continue
            if args.compute_norm_stats_only:
                tac_root_path = Path(getattr(args, "tactile_dataset_root",
                                             "/data1/zjb/data_lerobot_openpi_df_tactile"))
                ds_dir = str(tac_root_path / task) if getattr(args, "use_tactile", False) \
                    else str(DATASET_ROOT / task)
                _compute_norm_stats_from_parquet(task, dataset_dir=ds_dir)
            else:
                train_singletask(args, task)


if __name__ == "__main__":
    import os, signal, subprocess, time

    _exiting = False

    def _force_exit(signum, frame):
        global _exiting
        if _exiting:
            # 第二次 Ctrl+C: 真正强杀（最后手段）
            print(f"\n[train_df] Force kill (SIGKILL)...", flush=True)
            os._exit(128 + signum)
        _exiting = True

        print(f"\n[train_df] Received signal {signum}, shutting down...", flush=True)
        try:
            # 只杀 DataLoader worker 子进程，不杀自己。
            # 主进程随后走 sys.exit() → Python/JAX 析构 → CUDA context 正常释放。
            mypid = os.getpid()
            pgid  = os.getpgrp()
            # 给同进程组内除自身外的所有进程发 SIGKILL
            result = subprocess.run(
                ["ps", "-o", "pid=", "--no-headers", f"--ppid={mypid}"],
                capture_output=True, text=True, timeout=3,
            )
            child_pids = [int(p) for p in result.stdout.split() if p.strip().isdigit()]
            for cpid in child_pids:
                try:
                    os.kill(cpid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        except Exception:
            pass

        # 让 JAX/XLA 和 CUDA 有机会执行 atexit / 析构函数，正常释放显存
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT,  _force_exit)
    signal.signal(signal.SIGTERM, _force_exit)
    main()
