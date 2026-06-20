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
DATASET_ROOT   = Path("/data/zjb/UniVTAC/data_lerobot_openpi")
BASE_FPS       = 60
CKPT_PATH      = Path("/data/zjb/ckpts/pi05_jax")
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
    "prompt":                       "task",
}


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
    if not parts:
        return nnx.Nothing
    return nnx.Any(*parts)


def _make_weight_loader(params_path: str, use_tactile: bool, use_tactile_expert: bool = False):
    """Build the warm-start weight loader.

    The default ``CheckpointWeightLoader`` only back-fills missing params matching
    ``.*lora.*`` from the freshly-initialized model; everything else must exist in
    the checkpoint or ``check_pytree_equality`` fails. When tactile is enabled the
    ``tactile_encoder`` subtree is absent from a (non-tactile) warm-start ckpt, so we
    widen the missing regex to also let it initialize from scratch. Similarly for
    ``tac_expert_*`` params when the tactile expert is newly added.
    """
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
        use_tactile=getattr(args, "use_tactile", False),
        tactile_tokens_per_finger=getattr(args, "tactile_tokens_per_finger", 16),
        use_tactile_expert=getattr(args, "use_tactile_expert", False),
        tactile_expert_variant=getattr(args, "tactile_expert_variant", "gemma_300m"),
        tactile_expert_num_tokens=getattr(args, "tactile_expert_num_tokens", 32),
        tactile_expert_loss_weight=getattr(args, "tactile_expert_loss_weight", 0.5),
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

def _resolve_norm_stats(args, default_dir: Path, task_name: str) -> Path:
    """Resolve norm stats directory: default path → warm start ckpt → compute.

    Search order:
      1. default_dir (e.g. .../assets/univtac/univtac_insert_HDMI)
      2. warm_start_ckpt 旁边的 assets/ 子目录（任意 asset_id）
      3. 直接从 parquet 计算
    """
    # 1. 默认位置
    if (default_dir / "norm_stats.json").exists():
        print(f"[norm_stats] Found at {default_dir}")
        return default_dir

    # 2. warm_start_ckpt 旁边的 assets/
    if args.warm_start_ckpt:
        ckpt_root = Path(args.warm_start_ckpt).parent  # params → ckpt dir
        assets_root = ckpt_root / "assets"
        if assets_root.exists():
            for sub in sorted(assets_root.iterdir()):
                if sub.is_dir() and (sub / "norm_stats.json").exists():
                    print(f"[norm_stats] Reusing from warm start ckpt: {sub}")
                    return sub

    # 3. 计算
    print(f"[norm_stats] Not found, computing for {task_name} ...")
    return _compute_norm_stats_from_parquet(task_name)


def _compute_norm_stats_from_parquet(task_name: str) -> Path:
    """Compute norm stats directly from parquet files, bypassing LeRobotDataset."""
    import pyarrow.parquet as pq
    import tqdm as _tqdm
    sys.path.insert(0, str(OPENPI_ROOT / "src"))
    import openpi.shared.normalize as normalize

    dataset_dir = DATASET_ROOT / task_name / "data"
    parquet_files = sorted(dataset_dir.glob("**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {dataset_dir}")

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
    """Resolve norm stats for multi-task: default path → warm start ckpt → compute."""
    if (default_dir / "norm_stats.json").exists():
        print(f"[norm_stats] Found at {default_dir}")
        return default_dir
    if args.warm_start_ckpt:
        ckpt_root = Path(args.warm_start_ckpt).parent
        assets_root = ckpt_root / "assets"
        if assets_root.exists():
            for sub in sorted(assets_root.iterdir()):
                if sub.is_dir() and (sub / "norm_stats.json").exists():
                    print(f"[norm_stats] Reusing from warm start ckpt: {sub}")
                    return sub
    print(f"[norm_stats] Not found, computing for multitask ...")
    return _compute_norm_stats_multitask_from_parquet(dataset_dir)


def _compute_norm_stats_multitask_from_parquet(dataset_dir: str) -> Path:
    """Compute norm stats for multi-task dataset directly from parquet files."""
    import pyarrow.parquet as pq
    import tqdm as _tqdm
    sys.path.insert(0, str(OPENPI_ROOT / "src"))
    import openpi.shared.normalize as normalize

    data_dir = Path(dataset_dir) / "data"
    parquet_files = sorted(data_dir.glob("**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

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
    if use_tactile:
        tac_root = Path(getattr(args, "tactile_dataset_root", "/data1/zjb/data_lerobot_openpi_df_tactile"))
        dataset_dir = str(tac_root / task_name)
        repo_id = f"univtac_df_tac/{task_name}"
        repack_map["observation/tactile_left"] = "observation.images.tactile_left"
        repack_map["observation/tactile_right"] = "observation.images.tactile_right"
    else:
        dataset_dir = str(DATASET_ROOT / task_name)
        repo_id = f"univtac/{task_name}"

    norm_asset_id = f"univtac_{task_name}"
    norm_asset_dir = CKPT_PATH / "assets" / "univtac" / norm_asset_id
    actual_norm_dir = _resolve_norm_stats(args, norm_asset_dir, task_name)

    # Compute tactile delta_timestamps if needed
    extra_dt = None
    if use_tactile:
        _meta_path = Path(dataset_dir) / "meta" / "info.json"
        if _meta_path.exists():
            with open(_meta_path) as _f:
                _fps = json.load(_f).get("fps", BASE_FPS)
        else:
            _fps = BASE_FPS
        extra_dt = {
            "observation.images.tactile_left": [b * args.block_size / _fps for b in range(num_blocks)],
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

    output_root = Path(mt_config["output_root"])
    dataset_dir = str(output_root / "multitask")
    if not Path(dataset_dir).exists():
        raise FileNotFoundError(
            f"Multi-task dataset not found at {dataset_dir}. "
            "Run convert_multitask_to_openpi.py first."
        )

    num_blocks = args.action_horizon // args.block_size
    assert args.action_horizon % args.block_size == 0, (
        f"action_horizon ({args.action_horizon}) must be divisible by block_size ({args.block_size})"
    )

    norm_asset_id = "univtac_multitask_df"
    norm_asset_dir = CKPT_PATH / "assets" / "univtac" / norm_asset_id
    actual_norm_dir = _resolve_norm_stats_multitask(args, norm_asset_dir, dataset_dir)

    base = opi_config.get_config(OPENPI_CONFIG)
    data_factory = dataclasses.replace(
        base.data,
        repo_id="univtac/multitask",
        local_root=dataset_dir,
        default_prompt="Perform the manipulation task.",
        assets=opi_config.AssetsConfig(
            assets_dir=str(actual_norm_dir.parent),
            asset_id=actual_norm_dir.name,
        ),
        repack_transforms=_transforms.Group(inputs=[_transforms.RepackTransform(MULTITASK_REPACK_MAP)]),
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
    cfg = _load_cfg(DEFAULT_CFG)
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
    parser.add_argument("--config", default=str(DEFAULT_MT_CFG),
                        help="Path to multitask_config.json (only used when --task all)")

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

    # ── Tactile injection ───────────────────────────────────────────────────
    parser.add_argument("--use_tactile", type=_str2bool, default=cfg.get("use_tactile", False),
                        help="启用触觉注入（需要 convert_df_tactile.py 转出的数据集）")
    parser.add_argument("--tactile_tokens_per_finger", type=int,
                        default=cfg.get("tactile_tokens_per_finger", 16),
                        help="每个手指触觉图编码的 token 数（默认 16，左右共 32）")
    parser.add_argument("--tactile_dataset_root", type=str,
                        default=cfg.get("tactile_dataset_root", "/data1/zjb/data_lerobot_openpi_df_tactile"),
                        help="触觉数据集根路径")

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
                _compute_norm_stats_from_parquet(task)
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
