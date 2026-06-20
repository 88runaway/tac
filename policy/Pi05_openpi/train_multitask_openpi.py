#!/usr/bin/env python3
"""
UniVTAC Pi0.5 Multi-Task Fine-tuning with OpenPI (JAX) Framework.

Trains a single Pi0.5 model on multiple manipulation tasks simultaneously
using the merged multi-task dataset created by convert_multitask_to_openpi.py.

Usage:
    # Default: use multitask_config.json, train on merged dataset
    python policy/Pi05_openpi/train_multitask_openpi.py --gpu 0,1,2,3

    # Custom hyperparameters
    python policy/Pi05_openpi/train_multitask_openpi.py --gpu 0 \
        --batch_size 4 --steps 30000 --lr 2.5e-5

    # Compute norm stats only (no training)
    python policy/Pi05_openpi/train_multitask_openpi.py --compute_norm_stats_only

    # Resume from checkpoint
    python policy/Pi05_openpi/train_multitask_openpi.py --gpu 0 --resume

Prerequisites:
    1. Convert multi-task data:
       conda activate openpi
       python policy/Pi05_openpi/convert_multitask_to_openpi.py
    2. Install openpi:
       cd /data1/zjb/openpi && pip install -e .
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).parent
UNIVTAC_ROOT = SCRIPT_DIR.parent.parent
OPENPI_ROOT = Path("/data1/zjb/openpi")
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "multitask_config.json"
DEFAULT_TRAIN_YAML = SCRIPT_DIR / "config" / "train_full_openpi_multi.yaml"
CKPT_PATH = Path("/data1/zjb/ckpt/openpi-assets/checkpoints/pi05_base")
TASK_SETTINGS_PATH = UNIVTAC_ROOT / "policy" / "task_settings.json"


def load_train_yaml(yaml_path: Path) -> dict:
    """Load training hyperparameters from yaml config file."""
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def get_task_settings(task_name: str) -> dict:
    if TASK_SETTINGS_PATH.exists():
        with open(TASK_SETTINGS_PATH) as f:
            settings = json.load(f)
        return settings.get(task_name, {})
    return {}


def compute_norm_stats(dataset_dir: str, config: dict):
    """Compute normalization statistics for the merged multi-task dataset."""
    sys.path.insert(0, str(OPENPI_ROOT / "src"))

    import openpi.shared.normalize as normalize
    import openpi.training.config as opi_config
    from openpi import transforms as _transforms
    import dataclasses

    base_config = opi_config.get_config("pi05_univtac")

    repack_map = {
        "observation/base_0_rgb": "observation.images.head",
        "observation/left_wrist_0_rgb": "observation.images.wrist",
        "observation/state": "observation.state",
        "actions": "action",
        "prompt": "task",
    }

    data_factory = dataclasses.replace(
        base_config.data,
        repo_id="univtac/multitask",
        local_root=dataset_dir,
        default_prompt="Perform the manipulation task.",
        repack_transforms=_transforms.Group(
            inputs=[_transforms.RepackTransform(repack_map)]
        ),
    )
    data_config = data_factory.create(base_config.assets_dirs, base_config.model)

    import openpi.training.data_loader as data_loader

    dataset = data_loader.create_torch_dataset(
        data_config, base_config.model.action_horizon, base_config.model
    )
    dataset = data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
        ],
    )

    import torch
    import torch.utils.data

    def _numeric_collate(items):
        """Collate only numeric numpy arrays; skip strings."""
        result = {}
        for key in items[0]:
            vals = [item[key] for item in items]
            if isinstance(vals[0], np.ndarray) and np.issubdtype(vals[0].dtype, np.number):
                result[key] = np.stack(vals, axis=0)
            elif isinstance(vals[0], (int, float)):
                result[key] = np.array(vals)
        return result

    batch_size = 32
    num_batches = len(dataset) // batch_size
    raw_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        collate_fn=_numeric_collate,
    )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    import tqdm as _tqdm
    for i, batch in enumerate(_tqdm.tqdm(raw_loader, total=num_batches, desc="Computing norm stats")):
        if i >= num_batches:
            break
        for key in keys:
            if key in batch:
                stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: s.get_statistics() for key, s in stats.items()}

    output_dir = CKPT_PATH / "assets" / "univtac" / "univtac_multitask"
    output_dir.mkdir(parents=True, exist_ok=True)
    normalize.save(output_dir, norm_stats)
    print(f"Norm stats saved to: {output_dir}")
    return output_dir


def train_multitask(args):
    """Run openpi JAX multi-task training."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    sys.path.insert(0, str(OPENPI_ROOT / "src"))

    import dataclasses
    import openpi.training.config as opi_config
    from openpi import transforms as _transforms
    import openpi.training.optimizer as _optimizer
    import openpi.training.weight_loaders as weight_loaders
    import openpi.models.pi0_config as pi0_config

    config_path = Path(args.config)
    with open(config_path) as f:
        mt_config = json.load(f)

    output_root = Path(mt_config["output_root"])
    dataset_dir = str(output_root / "multitask")

    if not Path(dataset_dir).exists():
        raise FileNotFoundError(
            f"Multi-task dataset not found at {dataset_dir}. "
            "Run convert_multitask_to_openpi.py first."
        )

    norm_asset_id = "univtac_multitask"
    norm_asset_dir = CKPT_PATH / "assets" / "univtac" / norm_asset_id
    if not (norm_asset_dir / "norm_stats.json").exists():
        print(f"Norm stats not found at {norm_asset_dir}, computing...")
        compute_norm_stats(dataset_dir, mt_config)

    base_config = opi_config.get_config("pi05_univtac")

    repack_map = {
        "observation/base_0_rgb": "observation.images.head",
        "observation/left_wrist_0_rgb": "observation.images.wrist",
        "observation/state": "observation.state",
        "actions": "action",
        "prompt": "task",
    }

    data_factory = dataclasses.replace(
        base_config.data,
        repo_id="univtac/multitask",
        local_root=dataset_dir,
        default_prompt="Perform the manipulation task.",
        assets=opi_config.AssetsConfig(
            assets_dir=str(CKPT_PATH / "assets" / "univtac"),
            asset_id=norm_asset_id,
        ),
        repack_transforms=_transforms.Group(
            inputs=[_transforms.RepackTransform(repack_map)]
        ),
    )

    exp_name = "pi05_multitask"
    output_base = Path(args.output_dir) if args.output_dir else (UNIVTAC_ROOT / "outputs_openpi")

    model_config = pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=args.action_horizon,
    )

    train_config = dataclasses.replace(
        base_config,
        exp_name=exp_name,
        project_name=args.wandb_project if args.wandb else "openpi",
        model=model_config,
        data=data_factory,
        weight_loader=weight_loaders.CheckpointWeightLoader(str(CKPT_PATH / "params")),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=args.warmup_steps,
            peak_lr=args.lr,
            decay_steps=args.steps,
            decay_lr=args.lr / 10,
        ),
        optimizer=_optimizer.AdamW(
            b1=args.adamw_b1,
            b2=args.adamw_b2,
            eps=args.adamw_eps,
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
        keep_period=args.keep_period,
        save_best_only=args.save_best_only,
        overwrite=args.overwrite,
        resume=args.resume,
        wandb_enabled=args.wandb,
        checkpoint_base_dir=str(output_base / "checkpoints"),
        assets_base_dir=str(output_base / "assets"),
    )

    task_names = list(mt_config["tasks"].keys())
    total_eps = sum(v["num_episodes"] for v in mt_config["tasks"].values())

    print(f"\n{'='*60}")
    print(f" UniVTAC Pi0.5 Multi-Task OpenPI (JAX) Training")
    print(f"{'='*60}")
    print(f"  Tasks ({len(task_names)}):")
    for t_name in task_names:
        t_cfg = mt_config["tasks"][t_name]
        print(f"    - {t_name:25s}: {t_cfg['num_episodes']} eps")
    print(f"  Total episodes:  {total_eps}")
    print(f"  Dataset:         {dataset_dir}")
    print(f"  GPU:             {args.gpu}")
    print(f"  Batch size:      {args.batch_size}")
    print(f"  Steps:           {args.steps}")
    print(f"  LR:              {args.lr}")
    print(f"  Action horizon:  {args.action_horizon}")
    print(f"  EMA decay:       {0.99 if args.ema else 'disabled'}")
    print(f"  Checkpoint:      {CKPT_PATH}")
    print(f"  Output:          {output_base}")
    print(f"  WandB:           {args.wandb}")
    print(f"  Optimizer (AdamW):")
    print(f"    lr={args.lr:.2e}  warmup={args.warmup_steps}  b1={args.adamw_b1}  b2={args.adamw_b2}")
    print(f"    eps={args.adamw_eps:.1e}  weight_decay={args.adamw_weight_decay:.1e}  clip_grad={args.adamw_clip_grad_norm}")
    print(f"{'='*60}\n")

    import importlib.util
    train_script = OPENPI_ROOT / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("openpi_train", train_script)
    train_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_module)
    train_module.main(train_config)

    print(f"\nMulti-task training complete!")


def main():
    parser = argparse.ArgumentParser(
        description="UniVTAC Pi0.5 Multi-Task OpenPI (JAX) Fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH),
                        help="Path to multitask_config.json")
    parser.add_argument("--train_yaml", type=str, default=str(DEFAULT_TRAIN_YAML),
                        help="Path to training hyperparameter yaml (default: config/train_full_openpi_multi.yaml)")
    parser.add_argument("--gpu", type=str, default="0", help="GPU device ID(s)")

    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--save_freq", type=int, default=None)
    parser.add_argument("--log_freq", type=int, default=None)
    parser.add_argument("--keep_period", type=int, default=None)
    parser.add_argument("--save_best_only", action="store_true", default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action_horizon", type=int, default=None)
    parser.add_argument("--ema", action="store_true", default=True)
    parser.add_argument("--no_ema", dest="ema", action="store_false")

    parser.add_argument("--adamw_b1", type=float, default=None)
    parser.add_argument("--adamw_b2", type=float, default=None)
    parser.add_argument("--adamw_eps", type=float, default=None,
                        help="Adam epsilon; increase from default 1e-8 to e.g. 1e-6 to reduce "
                             "NaN risk in bfloat16 multi-task training.")
    parser.add_argument("--adamw_weight_decay", type=float, default=None)
    parser.add_argument("--adamw_clip_grad_norm", type=float, default=None)

    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: UniVTAC/outputs_openpi)")

    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing checkpoint directory")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint")

    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--fsdp_devices", type=int, default=None,
                        help="Number of FSDP devices (must divide GPU count). "
                             "Set to 1 for single-GPU, or equal to GPU count for full FSDP.")

    parser.add_argument("--compute_norm_stats_only", action="store_true",
                        help="Only compute norm stats, no training")

    args = parser.parse_args()

    train_yaml = load_train_yaml(Path(args.train_yaml))

    if args.batch_size is None:
        args.batch_size = train_yaml.get("batch_size", 4)
    if args.steps is None:
        args.steps = train_yaml.get("steps", 10000)
    if args.lr is None:
        args.lr = float(train_yaml.get("lr", 5e-5))
    if args.warmup_steps is None:
        args.warmup_steps = train_yaml.get("warmup_steps", 1000)
    if args.save_freq is None:
        args.save_freq = train_yaml.get("save_freq", 1000)
    if args.log_freq is None:
        args.log_freq = train_yaml.get("log_freq", 50)
    if args.keep_period is None:
        args.keep_period = train_yaml.get("keep_period", 5000)
    if args.save_best_only is None:
        args.save_best_only = train_yaml.get("save_best_only", True)
    if args.num_workers is None:
        args.num_workers = train_yaml.get("num_workers", 4)
    if args.action_horizon is None:
        args.action_horizon = train_yaml.get("action_horizon", 50)
    if args.wandb_project is None:
        args.wandb_project = train_yaml.get("wandb_project", "univtac-pi05-multitask")
    if args.fsdp_devices is None:
        args.fsdp_devices = train_yaml.get("fsdp_devices", 1)
    if args.output_dir is None:
        args.output_dir = train_yaml.get("output_dir", None)

    _opt_yaml = train_yaml.get("optimizer", {})
    if args.adamw_b1 is None:
        args.adamw_b1 = float(_opt_yaml.get("b1", 0.9))
    if args.adamw_b2 is None:
        args.adamw_b2 = float(_opt_yaml.get("b2", 0.95))
    if args.adamw_eps is None:
        args.adamw_eps = float(_opt_yaml.get("eps", 1e-8))
    if args.adamw_weight_decay is None:
        args.adamw_weight_decay = float(_opt_yaml.get("weight_decay", 1e-10))
    if args.adamw_clip_grad_norm is None:
        args.adamw_clip_grad_norm = float(_opt_yaml.get("clip_gradient_norm", 1.0))

    if args.compute_norm_stats_only:
        config_path = Path(args.config)
        with open(config_path) as f:
            mt_config = json.load(f)
        output_root = Path(mt_config["output_root"])
        dataset_dir = str(output_root / "multitask")
        compute_norm_stats(dataset_dir, mt_config)
    else:
        train_multitask(args)


if __name__ == "__main__":
    main()
