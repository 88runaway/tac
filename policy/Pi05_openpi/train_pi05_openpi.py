#!/usr/bin/env python3
"""
UniVTAC Pi0.5 Full Fine-tuning with OpenPI (JAX) Framework.

This script fine-tunes the pi05_base model on UniVTAC manipulation tasks
using the openpi JAX training pipeline.

Usage:
    # Single task
    python policy/Pi05_openpi/train_pi05_openpi.py --task lift_bottle --gpu 0

    # Custom hyperparameters
    python policy/Pi05_openpi/train_pi05_openpi.py --task lift_bottle --gpu 0 \
        --batch_size 4 --steps 10000 --lr 2.5e-5

    # Multiple tasks (trains sequentially)
    python policy/Pi05_openpi/train_pi05_openpi.py --task lift_can insert_tube --gpu 0

    # Compute norm stats only (no training)
    python policy/Pi05_openpi/train_pi05_openpi.py --task lift_bottle --compute_norm_stats_only

    # Resume from checkpoint
    python policy/Pi05_openpi/train_pi05_openpi.py --task lift_bottle --gpu 0 --resume

Prerequisites:
    1. Convert UniVTAC data to openpi-compatible LeRobot v2.1 format:
       conda activate openpi
       python policy/Pi05_openpi/convert_multitask_openpi.py --task <task>
    2. Install openpi:
       cd /data1/zjb/openpi && pip install -e .
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent
UNIVTAC_ROOT = SCRIPT_DIR.parent.parent
OPENPI_ROOT = Path("/data1/zjb/openpi")
DATASET_ROOT = Path("/data1/zjb/data_lerobot_openpi")
CKPT_PATH = Path("/data1/zjb/ckpt/openpi-assets/checkpoints/pi05_base")
TASK_SETTINGS_PATH = UNIVTAC_ROOT / "policy" / "task_settings.json"

TASK_INSTRUCTIONS = {
    "lift_can": "Pick up the can and place it in the basket.",
    "lift_bottle": "Pick up the bottle and place it upright.",
    "insert_tube": "Insert the tube into the connector.",
    "insert_hole": "Insert the peg into the hole.",
    "insert_HDMI": "Insert the HDMI cable into the port.",
    "pull_out_key": "Pull the key out of the lock.",
    "put_bottle_in_shelf": "Place the bottle on the shelf.",
    "grasp_classify": "Grasp the object and classify its texture.",
    "insert_card": "Insert the card into the slot.",
    "insert_lean": "Insert the peg at an angle.",
}


def get_task_settings(task_name: str) -> dict:
    if TASK_SETTINGS_PATH.exists():
        with open(TASK_SETTINGS_PATH) as f:
            settings = json.load(f)
        return settings.get(task_name, {})
    return {}


def build_repack_map(task_name: str, model_type: str = "vision_only") -> dict:
    """Build the repack transform map based on task camera configuration."""
    settings = get_task_settings(task_name)

    repack = {
        "observation/base_0_rgb": "observation.images.head",
        "observation/state": "observation.state",
        "actions": "action",
        "prompt": "task",
    }

    if model_type == "tactile":
        rename_map = settings.get("tactile_rename_map", {})
    else:
        rename_map = settings.get("rename_map", {})

    repack_inv = {}
    for orig_key, mapped_key in rename_map.items():
        if mapped_key == "observation.images.base_0_rgb":
            repack_inv["observation/base_0_rgb"] = orig_key
        elif mapped_key == "observation.images.left_wrist_0_rgb":
            repack_inv["observation/left_wrist_0_rgb"] = orig_key
        elif mapped_key == "observation.images.right_wrist_0_rgb":
            repack_inv["observation/right_wrist_0_rgb"] = orig_key

    final_repack = {}
    for target_key, default_src in repack.items():
        if target_key in repack_inv:
            final_repack[target_key] = repack_inv[target_key]
        else:
            final_repack[target_key] = default_src

    return final_repack


def compute_norm_stats(task_name: str, model_type: str = "vision_only"):
    """Compute normalization statistics for a task dataset."""
    sys.path.insert(0, str(OPENPI_ROOT / "src"))

    import openpi.shared.normalize as normalize
    import openpi.training.config as config

    config_name = "pi05_univtac"
    train_config = config.get_config(config_name)

    repack_map = build_repack_map(task_name, model_type)
    prompt = TASK_INSTRUCTIONS.get(task_name, "Perform the manipulation task.")

    from openpi import transforms as _transforms
    import dataclasses

    dataset_dir = str(DATASET_ROOT / task_name)
    data_factory = dataclasses.replace(
        train_config.data,
        repo_id=f"univtac/{task_name}",
        local_root=dataset_dir,
        default_prompt=prompt,
        repack_transforms=_transforms.Group(
            inputs=[_transforms.RepackTransform(repack_map)]
        ),
    )
    data_config = data_factory.create(train_config.assets_dirs, train_config.model)

    import openpi.training.data_loader as data_loader

    dataset = data_loader.create_torch_dataset(
        data_config, train_config.model.action_horizon, train_config.model
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
        shuffle=False,
        num_workers=2,
        drop_last=True,
        collate_fn=_numeric_collate,
    )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    import tqdm
    for i, batch in enumerate(tqdm.tqdm(raw_loader, total=num_batches, desc=f"Computing norm stats for {task_name}")):
        if i >= num_batches:
            break
        for key in keys:
            if key in batch:
                stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: s.get_statistics() for key, s in stats.items()}

    output_dir = CKPT_PATH / "assets" / "univtac" / f"univtac_{task_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    normalize.save(output_dir, norm_stats)
    print(f"Norm stats saved to: {output_dir}")
    return output_dir


def train_task(args, task_name: str):
    """Run openpi JAX training for a single task."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    sys.path.insert(0, str(OPENPI_ROOT / "src"))

    import dataclasses
    import openpi.training.config as config
    from openpi import transforms as _transforms
    import openpi.training.optimizer as _optimizer
    import openpi.training.weight_loaders as weight_loaders

    settings = get_task_settings(task_name)
    repack_map = build_repack_map(task_name, args.model_type)
    prompt = TASK_INSTRUCTIONS.get(task_name, "Perform the manipulation task.")

    norm_asset_id = f"univtac_{task_name}"
    norm_asset_dir = CKPT_PATH / "assets" / "univtac" / norm_asset_id
    if not (norm_asset_dir / "norm_stats.json").exists():
        print(f"Norm stats not found at {norm_asset_dir}, computing...")
        compute_norm_stats(task_name, args.model_type)

    base_config = config.get_config("pi05_univtac")

    dataset_dir = str(DATASET_ROOT / task_name)
    data_factory = dataclasses.replace(
        base_config.data,
        repo_id=f"univtac/{task_name}",
        local_root=dataset_dir,
        default_prompt=prompt,
        assets=config.AssetsConfig(
            assets_dir=str(CKPT_PATH / "assets" / "univtac"),
            asset_id=norm_asset_id,
        ),
        repack_transforms=_transforms.Group(
            inputs=[_transforms.RepackTransform(repack_map)]
        ),
    )

    exp_name = f"pi05_{task_name}_full"
    output_base = Path(args.output_dir) if args.output_dir else (UNIVTAC_ROOT / "outputs_openpi")

    import openpi.models.pi0_config as pi0_config

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
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.99 if args.ema else None,
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

    print(f"\n{'='*60}")
    print(f" UniVTAC Pi0.5 OpenPI (JAX) Training")
    print(f"{'='*60}")
    print(f"  Task:          {task_name}")
    print(f"  Prompt:        {prompt}")
    print(f"  Model type:    {args.model_type}")
    print(f"  GPU:           {args.gpu}")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  Steps:         {args.steps}")
    print(f"  LR:            {args.lr}")
    print(f"  Action horizon:{args.action_horizon}")
    print(f"  EMA decay:     {0.99 if args.ema else 'disabled'}")
    print(f"  Checkpoint:    {CKPT_PATH}")
    print(f"  Output:        {output_base}")
    print(f"  WandB:         {args.wandb}")
    print(f"{'='*60}\n")

    import importlib.util
    train_script = OPENPI_ROOT / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("openpi_train", train_script)
    train_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_module)
    train_module.main(train_config)

    print(f"\nTask '{task_name}' training complete!")


def main():
    parser = argparse.ArgumentParser(
        description="UniVTAC Pi0.5 OpenPI (JAX) Full Fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--task", nargs="+", required=True,
                        help="Task name(s) to train on. Use 'all' for all tasks.")
    parser.add_argument("--gpu", type=str, default="0", help="GPU device ID(s)")
    parser.add_argument("--model_type", default="vision_only",
                        choices=["vision_only", "tactile"],
                        help="Input modality mode")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=2.5e-5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--save_freq", type=int, default=1000)
    parser.add_argument("--log_freq", type=int, default=50)
    parser.add_argument("--keep_period", type=int, default=5000)
    parser.add_argument("--save_best_only", action="store_true", default=True,
                        help="Only save best-loss checkpoint and final checkpoint")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action_horizon", type=int, default=50)
    parser.add_argument("--ema", action="store_true", default=True,
                        help="Enable EMA (default: True)")
    parser.add_argument("--no_ema", dest="ema", action="store_false")

    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: UniVTAC/outputs_openpi)")

    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing checkpoint directory")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint")

    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--wandb_project", default="univtac-pi05-openpi")

    parser.add_argument("--compute_norm_stats_only", action="store_true",
                        help="Only compute norm stats, no training")

    args = parser.parse_args()

    if args.task == ["all"]:
        tasks = list(TASK_INSTRUCTIONS.keys())
    else:
        tasks = args.task

    for task in tasks:
        if task not in TASK_INSTRUCTIONS:
            print(f"WARNING: Unknown task '{task}', skipping. Available: {list(TASK_INSTRUCTIONS.keys())}")
            continue

        if args.compute_norm_stats_only:
            compute_norm_stats(task, args.model_type)
        else:
            train_task(args, task)


if __name__ == "__main__":
    main()
