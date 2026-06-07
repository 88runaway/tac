#!/usr/bin/env python3
"""
UniVTAC Pi0.5 LoRA Fine-tuning Script.

This script fine-tunes the pi05_base model on UniVTAC manipulation tasks
using LoRA adapters for parameter-efficient training.

Usage:
    # Default LoRA training on lift_can
    python policy/Pi05/train_pi05.py --task lift_can --gpu 0

    # Full fine-tuning (no LoRA)
    python policy/Pi05/train_pi05.py --task lift_can --gpu 0 --no_lora

    # Custom hyperparameters
    python policy/Pi05/train_pi05.py --task lift_can --gpu 0 \
        --batch_size 8 --steps 10000 --lora_rank 128

    # Multiple tasks (trains sequentially)
    python policy/Pi05/train_pi05.py --task lift_can insert_tube --gpu 0

Prerequisites:
    1. Convert UniVTAC data to LeRobot format:
       python scripts/convert_to_lerobot.py --task <task> --output_dir /data1/zjb/lerobot_datasets

    2. Install lerobot with pi and peft support:
       cd /data1/zjb/lerobot && pip install -e ".[pi,peft]"

    3. (Optional) Compute quantile stats for better normalization:
       python /data1/zjb/lerobot/src/lerobot/scripts/augment_dataset_quantile_stats.py \
           --repo-id univtac/<task> --root /data1/zjb/lerobot_datasets/<task>
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
UNIVTAC_ROOT = SCRIPT_DIR.parent.parent
LEROBOT_ROOT = Path("/data1/zjb/lerobot")
DATASET_ROOT = Path("/data1/zjb/lerobot_datasets")
TASK_SETTINGS_PATH = UNIVTAC_ROOT / "policy" / "task_settings.json"

DUAL_CAMERA_TASKS = {"lift_can", "insert_tube"}

TASK_INSTRUCTIONS = {
    "lift_can": "Pick up the can and place it in the basket.",
    "lift_bottle": "Pick up the bottle and place it upright.",
    "insert_tube": "Insert the tube into the connector.",
    "insert_hole": "Insert the peg into the hole.",
    "insert_HDMI": "Insert the HDMI cable into the port.",
    "pull_out_key": "Pull the key out of the lock.",
    "put_bottle_in_shelf": "Place the bottle on the shelf.",
    "grasp_classify": "Grasp the object and classify its texture.",
}


def get_camera_type(task_name: str) -> str:
    if TASK_SETTINGS_PATH.exists():
        with open(TASK_SETTINGS_PATH) as f:
            settings = json.load(f)
        return settings.get(task_name, {}).get("camera_type", "head")
    return "all" if task_name in DUAL_CAMERA_TASKS else "head"


def build_train_command(args, task_name: str) -> list[str]:
    """Build the lerobot-train CLI command for a single task."""

    dataset_dir = DATASET_ROOT / task_name
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_dir}\n"
            f"Run: python scripts/convert_to_lerobot.py --task {task_name} "
            f"--output_dir {DATASET_ROOT}"
        )

    camera_type = get_camera_type(task_name)
    has_wrist = camera_type == "all"

    rename_map = {"observation.images.head": "observation.images.base_0_rgb"}
    if has_wrist:
        rename_map["observation.images.wrist"] = "observation.images.left_wrist_0_rgb"

    # pi05_base expects 3 cameras; fill unused slots
    empty_cameras = 1 if has_wrist else 2

    output_dir = (
        UNIVTAC_ROOT / "outputs" / f"pi05_{task_name}_{'lora' if not args.no_lora else 'full'}"
    )

    cmd = [
        sys.executable, "-m", "lerobot.scripts.lerobot_train",
        f"--dataset.repo_id=univtac/{task_name}",
        f"--dataset.root={dataset_dir}",
        "--dataset.video_backend=pyav",
        "--policy.type=pi05",
        f"--policy.pretrained_path={args.pretrained_path}",
        f"--policy.dtype={args.dtype}",
        "--policy.gradient_checkpointing=true",
        f"--policy.compile_model={'true' if args.compile else 'false'}",
        f"--policy.freeze_vision_encoder={'true' if args.freeze_vision else 'false'}",
        f"--policy.train_expert_only={'true' if args.train_expert_only else 'false'}",
        f"--policy.empty_cameras={empty_cameras}",
        f"--policy.chunk_size={args.chunk_size}",
        f"--policy.n_action_steps={args.n_action_steps}",
    ]

    # Normalization
    if args.use_quantiles:
        norm_map = '{"VISUAL": "IDENTITY", "STATE": "QUANTILES", "ACTION": "QUANTILES"}'
    else:
        norm_map = '{"VISUAL": "IDENTITY", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"}'
    cmd.append(f"--policy.normalization_mapping={norm_map}")

    # LoRA config
    if not args.no_lora:
        cmd.extend([
            "--peft.method_type=LORA",
            f"--peft.r={args.lora_rank}",
            f"--peft.lora_alpha={args.lora_alpha}",
            f"--peft.target_modules={args.lora_targets}",
        ])

    # Training params
    cmd.extend([
        f"--output_dir={output_dir}",
        f"--job_name=pi05_{task_name}_{'lora' if not args.no_lora else 'full'}",
        f"--seed={args.seed}",
        f"--batch_size={args.batch_size}",
        f"--num_workers={args.num_workers}",
        f"--steps={args.steps}",
        f"--save_freq={args.save_freq}",
        "--save_checkpoint=true",
        f"--eval_freq={args.eval_freq}",
        f"--log_freq={args.log_freq}",
        f"--rename_map={json.dumps(rename_map)}",
    ])

    # WandB
    if args.wandb:
        cmd.extend([
            "--wandb.enable=true",
            f"--wandb.project={args.wandb_project}",
        ])
    else:
        cmd.append("--wandb.enable=false")

    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="UniVTAC Pi0.5 LoRA Fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--task", nargs="+", required=True,
                        help="Task name(s) to train on. Use 'all' for all tasks.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")

    # Model
    parser.add_argument("--pretrained_path", default="lerobot/pi05_base",
                        help="Pretrained model path/HuggingFace repo")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    parser.add_argument("--freeze_vision", action="store_true", default=True,
                        help="Freeze vision encoder (default: True for LoRA)")
    parser.add_argument("--no_freeze_vision", dest="freeze_vision", action="store_false")
    parser.add_argument("--train_expert_only", action="store_true",
                        help="Only train action expert (most memory efficient)")

    # LoRA
    parser.add_argument("--no_lora", action="store_true", help="Disable LoRA (full fine-tuning)")
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_targets", type=str,
                        default=r"(.*\.gemma_expert\..*\.self_attn\.(q|v)_proj|model\.(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out))",
                        help="Regex for LoRA target modules")

    # Training
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--save_freq", type=int, default=1000)
    parser.add_argument("--eval_freq", type=int, default=50000)
    parser.add_argument("--log_freq", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk_size", type=int, default=50)
    parser.add_argument("--n_action_steps", type=int, default=50)

    # Normalization
    parser.add_argument("--use_quantiles", action="store_true",
                        help="Use QUANTILES normalization (requires augmented stats)")

    # Logging
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--wandb_project", default="univtac-pi05")

    # Misc
    parser.add_argument("--dry_run", action="store_true", help="Print command without executing")

    args = parser.parse_args()

    # Resolve task list
    if args.task == ["all"]:
        tasks = list(TASK_INSTRUCTIONS.keys())
    else:
        tasks = args.task

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    for task in tasks:
        if task not in TASK_INSTRUCTIONS:
            print(f"WARNING: Unknown task '{task}', skipping. Available: {list(TASK_INSTRUCTIONS.keys())}")
            continue

        print(f"\n{'='*60}")
        print(f" Training Pi0.5 on: {task}")
        print(f" Mode: {'LoRA (r={})'.format(args.lora_rank) if not args.no_lora else 'Full fine-tuning'}")
        print(f" GPU: {args.gpu}")
        print(f"{'='*60}\n")

        try:
            cmd = build_train_command(args, task)
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            continue

        if args.dry_run:
            print("Command:")
            print("  " + " \\\n    ".join(cmd))
            continue

        env = os.environ.copy()
        env["PYTHONPATH"] = str(LEROBOT_ROOT / "src") + ":" + env.get("PYTHONPATH", "")

        result = subprocess.run(cmd, cwd=str(LEROBOT_ROOT), env=env)
        if result.returncode != 0:
            print(f"ERROR: Training failed for task '{task}' with exit code {result.returncode}")
            sys.exit(result.returncode)

        print(f"\nTask '{task}' training complete!")


if __name__ == "__main__":
    main()
