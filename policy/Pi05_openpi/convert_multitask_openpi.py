#!/usr/bin/env python3
"""
Convert UniVTAC HDF5 demo data directly to openpi-compatible LeRobot format.

This script combines the functionality of convert_to_lerobot.py and convert_to_openpi.py
into a single pipeline: HDF5 → LeRobot v2.1 (openpi env, lerobot==0.1.0).

Data source: /data1/zjb/data/UniVTAC/<task>/clean/*.hdf5
Output: /data1/zjb/data_lerobot_openpi/<task>/

HDF5 layout (per episode file):
  embodiment/joint          (T, 9)  float32  - 7 arm joints + 2 finger joints
  observation/head/rgb      (T,)    |S*      - JPEG-encoded images
  observation/wrist/rgb     (T,)    |S*      - JPEG-encoded wrist images
  tactile/left_gsmini/rgb_marker  (T,) |S*   - JPEG-encoded left tactile
  tactile/right_gsmini/rgb_marker (T,) |S*   - JPEG-encoded right tactile

State/Action convention:
  state  = joint[t,   :8]  (7 joints + 1 gripper mean)
  action = joint[t+1, :8]  (next-step target)

Usage:
    conda activate openpi

    # Single task
    python policy/Pi05_openpi/convert_multitask_openpi.py --task insert_HDMI

    # All 8 tasks
    python policy/Pi05_openpi/convert_multitask_openpi.py --task all

    # With tactile images
    python policy/Pi05_openpi/convert_multitask_openpi.py --task all --model tactile

    # Custom output, overwrite existing
    python policy/Pi05_openpi/convert_multitask_openpi.py --task all --output_dir /tmp/test --overwrite

    # Subsample to ~15fps (original 60fps / 4)
    python policy/Pi05_openpi/convert_multitask_openpi.py --task all --subsample 4
"""

import argparse
import io
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# ─── Paths ────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
UNIVTAC_ROOT = SCRIPT_DIR.parent.parent
DATA_ROOT = Path("/data1/zjb/ckpt/UniVTAC")
DEFAULT_OUTPUT_ROOT = Path("/data1/zjb/data_lerobot_openpi")
TASK_SETTINGS_PATH = UNIVTAC_ROOT / "policy" / "task_settings.json"

# ─── Task definitions ─────────────────────────────────────────────────────────

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

ALL_TASKS = list(TASK_INSTRUCTIONS.keys())

DUAL_CAMERA_TASKS = {"lift_can", "insert_tube"}

# ─── Constants ────────────────────────────────────────────────────────────────

TARGET_H, TARGET_W = 224, 224
JOINT_DIM = 8
BASE_FPS = 60
DEFAULT_SUBSAMPLE = 1


# ─── Image helpers ────────────────────────────────────────────────────────────

def decode_and_resize_jpeg(raw_bytes: bytes, target_hw=(TARGET_H, TARGET_W)) -> np.ndarray:
    """Decode JPEG bytes → uint8 (H, W, 3) RGB, resize to target."""
    try:
        import cv2
        buf = np.frombuffer(raw_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if img_rgb.shape[:2] != target_hw:
            img_rgb = cv2.resize(img_rgb, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_AREA)
        return img_rgb
    except ImportError:
        from PIL import Image
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        if (img.height, img.width) != target_hw:
            img = img.resize((target_hw[1], target_hw[0]), Image.LANCZOS)
        return np.array(img, dtype=np.uint8)


# ─── Data loading helpers ─────────────────────────────────────────────────────

def list_hdf5_files(task_name: str) -> list[Path]:
    """Return sorted list of HDF5 episode files for a task."""
    hdf5_dir = DATA_ROOT / task_name / "clean"
    if not hdf5_dir.exists():
        return []
    files = sorted(hdf5_dir.glob("*.hdf5"), key=lambda p: int(p.stem))
    return files


def load_task_settings() -> dict:
    if TASK_SETTINGS_PATH.exists():
        with open(TASK_SETTINGS_PATH) as f:
            return json.load(f)
    return {}


# ─── Feature schema ───────────────────────────────────────────────────────────

def build_features(has_wrist: bool, model_type: str = "vision_only") -> dict:
    """Build LeRobot feature schema for openpi consumption."""
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (JOINT_DIM,),
            "names": [
                "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
                "panda_joint5", "panda_joint6", "panda_joint7", "gripper",
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (JOINT_DIM,),
            "names": [
                "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
                "panda_joint5", "panda_joint6", "panda_joint7", "gripper",
            ],
        },
        "observation.images.head": {
            "dtype": "video",
            "shape": (TARGET_H, TARGET_W, 3),
            "names": ["height", "width", "channel"],
        },
    }
    if has_wrist:
        features["observation.images.wrist"] = {
            "dtype": "video",
            "shape": (TARGET_H, TARGET_W, 3),
            "names": ["height", "width", "channel"],
        }
    if model_type == "tactile":
        for side in ("tactile_left", "tactile_right"):
            features[f"observation.images.{side}"] = {
                "dtype": "video",
                "shape": (TARGET_H, TARGET_W, 3),
                "names": ["height", "width", "channel"],
            }
    return features


# ─── Core conversion ─────────────────────────────────────────────────────────

def convert_task(
    task_name: str,
    output_root: Path,
    subsample: int = DEFAULT_SUBSAMPLE,
    model_type: str = "vision_only",
    overwrite: bool = False,
    max_episodes: int | None = None,
) -> bool:
    """Convert a single task from HDF5 to openpi-compatible LeRobot format."""

    files = list_hdf5_files(task_name)
    if not files:
        print(f"[{task_name}] No HDF5 files found in {DATA_ROOT / task_name / 'clean'}, skipping.")
        return False

    if max_episodes is not None:
        files = files[:max_episodes]

    out_dir = output_root / task_name
    if out_dir.exists():
        if overwrite:
            print(f"[{task_name}] Removing existing {out_dir}")
            shutil.rmtree(out_dir)
        else:
            print(f"[{task_name}] Already exists at {out_dir} (use --overwrite to recreate)")
            return True

    has_wrist = task_name in DUAL_CAMERA_TASKS
    use_tactile = model_type == "tactile"
    instruction = TASK_INSTRUCTIONS.get(task_name, task_name.replace("_", " "))
    features = build_features(has_wrist, model_type=model_type)
    fps = BASE_FPS // subsample if subsample > 1 else BASE_FPS

    if use_tactile:
        with h5py.File(files[0], "r") as f:
            if "tactile/left_gsmini/rgb_marker" not in f or "tactile/right_gsmini/rgb_marker" not in f:
                print(f"[{task_name}] WARNING: tactile data not available, skipping tactile channels.")
                use_tactile = False
                features = build_features(has_wrist, model_type="vision_only")

    print(f"\n{'='*60}")
    print(f" Converting: {task_name}  (HDF5 → openpi LeRobot)")
    print(f"{'='*60}")
    print(f"  Episodes:      {len(files)}")
    print(f"  FPS:           {BASE_FPS} → {fps} (subsample={subsample})")
    print(f"  State/Action:  {JOINT_DIM}-dim (7 joints + 1 gripper)")
    print(f"  Has wrist cam: {has_wrist}")
    print(f"  Use tactile:   {use_tactile}")
    print(f"  Instruction:   \"{instruction}\"")
    print(f"  Output:        {out_dir}")
    print(f"  Features:      {list(features.keys())}")

    repo_id = f"univtac/{task_name}"
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=out_dir,
        robot_type="franka_panda",
        use_videos=True,
        image_writer_threads=4,
    )

    total_frames = 0
    episode_bar = tqdm(files, desc=f"[{task_name}] episodes", unit="ep")

    for ep_path in episode_bar:
        with h5py.File(ep_path, "r") as f:
            joint_all = f["embodiment/joint"][:]
            head_raw = f["observation/head/rgb"][()]
            wrist_raw = f["observation/wrist/rgb"][()] if (has_wrist and "observation/wrist/rgb" in f) else None
            tactile_left_raw = f["tactile/left_gsmini/rgb_marker"][()] if use_tactile else None
            tactile_right_raw = f["tactile/right_gsmini/rgb_marker"][()] if use_tactile else None

        T = joint_all.shape[0]
        S = subsample
        frame_indices = list(range(0, T - S, S))
        n_frames = len(frame_indices)
        if n_frames <= 0:
            continue

        needed_times = set(frame_indices) | {t + S for t in frame_indices if t + S < T}
        head_imgs = {t: decode_and_resize_jpeg(bytes(head_raw[t])) for t in needed_times if t < T}
        wrist_imgs = (
            {t: decode_and_resize_jpeg(bytes(wrist_raw[t])) for t in needed_times if t < T}
            if wrist_raw is not None else None
        )
        tactile_left_imgs = (
            {t: decode_and_resize_jpeg(bytes(tactile_left_raw[t])) for t in needed_times if t < T}
            if tactile_left_raw is not None else None
        )
        tactile_right_imgs = (
            {t: decode_and_resize_jpeg(bytes(tactile_right_raw[t])) for t in needed_times if t < T}
            if tactile_right_raw is not None else None
        )

        for t in frame_indices:
            t_next = t + S
            state = joint_all[t, :JOINT_DIM].astype(np.float32)
            action = joint_all[t_next, :JOINT_DIM].astype(np.float32)

            state[7] = float(joint_all[t, 7:9].mean())
            action[7] = float(joint_all[t_next, 7:9].mean())

            frame = {
                "observation.state": state,
                "action": action,
                "observation.images.head": head_imgs[t],
                "task": instruction,
            }
            if has_wrist and wrist_imgs is not None:
                frame["observation.images.wrist"] = wrist_imgs[t]
            if use_tactile:
                frame["observation.images.tactile_left"] = tactile_left_imgs[t]
                frame["observation.images.tactile_right"] = tactile_right_imgs[t]

            dataset.add_frame(frame)

        dataset.save_episode()
        total_frames += n_frames
        episode_bar.set_postfix({"frames": total_frames})

    if dataset.image_writer is not None:
        dataset.stop_image_writer()

    print(f"\n[{task_name}] Done! {len(files)} episodes, {total_frames} frames → {out_dir}")
    return True


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert UniVTAC HDF5 data to openpi-compatible LeRobot format (single script).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--task", nargs="+", required=True,
                        help=f"Task name(s) or 'all'. Available: {ALL_TASKS}")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_ROOT),
                        help="Output root directory (default: /data1/zjb/data_lerobot_openpi)")
    parser.add_argument("--model", type=str, default="vision_only",
                        choices=["vision_only", "tactile"],
                        help="Input modality: vision_only (head+wrist) or tactile (+gelsight)")
    parser.add_argument("--subsample", type=int, default=DEFAULT_SUBSAMPLE,
                        help=f"Temporal subsampling factor (default: {DEFAULT_SUBSAMPLE}). "
                             "Set 4 for ~15fps, 6 for ~10fps.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing converted datasets")
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="Limit episodes per task (for quick testing)")

    args = parser.parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.task == ["all"]:
        tasks = ALL_TASKS
    else:
        tasks = args.task

    print(f"\n{'#'*60}")
    print(f" UniVTAC Multi-Task Data Conversion")
    print(f" Tasks: {tasks}")
    print(f" Model: {args.model}")
    print(f" Subsample: {args.subsample}x (FPS: {BASE_FPS} → {BASE_FPS // args.subsample})")
    print(f" Output: {output_root}")
    print(f"{'#'*60}\n")

    success_count = 0
    for task in tasks:
        if task not in TASK_INSTRUCTIONS:
            print(f"WARNING: Unknown task '{task}'. Available: {ALL_TASKS}")
            continue
        ok = convert_task(
            task_name=task,
            output_root=output_root,
            subsample=args.subsample,
            model_type=args.model,
            overwrite=args.overwrite,
            max_episodes=args.max_episodes,
        )
        if ok:
            success_count += 1

    print(f"\n{'='*60}")
    print(f" Conversion complete: {success_count}/{len(tasks)} tasks successful")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
