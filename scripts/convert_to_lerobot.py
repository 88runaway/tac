#!/usr/bin/env python3
"""
Convert UniVTAC HDF5 demo data to LeRobotDataset format for π₀.₅ training.

UniVTAC HDF5 layout (per episode file):
  embodiment/joint          (T, 9)  float32  - 7 arm joints + 2 finger joints (fingers are equal)
  embodiment/ee             (T, 7)  float32  - end-effector pose (pos_xyz + quat_wxyz)
  observation/head/rgb      (T,)    |S*      - JPEG-encoded 480×270 images
  observation/wrist/rgb     (T,)    |S*      - JPEG-encoded 480×270 images (only some tasks)
  tactile/left_gsmini/rgb_marker  (T,) |S*   - JPEG-encoded 320×240 left finger tactile images
  tactile/right_gsmini/rgb_marker (T,) |S*   - JPEG-encoded 320×240 right finger tactile images
  step                      (T,)    int64    - step index

State / Action convention (mirrors UniVTAC process_data.py):
  state  = joint[t,   :8]   (current qpos: 7 joints + 1 gripper)
  action = joint[t+1, :8]   (next  qpos: next-step target)
  → T-1 valid (state, action) pairs per episode

Image resize: 480×270 → 224×224 (required by π₀.₅ PaliGemma vision encoder)

LeRobot output structure:
  {output_dir}/{task_name}/
    data/chunk-000/*.parquet   - observation.state, action, task_index, episode_index, ...
    videos/observation.images.head/chunk-000/*.mp4
    videos/observation.images.wrist/chunk-000/*.mp4  (only for lift_can, insert_tube)
    meta/info.json, stats.json, tasks.jsonl, episodes.jsonl

Usage:
  # Single task (use default data path: <univtac_root>/data/<task>/demo/hdf5/)
  python scripts/convert_to_lerobot.py --task lift_can --output_dir /data1/zjb/lerobot_datasets

  # All tasks
  python scripts/convert_to_lerobot.py --task all --output_dir /data1/zjb/lerobot_datasets

  # Include tactile images (left+right GelSight Mini rgb_marker)
  python scripts/convert_to_lerobot.py --task insert_tube --model tactile \
      --data_dir /data1/zjb/ckpt/UniVTAC/insert_tube/clean \
      --output_dir /data1/zjb/UniVTAC/data_lerobot

  # Custom source data directory (e.g. official dataset or another collection)
  python scripts/convert_to_lerobot.py --task lift_bottle \
      --data_dir /data1/zjb/ckpt/UniVTAC/lift_bottle/clean \
      --output_dir /data1/zjb/UniVTAC/data_lerobot_official

  # Dry-run (print stats without writing)
  python scripts/convert_to_lerobot.py --task lift_can --dry_run

  # Limit episodes (for quick testing)
  python scripts/convert_to_lerobot.py --task lift_can --max_episodes 10 --output_dir /tmp/test_lerobot
"""

import argparse
import io
import json
import os
import struct
import sys
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

# Add lerobot to path
LEROBOT_ROOT = Path(__file__).parent.parent.parent / "lerobot" / "src"
if LEROBOT_ROOT.exists():
    sys.path.insert(0, str(LEROBOT_ROOT))

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ─── Constants ────────────────────────────────────────────────────────────────

UNIVTAC_DATA_ROOT = Path(__file__).parent.parent / "data"
TASK_SETTINGS_PATH = Path(__file__).parent.parent / "policy" / "task_settings.json"

# Camera config: tasks that have wrist camera in addition to head camera
DUAL_CAMERA_TASKS = {"lift_can", "insert_tube"}

# π₀.₅ requires 224×224 images; source is 480×270
SOURCE_W, SOURCE_H = 480, 270
TARGET_W, TARGET_H = 224, 224

# Joint dimensions: take first 8 dims (7 arm DOFs + 1 gripper = mean of 2 fingers)
JOINT_DIM = 8

# Simulation: dt=1/120 s, decimation=1, save_frequency=2 → effective 60 fps
# univtac.yml: save_frequency=2, dt=1/120 → 120/2 = 60 fps
# For pi0.5 training, recommend downsampling to 10-15 fps (subsample_factor=4~6)
FPS = 60

# Temporal subsampling factor: take every N-th frame
# Effective FPS = FPS / SUBSAMPLE_FACTOR
# Default 1 = no subsampling; set to 4 for ~15 fps or 6 for ~10 fps
SUBSAMPLE_FACTOR = 1

# Task instruction templates
TASK_INSTRUCTIONS = {
    "lift_can":           "Pick up the can and place it in the basket.",
    "lift_bottle":        "Pick up the bottle and place it upright.",
    "insert_tube":        "Insert the tube into the connector.",
    "insert_hole":        "Insert the peg into the hole.",
    "insert_HDMI":        "Insert the HDMI cable into the port.",
    "pull_out_key":       "Pull the key out of the lock.",
    "put_bottle_in_shelf": "Place the bottle on the shelf.",
    "grasp_classify":     "Grasp the object and classify its texture.",
}

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _jpeg_wh(data: bytes):
    """Parse width/height from JPEG SOF0/SOF1/SOF2 marker without PIL."""
    i = 0
    while i < len(data) - 9:
        if data[i] == 0xFF and data[i + 1] in (0xC0, 0xC1, 0xC2):
            h = struct.unpack(">H", data[i + 5: i + 7])[0]
            w = struct.unpack(">H", data[i + 7: i + 9])[0]
            return w, h
        i += 1
    return None, None


def decode_and_resize_jpeg(raw_bytes: bytes, target_wh=(TARGET_W, TARGET_H)) -> np.ndarray:
    """
    Decode JPEG bytes → numpy uint8 (H, W, 3) RGB, then resize to target_wh.

    Tries cv2 first (faster), falls back to PIL/Pillow.
    """
    try:
        import cv2
        buf = np.frombuffer(raw_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if img_rgb.shape[:2] != (target_wh[1], target_wh[0]):
            img_rgb = cv2.resize(img_rgb, target_wh, interpolation=cv2.INTER_AREA)
        return img_rgb
    except ImportError:
        from PIL import Image
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        if img.size != target_wh:
            img = img.resize(target_wh, Image.LANCZOS)
        return np.array(img, dtype=np.uint8)


def load_task_settings() -> dict:
    if TASK_SETTINGS_PATH.exists():
        with open(TASK_SETTINGS_PATH) as f:
            return json.load(f)
    return {}


def list_hdf5_files(task_name: str, data_dir: Path | None = None) -> list[Path]:
    """Return sorted list of HDF5 episode files for a task."""
    if data_dir is not None:
        # --data_dir points directly to the directory containing .hdf5 files,
        # OR to a task root (we try data_dir/hdf5 and data_dir itself).
        hdf5_dir = data_dir / "hdf5"
        if not hdf5_dir.exists():
            hdf5_dir = data_dir
    else:
        hdf5_dir = UNIVTAC_DATA_ROOT / task_name / "demo" / "hdf5"
        if not hdf5_dir.exists():
            # fallback: files directly in demo/
            hdf5_dir = UNIVTAC_DATA_ROOT / task_name / "demo"
    files = sorted(hdf5_dir.glob("*.hdf5"), key=lambda p: int(p.stem))
    return files


def inspect_task(task_name: str, data_dir: Path | None = None) -> dict:
    """
    Inspect one HDF5 file to summarise data layout.
    Returns a dict with keys: has_wrist, num_frames_example, joint_shape, img_wh
    """
    files = list_hdf5_files(task_name, data_dir=data_dir)
    if not files:
        return {}
    with h5py.File(files[0], "r") as f:
        joint = f["embodiment/joint"][:]
        head_bytes = bytes(f["observation/head/rgb"][0])
        has_wrist = "wrist" in f["observation"]
    w, h = _jpeg_wh(head_bytes)
    return {
        "num_files": len(files),
        "num_frames_example": joint.shape[0],
        "joint_shape": joint.shape,
        "img_wh": (w, h),
        "has_wrist": has_wrist,
    }


# ─── Feature schema ───────────────────────────────────────────────────────────

def build_features(has_wrist: bool, model_type: str = "vision_only") -> dict:
    """
    Build LeRobotDataset feature schema.

    observation.state  : 8-dim joint qpos (float32)
    action             : 8-dim joint qpos target (float32)
    observation.images.head  : 224×224 RGB video
    observation.images.wrist : 224×224 RGB video (only if has_wrist)
    observation.images.tactile_left  : 224×224 RGB video (only if model_type == "tactile")
    observation.images.tactile_right : 224×224 RGB video (only if model_type == "tactile")
    """
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
    output_dir: Path,
    max_episodes: int | None = None,
    dry_run: bool = False,
    verbose: bool = True,
    overwrite: bool = False,
    data_dir: Path | None = None,
    model_type: str = "vision_only",
) -> None:
    """Convert one UniVTAC task to LeRobotDataset format."""

    files = list_hdf5_files(task_name, data_dir=data_dir)
    if not files:
        src = str(data_dir) if data_dir else str(UNIVTAC_DATA_ROOT / task_name)
        print(f"[{task_name}] No HDF5 files found in {src}, skipping.")
        return

    if max_episodes is not None:
        files = files[:max_episodes]

    has_wrist = task_name in DUAL_CAMERA_TASKS
    use_tactile = model_type == "tactile"
    instruction = TASK_INSTRUCTIONS.get(task_name, task_name.replace("_", " "))
    features = build_features(has_wrist, model_type=model_type)

    # Validate tactile data availability
    if use_tactile:
        with h5py.File(files[0], "r") as f:
            if "tactile/left_gsmini/rgb_marker" not in f or "tactile/right_gsmini/rgb_marker" not in f:
                raise ValueError(
                    f"[{task_name}] model=tactile requested but HDF5 files lack "
                    f"tactile/{{left,right}}_gsmini/rgb_marker data."
                )

    if verbose:
        info = inspect_task(task_name, data_dir=data_dir)
        print(f"\n{'='*60}")
        print(f"Task: {task_name}")
        print(f"  Model type:   {model_type}")
        print(f"  Episodes: {len(files)} (of {info.get('num_files', '?')} total)")
        print(f"  Source image: {info.get('img_wh')} → resize to ({TARGET_W}, {TARGET_H})")
        print(f"  Joint shape:  {info.get('joint_shape')} → use [:8]")
        print(f"  Has wrist cam: {has_wrist}")
        print(f"  Use tactile:   {use_tactile}")
        print(f"  Instruction: \"{instruction}\"")
        print(f"  Subsample:    {SUBSAMPLE_FACTOR}x (60fps → {FPS}fps)")
        print(f"  Output FPS: {FPS}")
        print(f"  Features: {list(features.keys())}")

    if dry_run:
        print(f"[dry-run] Skipping write for {task_name}.")
        return

    repo_id = f"univtac/{task_name}"
    task_output_dir = output_dir / task_name

    if task_output_dir.exists():
        if overwrite:
            import shutil
            print(f"[{task_name}] Removing existing dataset at {task_output_dir}")
            shutil.rmtree(task_output_dir)
        else:
            print(f"[{task_name}] ERROR: Output directory already exists: {task_output_dir}")
            print(f"  Use --overwrite to delete and re-convert, or remove it manually.")
            return

    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        features=features,
        root=task_output_dir,
        robot_type="franka_panda",
        use_videos=True,
        image_writer_threads=4,
    )

    total_frames = 0
    episode_bar = tqdm(files, desc=f"[{task_name}] episodes", unit="ep")
    for ep_path in episode_bar:
        with h5py.File(ep_path, "r") as f:
            joint_all = f["embodiment/joint"][:]          # (T, 9)
            head_raw  = f["observation/head/rgb"][()]     # (T,) bytes
            wrist_raw = f["observation/wrist/rgb"][()] if has_wrist else None
            tactile_left_raw = f["tactile/left_gsmini/rgb_marker"][()] if use_tactile else None
            tactile_right_raw = f["tactile/right_gsmini/rgb_marker"][()] if use_tactile else None

        T = joint_all.shape[0]
        S = SUBSAMPLE_FACTOR
        frame_indices = list(range(0, T - S, S))
        n_frames = len(frame_indices)
        if n_frames <= 0:
            continue

        needed_times = set(frame_indices) | {t + S for t in frame_indices}
        head_imgs = {t: decode_and_resize_jpeg(bytes(head_raw[t])) for t in needed_times if t < T}
        wrist_imgs = ({t: decode_and_resize_jpeg(bytes(wrist_raw[t])) for t in needed_times if t < T}
                      if has_wrist else None)
        tactile_left_imgs = ({t: decode_and_resize_jpeg(bytes(tactile_left_raw[t])) for t in needed_times if t < T}
                             if use_tactile else None)
        tactile_right_imgs = ({t: decode_and_resize_jpeg(bytes(tactile_right_raw[t])) for t in needed_times if t < T}
                              if use_tactile else None)

        for t in frame_indices:
            t_next = t + S
            state  = joint_all[t,      :JOINT_DIM].astype(np.float32)
            action = joint_all[t_next, :JOINT_DIM].astype(np.float32)

            state[7]  = float(joint_all[t,      7:9].mean())
            action[7] = float(joint_all[t_next, 7:9].mean())

            frame = {
                "observation.state": state,
                "action": action,
                "observation.images.head": head_imgs[t],
                "task": instruction,
            }
            if has_wrist:
                frame["observation.images.wrist"] = wrist_imgs[t]
            if use_tactile:
                frame["observation.images.tactile_left"] = tactile_left_imgs[t]
                frame["observation.images.tactile_right"] = tactile_right_imgs[t]

            dataset.add_frame(frame)

        dataset.save_episode()
        total_frames += n_frames
        episode_bar.set_postfix({"frames": total_frames})

    dataset.finalize()

    if verbose:
        print(f"[{task_name}] Done: {len(files)} episodes, {total_frames} frames")
        print(f"  Saved to: {task_output_dir}")


# ─── Entry point ─────────────────────────────────────────────────────────────

ALL_TASKS = list(TASK_INSTRUCTIONS.keys())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert UniVTAC HDF5 demo data to LeRobotDataset format."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="lift_can",
        help=f"Task name or 'all'. Available: {ALL_TASKS}",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data1/zjb/lerobot_datasets",
        help="Root output directory (a subfolder per task is created).",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Limit number of episodes per task (for quick testing).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print data stats without writing any files.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=FPS,
        help=f"Frames per second for the output dataset (default: {FPS}).",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        nargs=2,
        default=[TARGET_W, TARGET_H],
        metavar=("W", "H"),
        help=f"Target image size WxH (default: {TARGET_W} {TARGET_H}).",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=SUBSAMPLE_FACTOR,
        help=f"Temporal subsampling factor (default: {SUBSAMPLE_FACTOR}). "
             "Set to 4 for ~15fps or 6 for ~10fps (recommended for pi0.5).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and re-convert if output directory already exists.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help=(
            "Path to the source HDF5 data directory. "
            "Can point to a directory containing .hdf5 files directly, "
            "or a task root with a hdf5/ subdirectory. "
            "If not set, defaults to <univtac_root>/data/<task>/demo/hdf5/."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="vision_only",
        choices=["vision_only", "tactile"],
        help=(
            "Input modality mode. "
            "'vision_only': head + wrist cameras only (default). "
            "'tactile': additionally include left/right GelSight Mini rgb_marker images."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Allow overriding global constants from CLI
    global TARGET_W, TARGET_H, FPS, SUBSAMPLE_FACTOR
    TARGET_W, TARGET_H = args.target_size
    SUBSAMPLE_FACTOR = args.subsample
    FPS = args.fps // SUBSAMPLE_FACTOR if SUBSAMPLE_FACTOR > 1 else args.fps

    output_dir = Path(args.output_dir)

    tasks_to_convert = ALL_TASKS if args.task == "all" else [args.task]
    data_dir = Path(args.data_dir) if args.data_dir else None

    for task in tasks_to_convert:
        if task not in TASK_INSTRUCTIONS:
            print(f"Unknown task '{task}'. Available: {ALL_TASKS}")
            continue
        convert_task(
            task_name=task,
            output_dir=output_dir,
            max_episodes=args.max_episodes,
            dry_run=args.dry_run,
            verbose=True,
            overwrite=args.overwrite,
            data_dir=data_dir,
            model_type=args.model,
        )


if __name__ == "__main__":
    main()
