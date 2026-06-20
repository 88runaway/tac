#!/usr/bin/env python3
"""
Convert UniVTAC HDF5 demo data to openpi-compatible LeRobot format **with tactile**.

This script extends the standard conversion with left/right ``rgb_marker`` tactile
video channels.  During DF training the data loader uses ``delta_timestamps`` to
load tactile frames at block-boundary times so the model can select the correct
frame based on the monotone progress scalar ``p``.

Data source: /data1/zjb/ckpt/UniVTAC/<task>/clean/*.hdf5

HDF5 layout (per episode):
  embodiment/joint                       (T, 9)   float32
  observation/head/rgb                   (T,)     |S*  JPEG
  observation/wrist/rgb                  (T,)     |S*  JPEG  (optional)
  tactile/left_gsmini/rgb_marker         (T,)     |S*  JPEG
  tactile/right_gsmini/rgb_marker        (T,)     |S*  JPEG

Output features:
  observation.state          (8,)       float32
  action                     (8,)       float32
  observation.images.head    (224,224,3) video
  observation.images.tactile_left   (224,224,3) video   ← NEW
  observation.images.tactile_right  (224,224,3) video   ← NEW

The tactile channels store **every subsampled frame** as video.  At training time
the data loader retrieves only the block-boundary frames via ``delta_timestamps``:

    delta_timestamps["observation.images.tactile_left"]  = [b * block_size / fps for b in range(num_blocks)]
    delta_timestamps["observation.images.tactile_right"] = [b * block_size / fps for b in range(num_blocks)]

Usage:
    conda activate openpi

    # Single task
    python policy/Pi05_openpi_DF/convert_df_tactile.py --task insert_HDMI

    # All tasks
    python policy/Pi05_openpi_DF/convert_df_tactile.py --task all

    # With subsampling
    python policy/Pi05_openpi_DF/convert_df_tactile.py --task all --subsample 4
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
DEFAULT_OUTPUT_ROOT = Path("/data1/zjb/data_lerobot_openpi_df_tactile")
TASK_SETTINGS_PATH = UNIVTAC_ROOT / "policy" / "task_settings.json"

# ─── Task definitions ─────────────────────────────────────────────────────────

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

ALL_TASKS = list(TASK_INSTRUCTIONS.keys())
DUAL_CAMERA_TASKS = {"lift_can", "insert_tube"}

TARGET_H, TARGET_W = 224, 224
JOINT_DIM = 8
BASE_FPS = 60


# ─── Helpers ──────────────────────────────────────────────────────────────────

def decode_and_resize_jpeg(raw_bytes: bytes, target_hw=(TARGET_H, TARGET_W)) -> np.ndarray:
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


def list_hdf5_files(task_name: str) -> list[Path]:
    hdf5_dir = DATA_ROOT / task_name / "clean"
    if not hdf5_dir.exists():
        return []
    return sorted(hdf5_dir.glob("*.hdf5"), key=lambda p: int(p.stem))


# ─── Feature schema ──────────────────────────────────────────────────────────

def build_features(has_wrist: bool) -> dict:
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
    # Store tactile as image (not video) to avoid per-step video decoding cost.
    # LeRobot writes each frame as a PNG file; delta_timestamps still works via
    # frame-index lookup (no video codec involved).
    for side in ("tactile_left", "tactile_right"):
        features[f"observation.images.{side}"] = {
            "dtype": "image",
            "shape": (TARGET_H, TARGET_W, 3),
            "names": ["height", "width", "channel"],
        }
    return features


# ─── Core conversion ─────────────────────────────────────────────────────────

def convert_task(
    task_name: str,
    output_root: Path,
    subsample: int = 1,
    overwrite: bool = False,
    max_episodes: int | None = None,
) -> bool:
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
    instruction = TASK_INSTRUCTIONS.get(task_name, task_name.replace("_", " "))
    features = build_features(has_wrist)
    fps = BASE_FPS // subsample if subsample > 1 else BASE_FPS

    # Verify tactile availability
    with h5py.File(files[0], "r") as f:
        has_left = "tactile/left_gsmini/rgb_marker" in f
        has_right = "tactile/right_gsmini/rgb_marker" in f
    if not (has_left and has_right):
        print(f"[{task_name}] WARNING: tactile data not available — skipping task.")
        return False

    print(f"\n{'='*60}")
    print(f" Converting: {task_name}  (HDF5 → openpi LeRobot + tactile)")
    print(f"{'='*60}")
    print(f"  Episodes:      {len(files)}")
    print(f"  FPS:           {BASE_FPS} → {fps} (subsample={subsample})")
    print(f"  Has wrist cam: {has_wrist}")
    print(f"  Instruction:   \"{instruction}\"")
    print(f"  Output:        {out_dir}")

    repo_id = f"univtac_df_tac/{task_name}"
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
    S = subsample
    episode_bar = tqdm(files, desc=f"[{task_name}] episodes", unit="ep")

    for ep_path in episode_bar:
        with h5py.File(ep_path, "r") as f:
            joint_all = f["embodiment/joint"][:]
            head_raw = f["observation/head/rgb"][()]
            wrist_raw = f["observation/wrist/rgb"][()] if (has_wrist and "observation/wrist/rgb" in f) else None
            tac_left_raw = f["tactile/left_gsmini/rgb_marker"][()]
            tac_right_raw = f["tactile/right_gsmini/rgb_marker"][()]

        T = joint_all.shape[0]
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
        tac_left_imgs = {t: decode_and_resize_jpeg(bytes(tac_left_raw[t])) for t in needed_times if t < T}
        tac_right_imgs = {t: decode_and_resize_jpeg(bytes(tac_right_raw[t])) for t in needed_times if t < T}

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
                "observation.images.tactile_left": tac_left_imgs[t],
                "observation.images.tactile_right": tac_right_imgs[t],
                "task": instruction,
            }
            if has_wrist and wrist_imgs is not None:
                frame["observation.images.wrist"] = wrist_imgs[t]

            dataset.add_frame(frame)

        dataset.save_episode()
        total_frames += n_frames
        episode_bar.set_postfix({"frames": total_frames})

    if dataset.image_writer is not None:
        dataset.stop_image_writer()

    print(f"\n[{task_name}] Done! {len(files)} episodes, {total_frames} frames → {out_dir}")
    return True


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert UniVTAC HDF5 data to openpi LeRobot format with tactile channels.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--task", nargs="+", required=True,
                        help=f"Task name(s) or 'all'. Available: {ALL_TASKS}")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--subsample", type=int, default=1,
                        help="Temporal subsampling factor (default: 1 = full 60fps)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_episodes", type=int, default=None)

    args = parser.parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    tasks = ALL_TASKS if args.task == ["all"] else args.task

    print(f"\n{'#'*60}")
    print(f" UniVTAC DF-Tactile Data Conversion")
    print(f" Tasks: {tasks}")
    print(f" Subsample: {args.subsample}x (FPS: {BASE_FPS} → {BASE_FPS // args.subsample})")
    print(f" Output: {output_root}")
    print(f"{'#'*60}\n")

    ok = 0
    for task in tasks:
        if task not in TASK_INSTRUCTIONS:
            print(f"WARNING: Unknown task '{task}'. Available: {ALL_TASKS}")
            continue
        if convert_task(task, output_root, args.subsample, args.overwrite, args.max_episodes):
            ok += 1

    print(f"\n{'='*60}")
    print(f" Conversion complete: {ok}/{len(tasks)} tasks successful")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
