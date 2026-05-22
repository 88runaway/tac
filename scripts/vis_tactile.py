"""
Run one eval episode and save a video with tactile modalities visualized separately
(instead of the single rgb_marker strip in the default eval composite frame).
"""
import sys

sys.path.append(".")
sys.path.append("./policy")

import os
import time
import json
import yaml
import argparse
from pathlib import Path
from typing import Literal

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize tactile modalities during one eval rollout")
parser.add_argument("task_name", type=str, help="Task name")
parser.add_argument("task_config", type=str, default="demo", help="Task config stem")
parser.add_argument("deploy_config", type=str, default="ACT/config/deploy", help="Deploy config path")
parser.add_argument("--seed", type=int, default=-1, help="Eval seed (-1: 1000000 * (1 + deploy seed))")
parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: under eval_result/.../tactile_vis)")
parser.add_argument("--no_policy", action="store_true", help="Only reset and step with zero actions (debug env)")
AppLauncher.add_app_launcher_args(parser)

args_cli, _unknown_args = parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.livestream = 2
args_cli.num_envs = 1
args_cli.save_video = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib
import traceback

import cv2
import numpy as np
import torch
from envs.utils.data import VideoHandler

log_path = Path("./log")


def get_config(file, default_root: Path, type: Literal["yaml", "json"]):
    if type == "yaml":
        if file.endswith(".yml") or file.endswith(".yaml"):
            file = Path(file)
        else:
            file = default_root / f"{file}.yml"
        with open(file, "r") as f:
            config = yaml.load(f.read(), Loader=yaml.FullLoader)
        return config, file
    if file.endswith(".json"):
        file = Path(file)
    else:
        file = default_root / f"{file}.json"
    with open(file, "r") as f:
        config = json.load(f)
    return config, file


def log(msg):
    global log_path
    msg = f"[{time.strftime(r'%Y-%m-%d %H:%M:%S')}] {msg}"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(msg + "\n")
    print(msg)


def _to_hwc_uint8(img) -> np.ndarray:
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu()
    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[0] != img.shape[-1]:
        img = img.permute(1, 2, 0)
    if img.ndim == 2:
        img = img.unsqueeze(-1)
    if img.dtype.is_floating_point:
        if img.max() <= 1.0:
            img = (img * 255.0).clamp(0, 255)
        img = img.to(torch.uint8)
    arr = img.numpy() if isinstance(img, torch.Tensor) else np.asarray(img)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return arr


def _resize_rgb(arr: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(arr, (size, size), interpolation=cv2.INTER_LINEAR)


def _vis_depth(depth, size: int, far_plane: float = 34.0) -> np.ndarray:
    d = depth.detach().cpu().numpy() if isinstance(depth, torch.Tensor) else np.asarray(depth)
    d = np.squeeze(d).astype(np.float32)
    d = np.clip(d, 0, far_plane)
    d_norm = (d / max(far_plane, 1e-6) * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(d_norm, cv2.COLORMAP_VIRIDIS)
    return _resize_rgb(colored, size)


def _vis_marker(marker, size: int) -> np.ndarray:
    """Render marker motion as a dot field (displacement magnitude)."""
    m = marker.detach().cpu().numpy() if isinstance(marker, torch.Tensor) else np.asarray(marker)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    if m.size == 0:
        return canvas

    # TacEx: (2, num_markers, 2) — channel 1 is often marker uv in image space
    if m.ndim == 3 and m.shape[0] == 2:
        pts = m[1]
    elif m.ndim == 3:
        pts = m[0]
    else:
        pts = m.reshape(-1, 2)

    pts = np.asarray(pts, dtype=np.float32)
    if pts.shape[-1] != 2:
        return canvas

    # Normalize to canvas; support uv in [0,1] or pixel coords
    xs, ys = pts[:, 0], pts[:, 1]
    if xs.max() <= 1.5 and ys.max() <= 1.5:
        px = (xs * (size - 1)).astype(np.int32)
        py = (ys * (size - 1)).astype(np.int32)
    else:
        px = np.clip(xs.astype(np.int32), 0, size - 1)
        py = np.clip(ys.astype(np.int32), 0, size - 1)

    for x, y in zip(px, py):
        cv2.circle(canvas, (int(x), int(y)), 2, (0, 255, 0), -1)
    return canvas


def _label_panel(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(
        out, text, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
        (255, 255, 255), 2, cv2.LINE_AA,
    )
    return out


def build_tactile_vis_frame(obs: dict, far_plane: float = 34.0, panel: int = 160) -> np.ndarray:
    """
    Layout (W x H = 1280 x 640):
      [ head 640x320 | wrist 640x320 ]
      [ L: rgb | rgb_marker | depth | marker | R: rgb | rgb_marker | depth | marker ]  each panel x panel
    bottom width = 4 * panel * 2 = 1280 (panel=160), top also 640+640=1280.
    """
    n_modalities = 4  # rgb, rgb_marker, depth, marker
    bottom_w = n_modalities * panel * 2  # 1280 when panel=160
    top_w_half = bottom_w // 2           # 640

    head = _to_hwc_uint8(obs["observation"]["head"]["rgb"])
    if "wrist" in obs["observation"]:
        wrist = _to_hwc_uint8(obs["observation"]["wrist"]["rgb"])
        wrist_r = cv2.resize(wrist, (top_w_half, 320))
    else:
        wrist_r = np.zeros((320, top_w_half, 3), dtype=np.uint8)
    head_r = cv2.resize(head, (top_w_half, 320))
    top = np.concatenate([head_r, wrist_r], axis=1)

    modalities = [
        ("rgb", lambda t: _resize_rgb(_to_hwc_uint8(t["rgb"]), panel)),
        ("rgb_marker", lambda t: _resize_rgb(_to_hwc_uint8(t["rgb_marker"]), panel)),
        ("depth", lambda t: _vis_depth(t["depth"], panel, far_plane)),
        ("marker", lambda t: _vis_marker(t["marker"], panel)),
    ]

    def finger_row(side: str) -> np.ndarray:
        tac = obs["tactile"][side]
        panels = [_label_panel(fn(tac), name) for name, fn in modalities]
        return np.concatenate(panels, axis=1)

    bottom = np.concatenate([finger_row("left_tactile"), finger_row("right_tactile")], axis=1)
    frame = np.concatenate([top, bottom], axis=0)
    return frame


def run_one_episode(task, policy, seed: int, instructions: dict, instruction_type: str):
    task.mode = "eval"
    task.reset(seed=seed, instructions=instructions[instruction_type])
    if not task.plan_success:
        raise RuntimeError(f"pre_move failed for seed {seed}")

    policy.reset()
    far_plane = getattr(task.cfg.robot, "tactile_far_plane", 34.0)

    video_path = task.save_root / "video" / f"{seed}_tactile_vis.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    frame0 = build_tactile_vis_frame(task._get_observations(), far_plane=far_plane)
    h, w = frame0.shape[:2]
    vh = VideoHandler()
    vh.reset(video_path, (w, h))

    def write_frame(obs):
        frame = build_tactile_vis_frame(obs, far_plane=far_plane)
        vh.write(torch.from_numpy(frame))

    write_frame(task._get_observations())

    while task.take_action_cnt < task.cfg.step_lim:
        obs = task._get_observations()
        if not args_cli.no_policy:
            policy.eval(task, obs)
        else:
            qpos = obs["embodiment"]["joint"][:8]
            task.take_action(qpos, action_type="qpos")
        write_frame(obs)
        if task.eval_success or task.check_early_stop():
            break
        if not task.plan_success:
            break

    result = "success" if task.eval_success else "failed"
    vh.close(result)
    task.clean_cache(result=result)
    return video_path, result, task.eval_success


def main():
    global args_cli

    task_file_name = args_cli.task_name

    task_config, task_config_file = get_config(
        args_cli.task_config,
        default_root=Path(__file__).parent.parent / "task_config",
        type="yaml",
    )
    deploy_config, deploy_config_file = get_config(
        args_cli.deploy_config,
        default_root=Path(__file__).parent.parent / "policy",
        type="yaml",
    )
    policy_name = deploy_config["policy_name"]
    deploy_config["task_name"] = task_file_name
    deploy_config["task_config"] = task_config_file.stem
    deploy_config["save_video"] = False

    if deploy_config.get("instuction_file") is not None:
        instructions, _ = get_config(
            deploy_config["instuction_file"],
            default_root=Path(__file__).parent.parent / "instructions",
            type="json",
        )
    else:
        instructions = {"seen": ["Empty"], "unseen": ["Empty"]}

    task_module = importlib.import_module(f"envs.{task_file_name}")
    policy_module = importlib.import_module(f"policy.{policy_name}")

    curr_time = time.strftime(r"%Y-%m-%d_%H:%M:%S")
    model_tag = os.environ.get("CKPT_CONFIG", "univtac")

    env_cfg = task_module.TaskCfg()
    base_save = Path(args_cli.output_dir) if args_cli.output_dir else (
        Path("eval_result") / policy_name / task_file_name / model_tag / "tactile_vis" / curr_time
    )
    env_cfg.save_dir = base_save
    env_cfg.decimation = task_config.get("decimation", env_cfg.decimation)
    obs_cfg = task_config.get("observations", {})
    # Ensure all tactile modalities needed for visualization are requested
    tactile_types = list(obs_cfg.get("tactile", ["rgb", "rgb_marker", "marker", "depth", "pose"]))
    for key in ("rgb", "rgb_marker", "marker", "depth"):
        if key not in tactile_types:
            tactile_types.append(key)
    env_cfg.obs_data_type = {**obs_cfg, "tactile": tactile_types}
    env_cfg.video_frequency = 0
    env_cfg.random_texture = task_config.get("random_texture", False)
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    deploy_seed = deploy_config.get("seed", 0)
    eval_seed = args_cli.seed if args_cli.seed >= 0 else 1000000 * (1 + deploy_seed)

    policy = policy_module.Policy(deploy_config)
    task = task_module.Task(env_cfg, mode="eval")

    if os.environ.get("TRAIN_CONFIG"):
        deploy_config["train_config"] = os.environ["TRAIN_CONFIG"]

    global log_path
    log_path = task.save_root / "log.log"
    log(f"Task: {task_file_name}, seed: {eval_seed}")
    log(f"Tactile vis output: {task.save_root}")

    try:
        video_path, result, succ = run_one_episode(
            task, policy, eval_seed, instructions,
            deploy_config.get("instruction_type", "seen"),
        )
        log(f"Done: {result}, success={succ}, video={video_path}")
        print(f"\nSaved tactile visualization video:\n  {video_path}\n")
    except Exception as e:
        log(f"Failed: {e}\n{traceback.format_exc()}")
        raise
    finally:
        task.close()
        policy.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
