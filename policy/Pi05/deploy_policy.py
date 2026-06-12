"""
Pi0.5 deployment policy for UniVTAC benchmark evaluation.

Architecture (process isolation):
  ┌─────────────────────────────────┐      Unix socket      ┌──────────────────────────────────┐
  │  UniVTAC  (Python 3.10)         │ ◄──────────────────► │  Pi05InferenceServer (Python 3.12) │
  │  Isaac Lab + deploy_policy.py   │   observation/action  │  lerobot + PI05Policy              │
  └─────────────────────────────────┘                       └──────────────────────────────────┘

deploy_policy.py (this file) runs entirely in Python 3.10 and contains NO
top-level lerobot imports.  All inference is delegated to a subprocess that
uses the lerobot conda environment (Python 3.12).
"""

import sys
import os
import json
import time
import struct
import socket
import subprocess
import io
import tempfile
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from _base_policy import BasePolicy

import numpy as np
import torch
from torchvision import transforms

LEROBOT_PYTHON = Path("/data1/zjb/miniconda3/envs/lerobot/bin/python")
SERVER_SCRIPT  = Path(__file__).parent / "pi05_inference_server.py"


def _has_weights(d: Path) -> bool:
    """LoRA checkpoints have adapter_model.safetensors; full checkpoints have model.safetensors."""
    return d.exists() and (
        (d / "model.safetensors").exists() or
        (d / "adapter_model.safetensors").exists()
    )


def _send_msg(sock: socket.socket, data: bytes):
    sock.sendall(struct.pack(">I", len(data)) + data)


def _recv_msg(sock: socket.socket) -> bytes:
    raw_len = _recv_exactly(sock, 4)
    if not raw_len:
        return b""
    (length,) = struct.unpack(">I", raw_len)
    return _recv_exactly(sock, length)


def _recv_exactly(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return b""
        buf += chunk
    return buf


def _encode(obj: dict) -> bytes:
    import msgpack
    return msgpack.packb(obj, use_bin_type=True)


def _decode(data: bytes) -> dict:
    import msgpack
    return msgpack.unpackb(data, raw=False)


def _arr_to_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


class Policy(BasePolicy):
    def __init__(self, args):
        """Start the Pi0.5 inference server subprocess and connect via socket."""
        self.task_name = args["task_name"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(Path(__file__).parent.parent / "task_settings.json", "r") as f:
            task_settings = json.load(f)
        assert self.task_name in task_settings, (
            f"Task '{self.task_name}' not found in task_settings.json"
        )
        self.camera_type = task_settings[self.task_name].get("camera_type", "head")
        self.model_type = args.get("model_type", "vision_only")

        ckpt_dir = self._resolve_ckpt_dir(args)
        print(f"[Pi05] Checkpoint: {ckpt_dir}")
        self._log_ckpt_step(ckpt_dir)

        self._socket_path = tempfile.mktemp(prefix="/tmp/pi05_sock_")
        self._sock = None
        self._proc = None
        self._start_server(ckpt_dir, args)

        # Image preprocessing params (read from server's init reply)
        self._image_size = self._init_image_size
        print(f"[Pi05] Server ready. n_action_steps={self._n_action_steps}, "
              f"rtc={self._rtc_enabled}, image_size={self._image_size}, "
              f"camera={self.camera_type}, model={self.model_type}")

    # ──────────────────────────────────────────────────────────────────────────
    # Server lifecycle
    # ──────────────────────────────────────────────────────────────────────────

    def _start_server(self, ckpt_dir: Path, args: dict):
        """Launch inference server subprocess and perform handshake."""
        env = os.environ.copy()
        gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        env["CUDA_VISIBLE_DEVICES"] = gpu_id

        # Strip Isaac Sim / omniverse paths from PYTHONPATH so the lerobot
        # Python 3.12 subprocess uses its own packages, not Isaac Sim's bundled
        # pip_prebundle (which includes numpy compiled for Python 3.10).
        _BLOCK_KEYWORDS = ("isaac", "omni", "kit", "pip_prebundle", "omniverse", "carb")
        clean_paths = [
            p for p in env.get("PYTHONPATH", "").split(":")
            if p and not any(kw in p.lower() for kw in _BLOCK_KEYWORDS)
        ]
        # Prepend lerobot src so it takes precedence
        lerobot_src_path = str(Path("/data1/zjb/lerobot/src"))
        if lerobot_src_path not in clean_paths:
            clean_paths.insert(0, lerobot_src_path)
        env["PYTHONPATH"] = ":".join(clean_paths)

        # Also clear LD_PRELOAD (Isaac Sim sets libstdc++ which may conflict)
        env.pop("LD_PRELOAD", None)

        self._proc = subprocess.Popen(
            [str(LEROBOT_PYTHON), str(SERVER_SCRIPT), "--socket", self._socket_path],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        # Wait for server to create socket (up to 60s)
        for _ in range(120):
            if os.path.exists(self._socket_path):
                break
            if self._proc.poll() is not None:
                out = self._proc.stdout.read().decode(errors="replace")
                raise RuntimeError(f"[Pi05] Server process died early:\n{out}")
            time.sleep(0.5)
        else:
            raise TimeoutError("[Pi05] Server socket not created within 60s")

        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.connect(self._socket_path)

        # Resolve n_action_steps override
        n_action_steps = args.get("n_action_steps")
        if n_action_steps is not None and str(n_action_steps).strip() not in ("", "null", "None"):
            n_action_steps = int(n_action_steps)
        else:
            n_action_steps = None

        # Resolve RTC parameters
        rtc_enabled_raw = str(args.get("rtc_enabled", "false")).strip().lower()
        rtc_enabled = rtc_enabled_raw in ("true", "1", "yes")

        rtc_execution_horizon = args.get("rtc_execution_horizon")
        if rtc_execution_horizon is not None and str(rtc_execution_horizon).strip() not in ("", "null", "None"):
            rtc_execution_horizon = int(rtc_execution_horizon)
        else:
            rtc_execution_horizon = None

        num_inference_steps = args.get("num_inference_steps")
        if num_inference_steps is not None and str(num_inference_steps).strip() not in ("", "null", "None"):
            num_inference_steps = int(num_inference_steps)
        else:
            num_inference_steps = None

        init_args = {
            "ckpt_dir": str(ckpt_dir),
            "camera_type": self.camera_type,
            "rtc_enabled": rtc_enabled,
        }
        if n_action_steps is not None:
            init_args["n_action_steps"] = n_action_steps
        if rtc_execution_horizon is not None:
            init_args["rtc_execution_horizon"] = rtc_execution_horizon
        if num_inference_steps is not None:
            init_args["num_inference_steps"] = num_inference_steps
        tokenizer_name = args.get("tokenizer_name")
        if tokenizer_name and str(tokenizer_name).strip() not in ("", "null", "None"):
            init_args["tokenizer_name"] = str(tokenizer_name).strip()
        rename_map_override = args.get("rename_map_override")
        if rename_map_override is not None and str(rename_map_override).strip() not in ("null", "None"):
            init_args["rename_map_override"] = rename_map_override

        _send_msg(self._sock, _encode({"cmd": "init", "args": init_args}))
        reply = _decode(_recv_msg(self._sock))
        if reply.get("status") != "ok":
            raise RuntimeError(f"[Pi05] Server init failed: {reply.get('msg')}")

        # Defaults: read from server reply or fall back to known values
        self._init_image_size = (224, 224)
        self._n_action_steps = n_action_steps or 10
        self._rtc_enabled = rtc_enabled

    # ──────────────────────────────────────────────────────────────────────────
    # Checkpoint resolution
    # ──────────────────────────────────────────────────────────────────────────

    def _resolve_ckpt_dir(self, args) -> Path:
        ckpt_step = args.get("ckpt_step")
        if ckpt_step is not None and str(ckpt_step).strip() not in ("", "null", "None"):
            ckpt_step = int(ckpt_step)
        else:
            ckpt_step = None

        if args.get("ckpt_dir"):
            ckpt_dir = Path(args["ckpt_dir"])
        elif os.environ.get("CKPT_DIR"):
            ckpt_dir = Path(os.environ["CKPT_DIR"])
        else:
            ckpt_root = os.environ.get("CKPT_ROOT", None)
            ckpt_config = os.environ.get("CKPT_CONFIG", "train_lora")
            ckpt_timestamp = os.environ.get("CKPT_TIMESTAMP", "")

            if ckpt_root:
                # Legacy layout: CKPT_ROOT/task/config/checkpoints/
                checkpoints_base = Path(ckpt_root) / self.task_name / ckpt_config
            else:
                # New layout: outputs/pi05_<task>_<config>/<timestamp>/checkpoints/
                run_base = (
                    Path(__file__).parent.parent.parent
                    / "outputs"
                    / f"pi05_{self.task_name}_{ckpt_config}"
                )
                checkpoints_base = self._resolve_run_dir(run_base, ckpt_timestamp) / "checkpoints"

            if ckpt_step is not None:
                step_dir = self._find_step_dir(checkpoints_base, ckpt_step)
                ckpt_dir = step_dir / "pretrained_model"
            else:
                ckpt_dir = checkpoints_base / "last" / "pretrained_model"

        if not ckpt_dir.exists():
            parent = ckpt_dir.parent
            if parent.exists():
                candidates = sorted(parent.glob("checkpoint_*"))
                if candidates:
                    ckpt_dir = candidates[-1] / "pretrained_model"

        if ckpt_dir.exists() and not _has_weights(ckpt_dir):
            candidate = ckpt_dir / "pretrained_model"
            if _has_weights(candidate):
                ckpt_dir = candidate

        assert ckpt_dir.exists() and _has_weights(ckpt_dir), (
            f"No model weights (model.safetensors or adapter_model.safetensors) found in {ckpt_dir}. "
            f"Contents: {list(ckpt_dir.iterdir()) if ckpt_dir.exists() else 'dir not found'}"
        )
        return ckpt_dir

    def _resolve_run_dir(self, run_base: Path, timestamp: str) -> Path:
        """Return the timestamped run dir under run_base.

        If timestamp is given (e.g. '2026-06-06_00:07:00'), return that exact subdir.
        Otherwise return the most recently modified subdir (latest run).
        Falls back to run_base itself for legacy (non-timestamped) layouts.
        """
        if not run_base.exists():
            raise FileNotFoundError(f"Training output base not found: {run_base}")

        if timestamp:
            target = run_base / timestamp
            if not target.exists():
                raise FileNotFoundError(
                    f"Timestamp dir not found: {target}\n"
                    f"Available: {sorted(d.name for d in run_base.iterdir() if d.is_dir())}"
                )
            return target

        # Auto-select: prefer timestamped subdirs (YYYY-MM-DD_HH:MM:SS), else use run_base
        subdirs = sorted(
            [d for d in run_base.iterdir() if d.is_dir() and (d / "checkpoints").exists()],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        if subdirs:
            chosen = subdirs[0]
            print(f"[Pi05] Auto-selected run: {chosen.name} (latest of {len(subdirs)})")
            return chosen

        # Legacy layout: no timestamped subdirs, checkpoints/ sits directly under run_base
        return run_base

    def _find_step_dir(self, checkpoints_base: Path, step: int) -> Path:
        if not checkpoints_base.exists():
            raise FileNotFoundError(f"Checkpoints dir not found: {checkpoints_base}")
        for d in checkpoints_base.iterdir():
            if d.is_dir() and d.name != "last":
                try:
                    if int(d.name) == step:
                        return d
                except ValueError:
                    continue
        available = sorted(
            int(d.name) for d in checkpoints_base.iterdir()
            if d.is_dir() and d.name != "last" and d.name.isdigit()
        )
        raise FileNotFoundError(
            f"No checkpoint for step {step} in {checkpoints_base}. Available: {available}"
        )

    def _log_ckpt_step(self, ckpt_dir: Path):
        training_step_file = ckpt_dir.parent / "training_state" / "training_step.json"
        if training_step_file.exists():
            with open(training_step_file) as f:
                step = json.load(f).get("step", "?")
            print(f"[Pi05] Checkpoint step: {step}")
        else:
            step_dir = ckpt_dir.parent
            resolved = step_dir.resolve()
            print(f"[Pi05] Checkpoint step: {resolved.name} (resolved from '{step_dir.name}')")

    # ──────────────────────────────────────────────────────────────────────────
    # Observation preprocessing (Python 3.10 side — no lerobot imports)
    # ──────────────────────────────────────────────────────────────────────────

    def _preprocess_image(self, img_tensor: torch.Tensor) -> np.ndarray:
        """HWC uint8 [0,255] → CHW float32 [0,1], resized to model resolution."""
        img = img_tensor.float().permute(2, 0, 1) / 255.0
        img = transforms.functional.resize(img, list(self._image_size), antialias=True)
        return img.cpu().numpy().astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    # Eval loop
    # ──────────────────────────────────────────────────────────────────────────

    def eval(self, task, observation):
        """Send observation to server, receive action, execute in environment."""
        head_img = self._preprocess_image(observation["observation"]["head"]["rgb"])
        state = observation["embodiment"]["joint"][:8].float().cpu().numpy().astype(np.float32)

        images = {"base_0_rgb": _arr_to_bytes(head_img)}
        if self.camera_type == "all" and "wrist" in observation.get("observation", {}):
            wrist_img = self._preprocess_image(observation["observation"]["wrist"]["rgb"])
            images["left_wrist_0_rgb"] = _arr_to_bytes(wrist_img)
        if self.model_type == "tactile" and "tactile" in observation:
            tactile = observation["tactile"]
            if "left_gsmini" in tactile and "rgb_marker" in tactile["left_gsmini"]:
                left_tac = self._preprocess_image(tactile["left_gsmini"]["rgb_marker"])
                images["right_wrist_0_rgb"] = _arr_to_bytes(left_tac)
            if "right_gsmini" in tactile and "rgb_marker" in tactile["right_gsmini"]:
                right_tac = self._preprocess_image(tactile["right_gsmini"]["rgb_marker"])
                images["extra_0_rgb"] = _arr_to_bytes(right_tac)

        msg = {
            "cmd": "infer",
            "images": images,
            "state": _arr_to_bytes(state),
            "task": task.instruction,
        }
        _send_msg(self._sock, _encode(msg))
        reply = _decode(_recv_msg(self._sock))

        if "action" not in reply:
            print(f"[Pi05] Full server error:\n{reply.get('msg', '(no message)')}", flush=True)
            raise RuntimeError(f"[Pi05] Infer error: {reply.get('msg', '').splitlines()[0]}")

        action_np = np.load(io.BytesIO(reply["action"]))
        action_tensor = torch.from_numpy(action_np).to(task.device).float()
        task.take_action(action_tensor, action_type="qpos")

    def reset(self):
        """Reset policy action queue between episodes."""
        if self._sock:
            _send_msg(self._sock, _encode({"cmd": "reset"}))
            _decode(_recv_msg(self._sock))

    def close(self):
        """Gracefully shut down the inference server subprocess."""
        try:
            if self._sock:
                _send_msg(self._sock, _encode({"cmd": "close"}))
                _recv_msg(self._sock)
                self._sock.close()
                self._sock = None
        except Exception:
            pass
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            self._proc.wait(timeout=10)
