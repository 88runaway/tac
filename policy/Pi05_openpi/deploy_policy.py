"""
Pi0.5 JAX (openpi) deployment policy for UniVTAC benchmark evaluation.

Architecture (process isolation):
  ┌─────────────────────────────────┐      Unix socket      ┌───────────────────────────────────────┐
  │  UniVTAC  (Python 3.10)         │ ◄──────────────────► │  JAXInferenceServer (Python 3.11)      │
  │  Isaac Lab + deploy_policy.py   │   observation/action  │  openpi + JAX + Pi05 policy            │
  └─────────────────────────────────┘                       └───────────────────────────────────────┘

deploy_policy.py (this file) runs entirely in Python 3.10 and contains NO
openpi/JAX imports.  All inference is delegated to a subprocess that
uses the openpi conda environment (Python 3.11 + JAX).

The observation format sent to the server uses HWC uint8 images (matching
openpi's expected input format), unlike the lerobot variant which sends
CHW float32.
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

OPENPI_PYTHON = Path("/data1/zjb/miniconda3/envs/openpi/bin/python")
SERVER_SCRIPT = Path(__file__).parent / "pi05_jax_inference_server.py"


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
        """Start the JAX Pi0.5 inference server subprocess and connect via socket."""
        self.task_name = args["task_name"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(Path(__file__).parent.parent / "task_settings.json", "r") as f:
            task_settings = json.load(f)
        assert self.task_name in task_settings, (
            f"Task '{self.task_name}' not found in task_settings.json"
        )
        self.camera_type = task_settings[self.task_name].get("camera_type", "head")

        ckpt_dir = self._resolve_ckpt_dir(args)
        args["ckpt_dir"] = str(ckpt_dir)
        print(f"[Pi05_JAX] Checkpoint: {ckpt_dir}")

        self._n_action_steps = 10
        n_action_steps = args.get("n_action_steps")
        if n_action_steps is not None and str(n_action_steps).strip() not in ("", "null", "None"):
            self._n_action_steps = int(n_action_steps)

        self._num_inference_steps = None
        num_inference_steps = args.get("num_inference_steps")
        if num_inference_steps is not None and str(num_inference_steps).strip() not in ("", "null", "None"):
            self._num_inference_steps = int(num_inference_steps)

        # Diffusion forcing inference options (only used by Pi0DF checkpoints)
        self._infer_time_schedule = None
        infer_time_schedule = args.get("infer_time_schedule")
        if infer_time_schedule is not None and str(infer_time_schedule).strip() not in ("", "null", "None"):
            self._infer_time_schedule = str(infer_time_schedule).strip()

        # block_size: overrides the training-config num_blocks for the blockwise schedule.
        # num_blocks = action_horizon // block_size (computed in the server after model load).
        self._block_size = None
        block_size = args.get("block_size")
        if block_size is not None and str(block_size).strip() not in ("", "null", "None"):
            self._block_size = int(block_size)

        self._socket_path = tempfile.mktemp(prefix="/tmp/pi05_jax_sock_")
        self._sock = None
        self._proc = None
        self._start_server(ckpt_dir, args)

        ni_str = str(self._num_inference_steps) if self._num_inference_steps is not None else "default(10)"
        bs_str = str(self._block_size) if self._block_size is not None else "default(from config)"
        print(f"[Pi05_JAX] Server ready. n_action_steps={self._n_action_steps}, "
              f"num_inference_steps={ni_str}, block_size={bs_str}, camera={self.camera_type}")

    def _start_server(self, ckpt_dir: Path, args: dict):
        """Launch JAX inference server subprocess and perform handshake."""
        env = os.environ.copy()
        gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        env["CUDA_VISIBLE_DEVICES"] = gpu_id

        _BLOCK_KEYWORDS = ("isaac", "omni", "kit", "pip_prebundle", "omniverse", "carb")
        clean_paths = [
            p for p in env.get("PYTHONPATH", "").split(":")
            if p and not any(kw in p.lower() for kw in _BLOCK_KEYWORDS)
        ]
        openpi_src = str(Path("/data1/zjb/openpi/src"))
        if openpi_src not in clean_paths:
            clean_paths.insert(0, openpi_src)
        env["PYTHONPATH"] = ":".join(clean_paths)

        env.pop("LD_PRELOAD", None)

        self._proc = subprocess.Popen(
            [str(OPENPI_PYTHON), str(SERVER_SCRIPT), "--socket", self._socket_path],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        for _ in range(240):
            if os.path.exists(self._socket_path):
                break
            if self._proc.poll() is not None:
                out = self._proc.stdout.read().decode(errors="replace")
                raise RuntimeError(f"[Pi05_JAX] Server process died early:\n{out}")
            time.sleep(0.5)
        else:
            raise TimeoutError("[Pi05_JAX] Server socket not created within 120s")

        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.connect(self._socket_path)

        config_name = args.get("config_name", "pi05_univtac")

        init_args = {
            "ckpt_dir": str(ckpt_dir),
            "config_name": config_name,
            "camera_type": self.camera_type,
            "n_action_steps": self._n_action_steps,
            "num_inference_steps": self._num_inference_steps,
            "infer_time_schedule": self._infer_time_schedule,
            "block_size": self._block_size,
        }

        _send_msg(self._sock, _encode({"cmd": "init", "args": init_args}))
        reply = _decode(_recv_msg(self._sock))
        if reply.get("status") != "ok":
            raise RuntimeError(f"[Pi05_JAX] Server init failed: {reply.get('msg')}")

    def _resolve_ckpt_dir(self, args) -> Path:
        if args.get("ckpt_dir"):
            ckpt_dir = Path(args["ckpt_dir"])
        elif os.environ.get("CKPT_DIR"):
            ckpt_dir = Path(os.environ["CKPT_DIR"])
        else:
            raise ValueError(
                "[Pi05_JAX] Must specify ckpt_dir in deploy config or CKPT_DIR env var. "
                "JAX checkpoints use orbax format and cannot be auto-discovered from training outputs."
            )

        assert ckpt_dir.exists(), f"Checkpoint dir not found: {ckpt_dir}"
        return ckpt_dir

    def _preprocess_image(self, img_tensor: torch.Tensor) -> np.ndarray:
        """HWC uint8 [0,255] torch tensor → HWC uint8 numpy, resized to 224x224.

        openpi expects HWC uint8 images (unlike lerobot which expects CHW float32).
        """
        img = img_tensor.float().permute(2, 0, 1) / 255.0
        img = transforms.functional.resize(img, [224, 224], antialias=True)
        img = (img * 255.0).clamp(0, 255).byte()
        img = img.permute(1, 2, 0)
        return img.cpu().numpy()

    def eval(self, task, observation):
        """Send observation to server, receive action, execute in environment."""
        head_img = self._preprocess_image(observation["observation"]["head"]["rgb"])
        state = observation["embodiment"]["joint"][:8].float().cpu().numpy().astype(np.float32)

        images = {"base_0_rgb": _arr_to_bytes(head_img)}
        if self.camera_type == "all" and "wrist" in observation.get("observation", {}):
            wrist_img = self._preprocess_image(observation["observation"]["wrist"]["rgb"])
            images["left_wrist_0_rgb"] = _arr_to_bytes(wrist_img)

        msg = {
            "cmd": "infer",
            "images": images,
            "state": _arr_to_bytes(state),
            "task": task.instruction,
        }
        _send_msg(self._sock, _encode(msg))
        reply = _decode(_recv_msg(self._sock))

        if "action" not in reply:
            print(f"[Pi05_JAX] Full server error:\n{reply.get('msg', '(no message)')}", flush=True)
            raise RuntimeError(f"[Pi05_JAX] Infer error: {reply.get('msg', '').splitlines()[0]}")

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
