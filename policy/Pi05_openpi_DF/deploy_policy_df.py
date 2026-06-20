"""
Pi0DF deploy policy — Diffusion Forcing dedicated.

Launches pi0df_jax_inference_server.py as a subprocess (openpi conda env, Python 3.11)
and communicates over a Unix domain socket. This policy is for Pi0DF checkpoints only.

Architecture:
  UniVTAC (Python 3.10)  ←──socket──→  Pi0DFInferenceServer (Python 3.11, JAX)
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
SERVER_SCRIPT  = Path(__file__).parent / "pi0df_jax_inference_server.py"


def _send_msg(sock: socket.socket, data: bytes):
    sock.sendall(struct.pack(">I", len(data)) + data)


def _recv_msg(sock: socket.socket) -> bytes:
    raw = _recv_exactly(sock, 4)
    if not raw:
        return b""
    (length,) = struct.unpack(">I", raw)
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
    def __init__(self, args: dict):
        self.task_name = args["task_name"]
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(Path(__file__).parent.parent / "task_settings.json") as f:
            task_settings = json.load(f)
        assert self.task_name in task_settings, (
            f"Task '{self.task_name}' not found in task_settings.json"
        )
        self.camera_type = task_settings[self.task_name].get("camera_type", "head")

        ckpt_dir = self._resolve_ckpt_dir(args)
        args["ckpt_dir"] = str(ckpt_dir)
        print(f"[Pi0DF] Checkpoint: {ckpt_dir}")

        # ── inference params ──────────────────────────────────────────────────
        self._n_action_steps = int(args.get("n_action_steps") or 10)

        ni = args.get("num_inference_steps")
        self._num_inference_steps = int(ni) if ni and str(ni).strip() not in ("", "null", "None") else None

        its = args.get("infer_time_schedule")
        self._infer_time_schedule = str(its).strip() if its and str(its).strip() not in ("", "null", "None") else None

        bs = args.get("block_size")
        self._block_size = int(bs) if bs and str(bs).strip() not in ("", "null", "None") else None

        self._use_tactile = bool(args.get("use_tactile", False))

        bsz = args.get("block_size")
        self._deploy_block_size = int(bsz) if bsz and str(bsz).strip() not in ("", "null", "None") else 5

        self._socket_path = tempfile.mktemp(prefix="/tmp/pi0df_sock_")
        self._sock  = None
        self._proc  = None
        self._start_server(ckpt_dir, args)

        ni_str = str(self._num_inference_steps) if self._num_inference_steps else "default(50)"
        bs_str = str(self._block_size) if self._block_size else "from config"
        print(
            f"[Pi0DF] Server ready. n_action_steps={self._n_action_steps}, "
            f"num_inference_steps={ni_str}, block_size={bs_str}, "
            f"infer_time_schedule={self._infer_time_schedule or 'blockwise'}, "
            f"camera={self.camera_type}, use_tactile={self._use_tactile}"
        )

    def _start_server(self, ckpt_dir: Path, args: dict):
        env = os.environ.copy()
        gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        env["CUDA_VISIBLE_DEVICES"] = gpu_id

        _BLOCK = ("isaac", "omni", "kit", "pip_prebundle", "omniverse", "carb")
        clean  = [p for p in env.get("PYTHONPATH", "").split(":")
                  if p and not any(k in p.lower() for k in _BLOCK)]
        openpi_src = str(Path("/data1/zjb/UniVTAC/openpi/src"))
        if openpi_src not in clean:
            clean.insert(0, openpi_src)
        env["PYTHONPATH"] = ":".join(clean)
        env.pop("LD_PRELOAD", None)

        self._proc = subprocess.Popen(
            [str(OPENPI_PYTHON), str(SERVER_SCRIPT), "--socket", self._socket_path],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )

        for _ in range(240):
            if os.path.exists(self._socket_path):
                break
            if self._proc.poll() is not None:
                out = self._proc.stdout.read().decode(errors="replace")
                raise RuntimeError(f"[Pi0DF] Server died early:\n{out}")
            time.sleep(0.5)
        else:
            raise TimeoutError("[Pi0DF] Server socket not created within 120s")

        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.connect(self._socket_path)

        init_args = {
            "ckpt_dir":            str(ckpt_dir),
            "config_name":         args.get("config_name", "pi05_univtac_df"),
            "camera_type":         self.camera_type,
            "n_action_steps":      self._n_action_steps,
            "num_inference_steps": self._num_inference_steps,
            "infer_time_schedule": self._infer_time_schedule,
            "block_size":          self._block_size,
        }
        _send_msg(self._sock, _encode({"cmd": "init", "args": init_args}))
        reply = _decode(_recv_msg(self._sock))
        if reply.get("status") != "ok":
            raise RuntimeError(f"[Pi0DF] Server init failed: {reply.get('msg')}")

    def _resolve_ckpt_dir(self, args: dict) -> Path:
        if args.get("ckpt_dir"):
            d = Path(args["ckpt_dir"])
        elif os.environ.get("CKPT_DIR"):
            d = Path(os.environ["CKPT_DIR"])
        else:
            raise ValueError("[Pi0DF] Must specify ckpt_dir or CKPT_DIR env var.")
        assert d.exists(), f"Checkpoint dir not found: {d}"
        return d

    def _preprocess_image(self, img: torch.Tensor) -> np.ndarray:
        """HWC uint8 tensor → resized HWC uint8 numpy (224×224)."""
        img = img.float().permute(2, 0, 1) / 255.0
        img = transforms.functional.resize(img, [224, 224], antialias=True)
        img = (img * 255.0).clamp(0, 255).byte().permute(1, 2, 0)
        return img.cpu().numpy()

    def _get_tactile_images(self, observation) -> tuple[np.ndarray, np.ndarray]:
        """Extract and preprocess left/right tactile rgb_marker from observation."""
        tac = observation.get("tactile", {})
        left = tac.get("left_gsmini", {}).get("rgb_marker")
        right = tac.get("right_gsmini", {}).get("rgb_marker")
        if left is None or right is None:
            return np.zeros((224, 224, 3), dtype=np.uint8), np.zeros((224, 224, 3), dtype=np.uint8)
        if isinstance(left, torch.Tensor):
            left = self._preprocess_image(left)
        if isinstance(right, torch.Tensor):
            right = self._preprocess_image(right)
        return np.asarray(left, dtype=np.uint8), np.asarray(right, dtype=np.uint8)

    def eval(self, task, observation):
        if self._use_tactile:
            return self._eval_tactile(task, observation)
        return self._eval_standard(task, observation)

    def _eval_standard(self, task, observation):
        """Original non-tactile eval path."""
        head_img = self._preprocess_image(observation["observation"]["head"]["rgb"])
        state    = observation["embodiment"]["joint"][:8].float().cpu().numpy().astype(np.float32)

        images = {"base_0_rgb": _arr_to_bytes(head_img)}
        if self.camera_type == "all" and "wrist" in observation.get("observation", {}):
            wrist = self._preprocess_image(observation["observation"]["wrist"]["rgb"])
            images["left_wrist_0_rgb"] = _arr_to_bytes(wrist)

        msg = {"cmd": "infer", "images": images,
               "state": _arr_to_bytes(state), "task": task.instruction}
        _send_msg(self._sock, _encode(msg))
        reply = _decode(_recv_msg(self._sock))

        if "action" not in reply:
            print(f"[Pi0DF] Server error:\n{reply.get('msg', '')}", flush=True)
            raise RuntimeError(f"[Pi0DF] Infer error: {reply.get('msg', '').splitlines()[0]}")

        action_np = np.load(io.BytesIO(reply["action"]))
        task.take_action(torch.from_numpy(action_np).to(task.device).float(), action_type="qpos")

    def _eval_tactile(self, task, observation):
        """Block-level tactile feedback eval path.

        For each chunk: send initial obs + tactile → receive block 0 actions →
        execute → read new tactile → send back → receive block 1 → … until all
        blocks are done.
        """
        head_img = self._preprocess_image(observation["observation"]["head"]["rgb"])
        state = observation["embodiment"]["joint"][:8].float().cpu().numpy().astype(np.float32)
        tac_left, tac_right = self._get_tactile_images(observation)

        images = {"base_0_rgb": _arr_to_bytes(head_img)}
        if self.camera_type == "all" and "wrist" in observation.get("observation", {}):
            wrist = self._preprocess_image(observation["observation"]["wrist"]["rgb"])
            images["left_wrist_0_rgb"] = _arr_to_bytes(wrist)

        # Start interactive session
        msg = {
            "cmd": "infer_tactile_start",
            "images": images,
            "state": _arr_to_bytes(state),
            "task": task.instruction,
            "tactile_left": _arr_to_bytes(tac_left),
            "tactile_right": _arr_to_bytes(tac_right),
        }
        _send_msg(self._sock, _encode(msg))
        reply = _decode(_recv_msg(self._sock))

        if "block_actions" not in reply:
            raise RuntimeError(f"[Pi0DF] tactile_start error: {reply.get('msg', '')}")

        # Execute block 0
        block_actions = np.load(io.BytesIO(reply["block_actions"]))
        for a in block_actions:
            task.take_action(torch.from_numpy(a).to(task.device).float(), action_type="qpos")

        # Continue: for each remaining block
        while True:
            # Get fresh observation with new tactile
            new_obs = task._get_observations()
            tac_left, tac_right = self._get_tactile_images(new_obs)

            msg = {
                "cmd": "infer_tactile_continue",
                "tactile_left": _arr_to_bytes(tac_left),
                "tactile_right": _arr_to_bytes(tac_right),
            }
            _send_msg(self._sock, _encode(msg))
            reply = _decode(_recv_msg(self._sock))

            if "block_actions" not in reply:
                raise RuntimeError(f"[Pi0DF] tactile_continue error: {reply.get('msg', '')}")

            block_actions = np.load(io.BytesIO(reply["block_actions"]))
            for a in block_actions:
                task.take_action(torch.from_numpy(a).to(task.device).float(), action_type="qpos")

            if reply.get("done", False):
                break

    def reset(self):
        if self._sock:
            _send_msg(self._sock, _encode({"cmd": "reset"}))
            _decode(_recv_msg(self._sock))

    def close(self):
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
