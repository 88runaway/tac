"""
Pi0.5 JAX Inference Server — runs under the openpi conda environment (Python 3.11 + JAX).

Started as a subprocess by deploy_policy.py (UniVTAC Python 3.10).
Communicates via a Unix domain socket using the same length-prefixed msgpack
protocol as the lerobot Pi05 server.

Protocol:
  Client → Server: {"cmd": "init",   "args": {...}}
  Server → Client: {"status": "ok"}

  Client → Server: {"cmd": "reset"}
  Server → Client: {"status": "ok"}

  Client → Server: {"cmd": "infer",
                    "images": {"base_0_rgb": <HWC uint8 npy bytes>, ...},
                    "state":  <8-dim float32 npy bytes>,
                    "task":   "instruction string"}
  Server → Client: {"action": <8-dim float32 npy bytes>}

  Client → Server: {"cmd": "close"}
  Server → Client: {"status": "ok"}
"""

import sys
import os
import struct
import socket
import traceback
import io
from pathlib import Path

import numpy as np


def send_msg(sock: socket.socket, data: bytes):
    sock.sendall(struct.pack(">I", len(data)) + data)


def recv_msg(sock: socket.socket) -> bytes:
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


def encode_msg(obj: dict) -> bytes:
    import msgpack
    return msgpack.packb(obj, use_bin_type=True)


def decode_msg(data: bytes) -> dict:
    import msgpack
    return msgpack.unpackb(data, raw=False)


def arr_to_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def bytes_to_arr(b: bytes) -> np.ndarray:
    return np.load(io.BytesIO(b))


class JAXInferenceServer:
    def __init__(self):
        self.policy = None
        self.action_horizon = 50
        self.n_action_steps = 10
        self.num_inference_steps = None  # None → use openpi default (10)
        self._action_queue: list[np.ndarray] = []

    def handle_init(self, args: dict):
        ckpt_dir = Path(args["ckpt_dir"])
        config_name = args.get("config_name", "pi05_univtac")

        n_action_steps = args.get("n_action_steps")
        if n_action_steps is not None:
            self.n_action_steps = int(n_action_steps)
        else:
            self.n_action_steps = 10

        num_inference_steps = args.get("num_inference_steps")
        if num_inference_steps is not None:
            self.num_inference_steps = int(num_inference_steps)

        gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        first_gpu = gpu_id.split(",")[0].strip()
        os.environ["CUDA_VISIBLE_DEVICES"] = first_gpu
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

        import jax
        print(f"[JAXServer] JAX devices: {jax.devices()}", flush=True)

        from openpi.training import config as _config
        from openpi.policies import policy_config as _policy_config
        from openpi.shared import normalize as _normalize

        train_config = _config.get_config(config_name)
        print(f"[JAXServer] Loading policy with config '{config_name}' from {ckpt_dir}", flush=True)

        # Auto-detect the actual asset_id inside the checkpoint's assets/ dir,
        # since the config's asset_id ("univtac") may differ from the training
        # run's saved asset_id (e.g. "univtac_lift_bottle").
        norm_stats = None
        assets_dir = ckpt_dir / "assets"
        if assets_dir.exists():
            subdirs = [d for d in assets_dir.iterdir() if d.is_dir()]
            if subdirs:
                actual_asset_dir = subdirs[0]
                try:
                    norm_stats = _normalize.load(actual_asset_dir)
                    print(f"[JAXServer] Loaded norm_stats from {actual_asset_dir}", flush=True)
                except Exception as e:
                    print(f"[JAXServer] Warning: failed to load norm_stats from {actual_asset_dir}: {e}", flush=True)

        sample_kwargs = {}
        if self.num_inference_steps is not None:
            sample_kwargs["num_steps"] = self.num_inference_steps

        self.policy = _policy_config.create_trained_policy(
            train_config,
            str(ckpt_dir),
            norm_stats=norm_stats,
            sample_kwargs=sample_kwargs if sample_kwargs else None,
        )

        self.action_horizon = train_config.model.action_horizon
        ni_str = str(self.num_inference_steps) if self.num_inference_steps is not None else "default(10)"
        print(
            f"[JAXServer] Policy loaded. action_horizon={self.action_horizon}, "
            f"n_action_steps={self.n_action_steps}, num_inference_steps={ni_str}",
            flush=True,
        )

    def handle_reset(self):
        self._action_queue.clear()

    def handle_infer(self, msg: dict) -> np.ndarray:
        if self._action_queue:
            return self._action_queue.pop(0)

        head_img = bytes_to_arr(msg["images"]["base_0_rgb"])  # HWC uint8
        state = bytes_to_arr(msg["state"])  # (8,) float32
        instruction = msg["task"]

        has_wrist = "left_wrist_0_rgb" in msg.get("images", {})
        obs = {
            "observation/image": head_img,
            "observation/state": state.astype(np.float32),
            "prompt": instruction,
        }
        if has_wrist:
            wrist_img = bytes_to_arr(msg["images"]["left_wrist_0_rgb"])
            obs["observation/wrist_image"] = wrist_img

        result = self.policy.infer(obs)
        actions = result["actions"]  # (action_horizon, action_dim) or similar

        actions = np.asarray(actions)
        if actions.ndim == 1:
            actions = actions[np.newaxis, :]
        actions = actions[:, :8].astype(np.float32)

        n_steps = min(self.n_action_steps, len(actions))
        for i in range(1, n_steps):
            self._action_queue.append(actions[i])

        return actions[0]

    def run(self, socket_path: str):
        if os.path.exists(socket_path):
            os.unlink(socket_path)

        server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server_sock.bind(socket_path)
        server_sock.listen(1)
        print(f"[JAXServer] Listening on {socket_path}", flush=True)

        conn, _ = server_sock.accept()
        print("[JAXServer] Client connected", flush=True)

        try:
            while True:
                data = recv_msg(conn)
                if not data:
                    break

                msg = decode_msg(data)
                cmd = msg.get("cmd")

                try:
                    if cmd == "init":
                        self.handle_init(msg["args"])
                        send_msg(conn, encode_msg({"status": "ok"}))

                    elif cmd == "reset":
                        self.handle_reset()
                        send_msg(conn, encode_msg({"status": "ok"}))

                    elif cmd == "infer":
                        action = self.handle_infer(msg)
                        send_msg(conn, encode_msg({"action": arr_to_bytes(action)}))

                    elif cmd == "close":
                        send_msg(conn, encode_msg({"status": "ok"}))
                        break

                    else:
                        send_msg(conn, encode_msg({"status": "error", "msg": f"Unknown cmd: {cmd}"}))

                except Exception as e:
                    err = traceback.format_exc()
                    print(f"[JAXServer] Error handling cmd={cmd}: {err}", flush=True)
                    send_msg(conn, encode_msg({"status": "error", "msg": f"{str(e)}\n---TRACEBACK---\n{err}"}))

        finally:
            conn.close()
            server_sock.close()
            if os.path.exists(socket_path):
                os.unlink(socket_path)
            print("[JAXServer] Shutdown", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True, help="Unix socket path")
    args = parser.parse_args()
    JAXInferenceServer().run(args.socket)
