"""
Pi0DF JAX Inference Server — Diffusion Forcing dedicated.

Runs under the openpi conda environment (Python 3.11 + JAX).
Started as a subprocess by deploy_policy_df.py (UniVTAC Python 3.10).
Communicates via a Unix domain socket using length-prefixed msgpack.

This server is dedicated to Pi0DF (Pi0DFConfig) checkpoints only.
It always uses the blockwise pyramid denoising schedule by default.
For standard const-schedule inference on a DF checkpoint, set
infer_time_schedule=const in the deploy config.

Protocol:
  Client → Server: {"cmd": "init",   "args": {...}}
  Server → Client: {"status": "ok"}

  Client → Server: {"cmd": "reset"}
  Server → Client: {"status": "ok"}

  Client → Server: {"cmd": "infer",
                    "images": {"base_0_rgb": <HWC uint8 npy bytes>, ...},
                    "state":  <float32 npy bytes>,
                    "task":   "instruction string"}
  Server → Client: {"action": <float32 npy bytes>}

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


class Pi0DFInferenceServer:
    """Inference server dedicated to Pi0DF (block-wise diffusion forcing) models."""

    def __init__(self):
        self.policy = None
        self.model = None
        self.action_horizon = 50
        self.n_action_steps = 10
        self.num_inference_steps = 50
        self.infer_time_schedule = "blockwise"
        self.block_size = None
        self.num_blocks = None
        self.use_tactile = False

        self._action_queue: list[np.ndarray] = []
        # State for interactive tactile inference
        self._interactive_state = None

    def handle_init(self, args: dict):
        ckpt_dir    = Path(args["ckpt_dir"])
        config_name = args.get("config_name", "pi05_univtac_df")

        if n := args.get("n_action_steps"):
            self.n_action_steps = int(n)
        if ni := args.get("num_inference_steps"):
            self.num_inference_steps = int(ni)
        if its := args.get("infer_time_schedule"):
            its = str(its).strip()
            if its not in ("", "null", "None"):
                self.infer_time_schedule = its
        if bs := args.get("block_size"):
            bs = str(bs).strip()
            if bs not in ("", "null", "None"):
                self.block_size = int(bs)

        first_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
        os.environ["CUDA_VISIBLE_DEVICES"] = first_gpu
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

        import jax
        print(f"[Pi0DFServer] JAX devices: {jax.devices()}", flush=True)

        from openpi.training import config as _config
        from openpi.policies import policy_config as _policy_config
        from openpi.shared import normalize as _normalize

        train_config = _config.get_config(config_name)
        print(f"[Pi0DFServer] Loading '{config_name}' from {ckpt_dir}", flush=True)

        # Enforce this server is used with Pi0DFConfig only.
        model_cls = type(train_config.model).__name__
        if model_cls != "Pi0DFConfig":
            raise RuntimeError(
                f"[Pi0DFServer] Expected Pi0DFConfig but got {model_cls}. "
                f"Use config_name ending in '_df' (e.g. 'pi05_univtac_df')."
            )

        # Override use_tactile from deploy args. The static registry config defaults
        # use_tactile=False, but the checkpoint may have been trained with use_tactile=True
        # via train_df.py --use_tactile true. Pass use_tactile through init_args so that
        # the model is reconstructed with the correct architecture before loading weights.
        use_tactile_override = bool(args.get("use_tactile", False))
        if use_tactile_override and not getattr(train_config.model, "use_tactile", False):
            import dataclasses
            train_config = dataclasses.replace(
                train_config,
                model=dataclasses.replace(train_config.model, use_tactile=True),
            )
            print(f"[Pi0DFServer] use_tactile overridden to True from deploy args", flush=True)

        # Auto-detect norm stats from checkpoint assets/.
        norm_stats = None
        assets_dir = ckpt_dir / "assets"
        if assets_dir.exists():
            subdirs = [d for d in assets_dir.iterdir() if d.is_dir()]
            if subdirs:
                try:
                    norm_stats = _normalize.load(subdirs[0])
                    print(f"[Pi0DFServer] norm_stats ← {subdirs[0]}", flush=True)
                except Exception as e:
                    print(f"[Pi0DFServer] Warning: norm_stats load failed: {e}", flush=True)

        # Resolve num_blocks: eval-time block_size overrides training config.
        action_horizon = train_config.model.action_horizon
        if self.block_size is not None:
            if action_horizon % self.block_size != 0:
                raise RuntimeError(
                    f"[Pi0DFServer] block_size={self.block_size} does not divide "
                    f"action_horizon={action_horizon} evenly."
                )
            num_blocks = action_horizon // self.block_size
            print(
                f"[Pi0DFServer] block_size={self.block_size} → num_blocks={num_blocks} "
                f"(training config had num_blocks={train_config.model.num_blocks})",
                flush=True,
            )
        else:
            num_blocks = train_config.model.num_blocks
            self.block_size = action_horizon // num_blocks

        sample_kwargs = {
            "num_steps":           self.num_inference_steps,
            "infer_time_schedule": self.infer_time_schedule,
            "num_blocks":          num_blocks,
        }

        self.policy = _policy_config.create_trained_policy(
            train_config,
            str(ckpt_dir),
            norm_stats=norm_stats,
            sample_kwargs=sample_kwargs,
        )

        self.action_horizon = action_horizon
        self.num_blocks = num_blocks
        self.use_tactile = getattr(train_config.model, "use_tactile", False)

        # Store action unnormalization stats for the interactive tactile path.
        # policy.infer() applies _output_transform (Unnormalize) automatically, but
        # the interactive path calls _jit_denoise directly and must unnormalize manually.
        self._action_mean = None
        self._action_std = None
        if norm_stats is not None and "actions" in norm_stats:
            self._action_mean = np.array(norm_stats["actions"].mean, dtype=np.float32)
            self._action_std  = np.array(norm_stats["actions"].std,  dtype=np.float32)
            print(f"[Pi0DFServer] action unnorm: mean={self._action_mean}, std={self._action_std}", flush=True)

        # Keep direct access to the raw model for interactive tactile inference.
        if self.use_tactile:
            self.model = self.policy._model  # Pi0DF nnx model
            # JIT-compile the three hot-path methods to avoid per-step Python dispatch.
            # Without JIT, every call traces through Python layer-by-layer (extremely slow).
            from openpi.shared import nnx_utils
            self._jit_prepare   = nnx_utils.module_jit(self.model.prepare_interactive_inference)
            self._jit_encode_tac = nnx_utils.module_jit(
                self.model.encode_tactile, static_argnames=("train",)
            )
            self._jit_denoise   = nnx_utils.module_jit(self.model.denoise_segment)

        print(
            f"[Pi0DFServer] Ready. "
            f"action_horizon={self.action_horizon}, "
            f"block_size={self.block_size}, num_blocks={num_blocks}, "
            f"infer_time_schedule={self.infer_time_schedule}, "
            f"num_inference_steps={self.num_inference_steps}, "
            f"n_action_steps={self.n_action_steps}, "
            f"use_tactile={self.use_tactile}",
            flush=True,
        )

    def handle_reset(self):
        self._action_queue.clear()

    def handle_infer(self, msg: dict) -> np.ndarray:
        """Standard non-tactile inference (unchanged from original)."""
        if self._action_queue:
            return self._action_queue.pop(0)

        head_img    = bytes_to_arr(msg["images"]["base_0_rgb"])
        state       = bytes_to_arr(msg["state"]).astype(np.float32)
        instruction = msg["task"]

        obs = {
            "observation/image": head_img,
            "observation/state": state,
            "prompt":            instruction,
        }
        if "left_wrist_0_rgb" in msg.get("images", {}):
            obs["observation/wrist_image"] = bytes_to_arr(msg["images"]["left_wrist_0_rgb"])

        result  = self.policy.infer(obs)
        actions = np.asarray(result["actions"])
        if actions.ndim == 1:
            actions = actions[np.newaxis, :]
        actions = actions[:, :8].astype(np.float32)

        n_steps = min(self.n_action_steps, len(actions))
        for i in range(1, n_steps):
            self._action_queue.append(actions[i])
        return actions[0]

    def _unnormalize_actions(self, actions: np.ndarray) -> np.ndarray:
        """Undo z-score normalization on action array of shape (..., action_dim_<=8)."""
        if self._action_mean is None:
            return actions
        d = actions.shape[-1]
        mean = self._action_mean[:d]
        std  = self._action_std[:d]
        return actions * (std + 1e-6) + mean

    # ── Interactive tactile inference ─────────────────────────────────────

    def handle_infer_tactile_start(self, msg: dict) -> np.ndarray:
        """Begin a new interactive tactile inference session.

        Protocol:
          Client → Server: {"cmd": "infer_tactile_start",
                            "images": {...}, "state": ..., "task": ...,
                            "tactile_left": <npy>, "tactile_right": <npy>}
          Server → Client: {"block_actions": <npy>, "block_idx": 0}

        The server prepares the prefix KV-cache + noise, encodes the initial
        tactile, runs the first denoising segment, and returns the first clean
        block's actions.  The client should execute them, then call
        ``infer_tactile_continue`` with new tactile images.
        """
        import jax
        import jax.numpy as jnp

        head_img    = bytes_to_arr(msg["images"]["base_0_rgb"])
        state       = bytes_to_arr(msg["state"]).astype(np.float32)
        instruction = msg["task"]
        tac_left    = bytes_to_arr(msg["tactile_left"]).astype(np.float32)
        tac_right   = bytes_to_arr(msg["tactile_right"]).astype(np.float32)

        # Build observation through the policy transforms (reuses the policy pipeline)
        obs_dict = {
            "observation/image": head_img,
            "observation/state": state,
            "prompt":            instruction,
        }
        if "left_wrist_0_rgb" in msg.get("images", {}):
            obs_dict["observation/wrist_image"] = bytes_to_arr(msg["images"]["left_wrist_0_rgb"])

        # Run the policy transforms to get the model-ready observation
        transformed = self.policy._input_transform(obs_dict)
        from openpi.models.model import Observation
        observation = Observation.from_dict(jax.tree.map(lambda x: jnp.array(x)[None], transformed))

        # Normalize tactile to [0,1] float32, add batch dim
        if tac_left.max() > 1.0:
            tac_left = tac_left / 255.0
        if tac_right.max() > 1.0:
            tac_right = tac_right / 255.0
        tac_left_j = jnp.array(tac_left)[None]   # (1, H, W, 3)
        tac_right_j = jnp.array(tac_right)[None]  # (1, H, W, 3)

        model = self.model
        rng = jax.random.PRNGKey(0)

        import time as _time
        # Prepare prefix and noise
        _t0 = _time.perf_counter()
        noise, obs_proc, prefix_tokens, prefix_mask, kv_cache = self._jit_prepare(rng, observation)
        # Force GPU sync so timing is accurate (block on the noise array as a proxy)
        noise.block_until_ready()
        _t1 = _time.perf_counter()
        print(f"[Pi0DF] jit_prepare: {_t1-_t0:.2f}s", flush=True)

        # Encode initial tactile
        tactile_tokens = self._jit_encode_tac(tac_left_j, tac_right_j, train=False)
        tactile_tokens.block_until_ready()
        _t2 = _time.perf_counter()
        print(f"[Pi0DF] jit_encode_tac: {_t2-_t1:.2f}s", flush=True)

        # Build the full blockwise time schedule
        t_schedule = model._blockwise_time_schedule(self.num_inference_steps, self.num_blocks)
        t_schedule = jnp.broadcast_to(
            t_schedule[:, None, :], (t_schedule.shape[0], 1, self.action_horizon)
        )
        dt_schedule = t_schedule[1:] - t_schedule[:-1]
        t_starts = t_schedule[:-1]

        # Determine steps per block
        steps_per_block = self.num_inference_steps // self.num_blocks

        # Run segment 0
        seg_start = 0
        seg_end = steps_per_block
        _t3 = _time.perf_counter()
        x_t = self._jit_denoise(
            noise,
            t_starts[seg_start:seg_end],
            dt_schedule[seg_start:seg_end],
            obs_proc, prefix_tokens, prefix_mask, kv_cache,
            tactile_tokens,
        )
        x_t.block_until_ready()
        _t4 = _time.perf_counter()
        print(f"[Pi0DF] jit_denoise block0: {_t4-_t3:.2f}s (total start: {_t4-_t0:.2f}s)", flush=True)

        # Save state for continue calls
        self._interactive_state = {
            "x_t": x_t,
            "obs": obs_proc,
            "prefix_tokens": prefix_tokens,
            "prefix_mask": prefix_mask,
            "kv_cache": kv_cache,
            "t_starts": t_starts,
            "dt_schedule": dt_schedule,
            "steps_per_block": steps_per_block,
            "next_block_idx": 1,
        }

        # Extract block 0 actions and unnormalize from model's z-score space to real joint space
        bs = self.block_size
        block_actions = self._unnormalize_actions(
            np.asarray(x_t[0, :bs, :8]).astype(np.float32)
        )
        return block_actions, 0

    def handle_infer_tactile_continue(self, msg: dict):
        """Continue the interactive tactile session after executing a block.

        Protocol:
          Client → Server: {"cmd": "infer_tactile_continue",
                            "tactile_left": <npy>, "tactile_right": <npy>}
          Server → Client: {"block_actions": <npy>, "block_idx": k, "done": bool}
        """
        import time as _time
        import jax.numpy as jnp

        st = self._interactive_state
        if st is None:
            raise RuntimeError("No interactive session — call infer_tactile_start first.")

        block_idx = st["next_block_idx"]

        # Encode new tactile
        tac_left = bytes_to_arr(msg["tactile_left"]).astype(np.float32)
        tac_right = bytes_to_arr(msg["tactile_right"]).astype(np.float32)
        if tac_left.max() > 1.0:
            tac_left = tac_left / 255.0
        if tac_right.max() > 1.0:
            tac_right = tac_right / 255.0
        tac_left_j = jnp.array(tac_left)[None]
        tac_right_j = jnp.array(tac_right)[None]
        _t0 = _time.perf_counter()
        tactile_tokens = self._jit_encode_tac(tac_left_j, tac_right_j, train=False)
        tactile_tokens.block_until_ready()
        _t1 = _time.perf_counter()

        # Run next segment
        spb = st["steps_per_block"]
        seg_start = block_idx * spb
        seg_end = min((block_idx + 1) * spb, self.num_inference_steps)

        x_t = self._jit_denoise(
            st["x_t"],
            st["t_starts"][seg_start:seg_end],
            st["dt_schedule"][seg_start:seg_end],
            st["obs"], st["prefix_tokens"], st["prefix_mask"], st["kv_cache"],
            tactile_tokens,
        )
        x_t.block_until_ready()
        _t2 = _time.perf_counter()
        print(f"[Pi0DF] block{block_idx} encode_tac={_t1-_t0:.2f}s denoise={_t2-_t1:.2f}s total={_t2-_t0:.2f}s", flush=True)
        st["x_t"] = x_t

        # Extract this block's actions and unnormalize from model's z-score space to real joint space
        bs = self.block_size
        block_actions = self._unnormalize_actions(
            np.asarray(x_t[0, block_idx * bs:(block_idx + 1) * bs, :8]).astype(np.float32)
        )
        done = (block_idx >= self.num_blocks - 1)

        st["next_block_idx"] = block_idx + 1
        if done:
            self._interactive_state = None

        return block_actions, block_idx, done

    def run(self, socket_path: str):
        if os.path.exists(socket_path):
            os.unlink(socket_path)

        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(socket_path)
        srv.listen(1)
        print(f"[Pi0DFServer] Listening on {socket_path}", flush=True)

        conn, _ = srv.accept()
        print("[Pi0DFServer] Client connected", flush=True)
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
                        self._interactive_state = None
                        send_msg(conn, encode_msg({"status": "ok"}))
                    elif cmd == "infer":
                        action = self.handle_infer(msg)
                        send_msg(conn, encode_msg({"action": arr_to_bytes(action)}))
                    elif cmd == "infer_tactile_start":
                        block_actions, block_idx = self.handle_infer_tactile_start(msg)
                        send_msg(conn, encode_msg({
                            "block_actions": arr_to_bytes(block_actions),
                            "block_idx": block_idx,
                        }))
                    elif cmd == "infer_tactile_continue":
                        block_actions, block_idx, done = self.handle_infer_tactile_continue(msg)
                        send_msg(conn, encode_msg({
                            "block_actions": arr_to_bytes(block_actions),
                            "block_idx": block_idx,
                            "done": done,
                        }))
                    elif cmd == "close":
                        send_msg(conn, encode_msg({"status": "ok"}))
                        break
                    else:
                        send_msg(conn, encode_msg({"status": "error",
                                                   "msg": f"Unknown cmd: {cmd}"}))
                except Exception as e:
                    err = traceback.format_exc()
                    print(f"[Pi0DFServer] Error cmd={cmd}: {err}", flush=True)
                    send_msg(conn, encode_msg({"status": "error",
                                               "msg": f"{e}\n---TRACEBACK---\n{err}"}))
        finally:
            conn.close()
            srv.close()
            if os.path.exists(socket_path):
                os.unlink(socket_path)
            print("[Pi0DFServer] Shutdown", flush=True)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--socket", required=True)
    Pi0DFInferenceServer().run(p.parse_args().socket)
