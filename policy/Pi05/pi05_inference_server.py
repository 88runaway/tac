"""
Pi0.5 Inference Server — runs under the lerobot Python 3.12 environment.

Started as a subprocess by deploy_policy.py (UniVTAC Python 3.10).
Communicates via a Unix domain socket: receives observation dicts,
returns actions as numpy arrays.

Protocol (length-prefixed msgpack over Unix socket):
  Client → Server: {"cmd": "init",   "args": {...}}
  Server → Client: {"status": "ok"}  |  {"status": "error", "msg": "..."}

  Client → Server: {"cmd": "reset"}
  Server → Client: {"status": "ok"}

  Client → Server: {"cmd": "infer",
                    "images": {"base_0_rgb": <CHW float32 bytes>, ...},
                    "state":  <8-dim float32 bytes>,
                    "task":   "instruction string"}
  Server → Client: {"action": <8-dim float32 bytes>}

  Client → Server: {"cmd": "close"}
  Server → Client: {"status": "ok"}

RTC 模式说明
  当 init_args 中 rtc_enabled=True 时，使用 predict_action_chunk + 手动 ActionQueue
  代替 select_action，每次 chunk 耗尽时把 prev_chunk_left_over（上一 chunk 未执行部分）
  传给 flow matching 去噪过程，引导新 chunk 的开头与旧 chunk 末尾保持连续。

  inference_delay=0：当前为同步推理（机器人等待推理完成），故延迟为 0 步。
"""

import sys
import os
import json
import struct
import socket
import traceback
import numpy as np
from pathlib import Path
import io

LEROBOT_SRC = Path(__file__).parent.parent.parent / "lerobot" / "src"
sys.path.insert(0, str(LEROBOT_SRC))

import torch
from torchvision import transforms

from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.factory import make_pre_post_processors


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


class InferenceServer:
    def __init__(self):
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None
        self.device = None
        self.image_size = None
        self.camera_type = "head"

        # RTC configuration
        self.rtc_enabled = False
        self.rtc_execution_horizon = 25

        # Action buffer (used in both standard and RTC modes for chunking)
        # _raw_chunk  : full chunk in model's normalized space, (chunk_size, action_dim)
        #               kept for prev_chunk_left_over computation in RTC mode
        # _proc_chunk : first n_action_steps postprocessed actions, (n_action_steps, action_dim)
        #               popped one-by-one for robot execution
        # _chunk_idx  : next step to return from _proc_chunk
        self._raw_chunk: torch.Tensor | None = None
        self._proc_chunk: torch.Tensor | None = None
        self._chunk_idx: int = 0

    # ──────────────────────────────────────────────────────────────────────────
    # Command handlers
    # ──────────────────────────────────────────────────────────────────────────

    def handle_init(self, args: dict):
        ckpt_dir = Path(args["ckpt_dir"])
        self.camera_type = args.get("camera_type", "head")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        is_lora = (ckpt_dir / "adapter_config.json").exists()

        if is_lora:
            import json as _json
            from peft import PeftModel

            with open(ckpt_dir / "config.json") as f:
                cfg_dict = _json.load(f)
            pretrained_path = cfg_dict.get("pretrained_path") or cfg_dict.get("pretrained_model_name_or_path")
            assert pretrained_path, (
                "Cannot find pretrained_path in config.json. "
                "Please set it manually or merge the LoRA weights first."
            )
            print(f"[Server] LoRA checkpoint detected. Loading base model from: {pretrained_path}", flush=True)
            self.policy = PI05Policy.from_pretrained(pretrained_path)
            print(f"[Server] Merging LoRA adapter from: {ckpt_dir}", flush=True)
            self.policy = PeftModel.from_pretrained(self.policy, str(ckpt_dir))
            self.policy = self.policy.merge_and_unload()

            from lerobot.configs.policies import PreTrainedConfig
            finetuned_config = PreTrainedConfig.from_pretrained(str(ckpt_dir))
            self.policy.config.input_features = finetuned_config.input_features
            self.policy.config.output_features = finetuned_config.output_features
            print(f"[Server] Overrode features from fine-tuned config: "
                  f"action_dim={finetuned_config.output_features['action'].shape[0]}", flush=True)
        else:
            self.policy = PI05Policy.from_pretrained(str(ckpt_dir))

        self.policy.eval()
        self.policy.to(self.device)

        # n_action_steps override
        n_action_steps = args.get("n_action_steps")
        if n_action_steps is not None:
            self.policy.config.n_action_steps = int(n_action_steps)

        # num_inference_steps override (flow matching denoising steps)
        num_inference_steps = args.get("num_inference_steps")
        if num_inference_steps is not None:
            self.policy.config.num_inference_steps = int(num_inference_steps)
            print(f"[Server] num_inference_steps overridden to: {self.policy.config.num_inference_steps}", flush=True)
        else:
            print(f"[Server] num_inference_steps: {self.policy.config.num_inference_steps} (from checkpoint)", flush=True)

        # RTC configuration
        self.rtc_enabled = bool(args.get("rtc_enabled", False))
        self.rtc_execution_horizon = int(args.get("rtc_execution_horizon", 25))

        if self.rtc_enabled:
            from lerobot.policies.rtc.configuration_rtc import RTCConfig
            self.policy.config.rtc_config = RTCConfig(
                enabled=True,
                execution_horizon=self.rtc_execution_horizon,
            )
            self.policy.init_rtc_processor()
            print(
                f"[Server] RTC enabled: execution_horizon={self.rtc_execution_horizon}, "
                f"chunk_size={self.policy.config.chunk_size}, "
                f"n_action_steps={self.policy.config.n_action_steps}",
                flush=True,
            )
            overlap = self.policy.config.chunk_size - self.policy.config.n_action_steps
            if overlap <= 0:
                print(
                    f"[Server] WARNING: n_action_steps ({self.policy.config.n_action_steps}) == "
                    f"chunk_size ({self.policy.config.chunk_size}). RTC left_over will always be empty. "
                    "Consider reducing n_action_steps (e.g. to chunk_size/2) for effective RTC.",
                    flush=True,
                )

        self.image_size = tuple(self.policy.config.image_resolution)

        preprocessor_overrides = {
            "device_processor": {"device": str(self.device)},
        }
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=ckpt_dir,
            preprocessor_overrides=preprocessor_overrides,
        )

        mode = "RTC" if self.rtc_enabled else "standard"
        print(
            f"[Server] Policy loaded from {ckpt_dir}\n"
            f"[Server] mode={mode}, n_action_steps={self.policy.config.n_action_steps}, "
            f"chunk_size={self.policy.config.chunk_size}, device={self.device}",
            flush=True,
        )

    def handle_reset(self):
        """Reset policy state between episodes."""
        if self.policy is not None:
            self.policy.reset()
        self._raw_chunk = None
        self._proc_chunk = None
        self._chunk_idx = 0

    def handle_infer(self, msg: dict) -> np.ndarray:
        # ── Decode observation ───────────────────────────────────────────────
        head_img = bytes_to_arr(msg["images"]["base_0_rgb"])  # CHW float32
        state = bytes_to_arr(msg["state"])                    # (8,) float32
        instruction = msg["task"]

        head_tensor = torch.from_numpy(head_img).unsqueeze(0).to(self.device)
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)

        batch = {
            "observation.images.base_0_rgb": head_tensor,
            "observation.state": state_tensor,
            "task": [instruction],
        }

        if "left_wrist_0_rgb" in msg.get("images", {}):
            wrist_img = bytes_to_arr(msg["images"]["left_wrist_0_rgb"])
            batch["observation.images.left_wrist_0_rgb"] = (
                torch.from_numpy(wrist_img).unsqueeze(0).to(self.device)
            )

        processed = self.preprocessor(batch)

        if self.rtc_enabled:
            return self._infer_rtc(processed)
        else:
            return self._infer_standard(processed)

    # ──────────────────────────────────────────────────────────────────────────
    # Inference implementations
    # ──────────────────────────────────────────────────────────────────────────

    def _infer_standard(self, processed) -> np.ndarray:
        """Standard action-queue chunking via select_action."""
        with torch.inference_mode():
            action = self.policy.select_action(processed)
        action = self.postprocessor(action)
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    def _infer_rtc(self, processed) -> np.ndarray:
        """RTC-guided chunking: refill buffer when exhausted, pass prev_chunk_left_over."""
        if self._proc_chunk is None or self._chunk_idx >= len(self._proc_chunk):
            self._refill_rtc_buffer(processed)

        action = self._proc_chunk[self._chunk_idx]  # (action_dim,)
        self._chunk_idx += 1
        return action.cpu().numpy().astype(np.float32)

    def _refill_rtc_buffer(self, processed):
        """Run predict_action_chunk with RTC guidance and populate action buffers.

        prev_chunk_left_over:
            The unexecuted tail of the previous chunk (in model-normalized space).
            Shape: (chunk_size - n_action_steps, action_dim), or None on first call.
            Passed to the flow-matching denoiser so that the new chunk's leading
            steps are guided to be continuous with the previous chunk's tail.

        inference_delay=0:
            We run in synchronous mode — the robot waits while the model infers.
            No steps are executed during inference, so delay is always 0.
        """
        n_steps = self.policy.config.n_action_steps
        chunk_size = self.policy.config.chunk_size

        # Build left_over from the unexecuted tail of the previous chunk
        left_over = None
        if self._raw_chunk is not None:
            tail = self._raw_chunk[n_steps:]          # (chunk_size - n_steps, action_dim)
            if tail.shape[0] > 0:
                left_over = tail.unsqueeze(0).to(self.device)   # (1, remaining, action_dim)

        # NOTE: do NOT use torch.inference_mode() here.
        # predict_action_chunk uses @torch.no_grad(), but RTC internally calls
        # torch.enable_grad() for autograd-based correction — which works under
        # no_grad but would be silently disabled under inference_mode.
        raw_actions = self.policy.predict_action_chunk(
            processed,
            prev_chunk_left_over=left_over,
            inference_delay=0,
            execution_horizon=self.rtc_execution_horizon,
        )  # (1, chunk_size, action_dim)

        raw_actions = raw_actions.squeeze(0).detach()   # (chunk_size, action_dim)

        # Keep full chunk for next call's left_over computation
        self._raw_chunk = raw_actions.cpu()

        # Postprocess first n_steps for execution (one step at a time to match
        # the postprocessor's expected (1, action_dim) input shape)
        exec_steps = min(n_steps, chunk_size)
        proc_steps = []
        for i in range(exec_steps):
            step = raw_actions[i : i + 1]          # (1, action_dim)
            step_proc = self.postprocessor(step)    # (1, action_dim) unnormalized
            proc_steps.append(step_proc.squeeze(0).detach().cpu())

        self._proc_chunk = torch.stack(proc_steps, dim=0)   # (exec_steps, action_dim)
        self._chunk_idx = 0

    # ──────────────────────────────────────────────────────────────────────────
    # Socket server
    # ──────────────────────────────────────────────────────────────────────────

    def run(self, socket_path: str):
        if os.path.exists(socket_path):
            os.unlink(socket_path)

        server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server_sock.bind(socket_path)
        server_sock.listen(1)
        print(f"[Server] Listening on {socket_path}", flush=True)

        conn, _ = server_sock.accept()
        print("[Server] Client connected", flush=True)

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
                    print(f"[Server] Error handling cmd={cmd}: {err}", flush=True)
                    send_msg(conn, encode_msg({"status": "error", "msg": f"{str(e)}\n---TRACEBACK---\n{err}"}))

        finally:
            conn.close()
            server_sock.close()
            if os.path.exists(socket_path):
                os.unlink(socket_path)
            print("[Server] Shutdown", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True, help="Unix socket path")
    args = parser.parse_args()
    InferenceServer().run(args.socket)
