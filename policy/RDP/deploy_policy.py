import sys
import os
import json
import pickle
from pathlib import Path
from collections import deque

sys.path.append(str(Path(__file__).parent.parent))

from .._base_policy import BasePolicy

import numpy as np
import torch
import dill
from torchvision import transforms


# ---------------------------------------------------------------------------
# Lightweight PCA reducer (mirrors reactive_diffusion_policy PCAEmbedding.pca_reduction)
# ---------------------------------------------------------------------------

class _PCAReducer:
    """Load pre-trained PCA matrices and apply online reduction."""
    def __init__(self, pca_dir: str):
        pca_path = Path(pca_dir)
        self.W    = np.load(pca_path / "pca_transform_matrix.npy")  # (D, K)
        self.mean = np.load(pca_path / "pca_mean_matrix.npy")       # (D,)
        self.n_components = self.W.shape[1]

    def reduce(self, flat_offset: np.ndarray) -> np.ndarray:
        """flat_offset: (D,) -> (K,) float32"""
        return ((flat_offset - self.mean) @ self.W).astype(np.float32)


RDP_ROOT = Path(os.environ.get(
    "RDP_ROOT", "/data1/zjb/reactive_diffusion_policy"))


class Policy(BasePolicy):
    """
    Adapter that loads a trained RDP (Diffusion Policy / Latent Diffusion Policy)
    checkpoint and wraps it as a UniVTAC BasePolicy for benchmark evaluation.
    """

    def __init__(self, args):
        self.task_name = args["task_name"]
        self.save_image = args.get("save_image", False)

        ckpt_config = os.environ.get("CKPT_CONFIG", "univtac")
        ckpt_root = os.environ.get("CKPT_ROOT", None)
        env_ckpt_dir = os.environ.get("CKPT_DIR", None)
        if "ckpt_dir" in args and args["ckpt_dir"]:
            ckpt_dir = Path(args["ckpt_dir"])
        elif env_ckpt_dir:
            ckpt_dir = Path(env_ckpt_dir)
        elif ckpt_root:
            ckpt_dir = Path(ckpt_root) / args["task_name"] / ckpt_config
        else:
            ckpt_dir = RDP_ROOT / "ckpt" / args["task_name"] / ckpt_config

        self.device = torch.device(args.get("device", "cuda:0"))

        # Tactile mode: auto-detected from checkpoint cfg, or overridden by deploy config.
        # Detection: if obs shape_meta contains "tac_left_emb" -> marker_emb, else rgb.
        self._tactile_mode_override = args.get("tactile_mode", None)
        self.tactile_mode = self._tactile_mode_override or "rgb"  # finalized after ckpt load
        self.pca_reducer: _PCAReducer = None

        # Determine policy type: ldp (latent diffusion) vs dp (diffusion)
        self.policy_type = args.get("policy_type", "ldp")
        self.n_obs_steps = args.get("n_obs_steps", 2)
        self.dataset_obs_temporal_downsample_ratio = args.get(
            "dataset_obs_temporal_downsample_ratio", 1)

        with open(Path(__file__).parent.parent / "task_settings.json", "r") as f:
            task_settings = json.load(f)
        assert self.task_name in task_settings, \
            f"Task '{self.task_name}' not found in task_settings.json"
        self.camera_type = task_settings[self.task_name].get("camera_type", "head")

        # Load checkpoint and reconstruct policy
        ckpt_path = self._find_checkpoint(ckpt_dir)
        print(f"Loading RDP checkpoint from {ckpt_path}")
        payload = torch.load(ckpt_path, pickle_module=dill,
                             map_location=self.device, weights_only=False)
        cfg = payload["cfg"]

        # Instantiate workspace to reconstruct the model
        sys.path.insert(0, str(RDP_ROOT))
        import hydra
        from omegaconf import OmegaConf, open_dict
        OmegaConf.register_new_resolver("eval", eval, replace=True)

        # Override training.device in cfg to match the current inference device.
        # The saved cfg may contain e.g. "cuda:4" from training, which could be
        # invalid or different from the current CUDA_VISIBLE_DEVICES mapping.
        with open_dict(cfg):
            cfg.training.device = str(self.device)

        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg, output_dir=str(ckpt_dir))
        workspace.load_payload(payload, exclude_keys=set(), include_keys=set())

        # Extract policy model
        if hasattr(workspace, "ema_model") and workspace.ema_model is not None:
            self.model = workspace.ema_model
        else:
            self.model = workspace.model
        self.model.eval()
        self.model.to(self.device)

        # Load normalizer
        normalizer_path = ckpt_dir / "normalizer.pkl"
        if normalizer_path.exists():
            with open(normalizer_path, "rb") as f:
                normalizer = pickle.load(f)
            # normalizer.pkl is saved with CPU tensors; move to inference device
            # so that normalizer._normalize() doesn't pull obs tensors back to CPU.
            if hasattr(normalizer, "to"):
                normalizer = normalizer.to(self.device)
            self.model.set_normalizer(normalizer)

        # Auto-detect tactile_mode from checkpoint shape_meta if not explicitly set
        if self._tactile_mode_override is None:
            try:
                obs_keys = set(cfg.task.shape_meta.obs.keys())
                if "tac_left_emb" in obs_keys:
                    self.tactile_mode = "marker_emb"
                else:
                    self.tactile_mode = "rgb"
                print(f"[RDP] Auto-detected tactile_mode: {self.tactile_mode}")
            except Exception:
                self.tactile_mode = "rgb"

        if self.tactile_mode == "marker_emb":
            pca_dir = args.get("pca_dir", None) or os.environ.get("PCA_DIR", None)
            if pca_dir is None:
                # Auto-detect: RDP_ROOT/data/PCA_Transform_UniVTAC_<task_name>
                _auto = RDP_ROOT / "data" / f"PCA_Transform_UniVTAC_{self.task_name}"
                if _auto.exists():
                    pca_dir = str(_auto)
            if pca_dir is None:
                raise ValueError(
                    "tactile_mode=marker_emb requires 'pca_dir' in deploy config, "
                    "PCA_DIR env var, or RDP_ROOT/data/PCA_Transform_UniVTAC_<task_name>/"
                )
            self.pca_reducer = _PCAReducer(pca_dir)
            print(f"[RDP] Tactile: marker_emb, PCA_K={self.pca_reducer.n_components}, pca_dir={pca_dir}")
        else:
            print(f"[RDP] Tactile: rgb")

        # Check if this is LDP with RNN decoder
        self.is_ldp = hasattr(self.model, "at")
        self.use_rnn_decoder = (self.is_ldp and
                                hasattr(self.model.at, "use_rnn_decoder") and
                                self.model.at.use_rnn_decoder)

        # Observation history buffer for n_obs_steps
        self.obs_history = deque(maxlen=self.n_obs_steps)

        print(f"RDP policy loaded: type={'LDP' if self.is_ldp else 'DP'}, "
              f"n_obs_steps={self.n_obs_steps}, rnn_decoder={self.use_rnn_decoder}")

    def _find_checkpoint(self, ckpt_dir: Path) -> Path:
        """Find the best or latest checkpoint in the directory."""
        ckpt_dir = Path(ckpt_dir)
        if ckpt_dir.is_file() and ckpt_dir.suffix == ".ckpt":
            return ckpt_dir
        checkpoints_dir = ckpt_dir / "checkpoints"

        # Try checkpoints/ subdirectory first
        if checkpoints_dir.exists():
            # Prefer latest.ckpt
            latest = checkpoints_dir / "latest.ckpt"
            if latest.exists():
                return latest
            # Fall back to any .ckpt file
            ckpts = sorted(checkpoints_dir.glob("*.ckpt"))
            if ckpts:
                return ckpts[-1]

        # Try direct .ckpt files in ckpt_dir
        ckpts = sorted(ckpt_dir.glob("*.ckpt"))
        if ckpts:
            return ckpts[-1]

        raise FileNotFoundError(
            f"No checkpoint found in {ckpt_dir} or {checkpoints_dir}")

    def encode_obs(self, observation):
        """
        Convert UniVTAC observation dict to RDP obs_dict format.

        Input (UniVTAC):
            observation = {
                "observation": {"head": {"rgb": Tensor(H,W,3)}},
                "tactile": {
                    "left_tactile":  {"rgb_marker": Tensor(H,W,3),
                                      "marker":     Tensor(2, N, 2)},
                    "right_tactile": {"rgb_marker": Tensor(H,W,3),
                                      "marker":     Tensor(2, N, 2)}
                },
                "embodiment": {"joint": Tensor(N,)}
            }

        Output (RDP) — tactile_mode "rgb":
            obs_dict_np = {
                "cam_high":  (C, H, W) float32 [0,1]
                "tac_left":  (C, H, W) float32 [0,1]
                "tac_right": (C, H, W) float32 [0,1]
                "qpos":      (8,) float32
            }

        Output (RDP) — tactile_mode "marker_emb":
            obs_dict_np = {
                "cam_high":      (C, H, W) float32 [0,1]
                "tac_left_emb":  (K,) float32   K = PCA n_components
                "tac_right_emb": (K,) float32
                "qpos":          (8,) float32
            }
        """
        def img_to_chw(img_tensor, target_h=240, target_w=320):
            """HWC uint8 tensor -> CHW float32 numpy [0,1]"""
            img = img_tensor.cpu().numpy().astype(np.float32) / 255.0
            img = np.moveaxis(img, -1, 0)  # HWC -> CHW
            c, h, w = img.shape
            if h != target_h or w != target_w:
                img_t = torch.from_numpy(img).unsqueeze(0)
                img_t = torch.nn.functional.interpolate(
                    img_t, size=(target_h, target_w),
                    mode="bilinear", align_corners=False)
                img = img_t.squeeze(0).numpy()
            return img

        def marker_to_emb(marker_tensor):
            """
            marker_tensor: Tensor(2, N, 2) — channel 0=initial, 1=current (uv, normalized)
            returns: (K,) float32 PCA embedding
            """
            m = marker_tensor.cpu().numpy()          # (2, N, 2)
            offset = m[1] - m[0]                     # (N, 2) current - initial
            flat = offset.reshape(-1).astype(np.float32)  # (N*2,)
            return self.pca_reducer.reduce(flat)      # (K,)

        cam_high = img_to_chw(observation["observation"][self.camera_type]["rgb"])
        qpos = observation["embodiment"]["joint"][:8].cpu().numpy().astype(np.float32)

        if self.tactile_mode == "rgb":
            tac_left  = img_to_chw(observation["tactile"]["left_tactile"]["rgb_marker"])
            tac_right = img_to_chw(observation["tactile"]["right_tactile"]["rgb_marker"])
            obs_dict_np = {
                "cam_high":  cam_high,
                "tac_left":  tac_left,
                "tac_right": tac_right,
                "qpos":      qpos,
            }
        else:  # marker_emb
            tac_left_emb  = marker_to_emb(observation["tactile"]["left_tactile"]["marker"])
            tac_right_emb = marker_to_emb(observation["tactile"]["right_tactile"]["marker"])
            obs_dict_np = {
                "cam_high":      cam_high,
                "tac_left_emb":  tac_left_emb,
                "tac_right_emb": tac_right_emb,
                "qpos":          qpos,
            }

        return obs_dict_np

    def _build_obs_batch(self, obs_dict_np):
        """
        Build a batched observation dict with temporal history for policy input.
        Stacks the obs history along a time dimension: (B=1, T=n_obs_steps, ...)
        """
        self.obs_history.append(obs_dict_np)

        # Pad history if not enough steps yet (repeat first obs)
        while len(self.obs_history) < self.n_obs_steps:
            self.obs_history.appendleft(self.obs_history[0])

        obs_batch = {}
        for key in obs_dict_np.keys():
            # Stack along time: list of (C,H,W) or (D,) -> (T, ...)
            stacked = np.stack([h[key] for h in self.obs_history], axis=0)
            # Add batch dim: (1, T, ...)
            obs_batch[key] = torch.from_numpy(stacked).unsqueeze(0).to(
                device=self.device, dtype=torch.float32)

        return obs_batch

    def _tactile_keys(self):
        """Return the tactile observation keys for the current tactile mode."""
        if self.tactile_mode == "rgb":
            return ["tac_left", "tac_right"]
        else:
            return ["tac_left_emb", "tac_right_emb"]

    def _encode_tactile(self, obs_dict_np):
        """Extract only tactile entries from an encoded obs dict."""
        return {k: obs_dict_np[k] for k in self._tactile_keys() if k in obs_dict_np}

    def _build_extended_obs_history(self, tactile_list, min_len=None):
        """
        Build extended_obs tensor dict from a list of per-step tactile dicts.

        tactile_list: list of length T, each entry is {key: np.ndarray(...)}
        min_len: if provided, pad the front with the first frame so that the
                 resulting time axis has at least min_len steps.  This ensures
                 DecoderRNN receives enough temporal context when the real
                 tactile history is shorter than extended_obs_last_step.
        Returns: {key: Tensor(1, T, ...)} on self.device
        """
        extended = {}
        for key in self._tactile_keys():
            frames = [t[key] for t in tactile_list if key in t]
            if min_len is not None:
                while len(frames) < min_len:
                    frames.insert(0, frames[0])
            stacked = np.stack(frames, axis=0)
            extended[key] = (torch.from_numpy(stacked)
                             .unsqueeze(0)  # (1, T, ...)
                             .to(device=self.device, dtype=torch.float32))
        return extended

    def _build_extended_obs(self, obs_dict_np):
        """Build single-frame extended_obs dict (used for non-reactive LDP path)."""
        extended = {}
        for key in self._tactile_keys():
            if key in obs_dict_np:
                val = obs_dict_np[key]
                extended[key] = (torch.from_numpy(val)
                                 .unsqueeze(0).unsqueeze(0)  # (1, 1, ...)
                                 .to(device=self.device, dtype=torch.float32))
        return extended

    def eval(self, task, observation):
        """
        Evaluate RDP policy on a UniVTAC task for one action chunk.

        For LDP with RNN decoder (reactive mode):
          1. Slow policy (predict_action, return_latent_action=True) runs once to
             obtain a latent_action from current visual + proprioceptive history.
          2. At every control step, fresh tactile observations are acquired.  The
             growing tactile history (length = step_index + 1) is fed to the fast
             policy (predict_from_latent_action), which re-decodes the full action
             chunk and returns action_pred[-1] — the most tactile-informed action.
          This matches the original RDP real_runner action_command_thread logic.

        For LDP without RNN decoder or plain DP:
          predict_action runs once and the returned chunk is executed step-by-step
          while the obs history is updated in between (unchanged behaviour).
        """
        obs_dict_np = self.encode_obs(observation)
        obs_batch = self._build_obs_batch(obs_dict_np)

        with torch.no_grad():
            # ------------------------------------------------------------------
            # Reactive fast-policy path (LDP + RNN decoder)
            # ------------------------------------------------------------------
            if self.is_ldp and self.use_rnn_decoder:
                # Step 1: slow policy — get latent action once
                slow_result = self.model.predict_action(
                    obs_batch,
                    dataset_obs_temporal_downsample_ratio=self.dataset_obs_temporal_downsample_ratio,
                    extended_obs_dict=None,
                    return_latent_action=True)
                # When return_latent_action=True, predict_action expands state_vq to
                # action_pred [B, original_horizon, latent_dim] and returns
                # result["action"] = action_pred[:, start:end] → [B, n_action_steps, latent_dim].
                # predict_from_latent_action expects state_vq of shape [B, latent_dim],
                # so we recover it by taking the first time step of action_pred.
                latent_action = slow_result["action_pred"][:, 0, :]  # [B, latent_dim]

                # Step 2: fast policy loop — re-decode at every control step
                # Collect the first tactile frame before we execute any action.
                tactile_history = [self._encode_tactile(obs_dict_np)]

                n_action_steps = self.model.n_action_steps
                # extended_obs_last_step mirrors the original RDP real_runner convention:
                # it starts at n_obs_steps * dataset_obs_temporal_downsample_ratio (= To),
                # matching the step indices appended to action_all in the slow-policy thread
                # (np.arange(To, To + n_action_steps)).
                # predict_from_latent_action slices action_pred[:, To-1 : To-1+n_action_steps],
                # so the decoder output must have at least To steps; starting at To guarantees
                # the temporal_cond fed to DecoderRNN has length >= To.
                To = self.n_obs_steps * self.dataset_obs_temporal_downsample_ratio
                for step_i in range(n_action_steps):
                    extended_obs_last_step = To + step_i  # matches original RDP step indexing
                    extended_obs_dict = self._build_extended_obs_history(
                        tactile_history, min_len=extended_obs_last_step)

                    result = self.model.predict_from_latent_action(
                        latent_action,
                        extended_obs_dict,
                        extended_obs_last_step,
                        self.dataset_obs_temporal_downsample_ratio)
                    # result['action']: (1, n_action_steps, action_dim)
                    # Take the last action in the decoded chunk — it is the most
                    # tactile-informed output given the accumulated history.
                    action_np = result["action"][0, -1].detach().cpu().numpy()
                    action = torch.from_numpy(action_np).to(task.device).float()

                    exec_succ, eval_succ = task.take_action(action, action_type="qpos")
                    if task.eval_success or task.check_early_stop():
                        break

                    if step_i < n_action_steps - 1:
                        # Acquire fresh observation for next step
                        observation = task._get_observations()
                        obs_dict_np = self.encode_obs(observation)
                        # Update slow-policy obs history (kept for next chunk's slow inference)
                        self.obs_history.append(obs_dict_np)
                        # Append latest tactile frame to the growing history
                        tactile_history.append(self._encode_tactile(obs_dict_np))

            # ------------------------------------------------------------------
            # Non-reactive path: LDP without RNN decoder, or plain DP
            # ------------------------------------------------------------------
            else:
                if self.is_ldp:
                    extended_obs = self._build_extended_obs(obs_dict_np)
                    action_dict = self.model.predict_action(
                        obs_batch,
                        dataset_obs_temporal_downsample_ratio=self.dataset_obs_temporal_downsample_ratio,
                        extended_obs_dict=extended_obs,
                        return_latent_action=False)
                else:
                    action_dict = self.model.predict_action(obs_batch)

                actions = action_dict["action"].squeeze(0).cpu().numpy()

                for i in range(actions.shape[0]):
                    action = torch.from_numpy(actions[i]).to(task.device).float()
                    exec_succ, eval_succ = task.take_action(action, action_type="qpos")
                    if task.eval_success or task.check_early_stop():
                        break
                    if i < actions.shape[0] - 1:
                        observation = task._get_observations()
                        obs_dict_np = self.encode_obs(observation)
                        self.obs_history.append(obs_dict_np)

    def reset(self):
        """Reset observation history buffer."""
        self.obs_history.clear()

    def close(self):
        """Cleanup."""
        if hasattr(self, "model") and self.model is not None:
            del self.model
            torch.cuda.empty_cache()
