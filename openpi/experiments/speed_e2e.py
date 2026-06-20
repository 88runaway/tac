"""Pi0.5 (openpi) E2E Inference Speed Benchmark.

Measures single-step inference latency of the Pi0.5 model, broken down into:
  - Prefix encoding (SigLIP vision + language embedding + KV cache)
  - Denoising loop (flow-matching Euler steps via action expert)
  - End-to-end total

Inputs are kept consistent with Mantis2 benchmark:
  - caption: "pick up the red cube and place it on the blue plate"
  - image resolution: 224x224 (Pi0.5 native resolution)
  - n_images: 3 (base + left_wrist + right_wrist, Pi0.5 standard)
  - state_dim / action_dim: 14 (RoboTwin dual-arm, padded to model's max_dim=32)
  - num_steps: 10 (default flow-matching denoising steps)
"""

import argparse
import os
import sys
import warnings

os.environ.setdefault("TRITON_CACHE_DIR", "/data1/zjb/cache/triton")

import numpy as np
import safetensors.torch
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class CudaTimer:
    """Context-manager for accurate GPU timing via CUDA events."""

    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.start_event.record()
        return self

    def __exit__(self, *args):
        self.end_event.record()
        torch.cuda.synchronize()

    @property
    def elapsed_ms(self) -> float:
        return self.start_event.elapsed_time(self.end_event)


class _SigLIPTimer:
    """Hook ``paligemma_with_expert.embed_image`` to time ViT calls."""

    def __init__(self, model: PI0Pytorch):
        self._target = model.paligemma_with_expert
        self._orig_fn = self._target.embed_image
        self._events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

    def _wrapped(self, *args, **kwargs):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        out = self._orig_fn(*args, **kwargs)
        e.record()
        self._events.append((s, e))
        return out

    def __enter__(self):
        self._target.embed_image = self._wrapped
        return self

    def __exit__(self, *exc):
        self._target.embed_image = self._orig_fn

    @property
    def elapsed_ms(self) -> float:
        if not self._events:
            return 0.0
        torch.cuda.synchronize()
        return sum(s.elapsed_time(e) for s, e in self._events)


class SimpleObservation:
    """Minimal observation object accepted by PI0Pytorch._preprocess_observation."""

    def __init__(self, images, image_masks, state, tokenized_prompt, tokenized_prompt_mask):
        self.images = images
        self.image_masks = image_masks
        self.state = state
        self.tokenized_prompt = tokenized_prompt
        self.tokenized_prompt_mask = tokenized_prompt_mask
        self.token_ar_mask = None
        self.token_loss_mask = None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: str = "cuda", compile_mode: str | None = None):
    """Load PI0Pytorch from a safetensors checkpoint.

    Args:
        checkpoint_path: Directory containing model.safetensors (LeRobot format)
                         or direct path to a .safetensors file.
        device: CUDA device string.
        compile_mode: torch.compile mode. None to disable (faster startup for benchmarking).
    """
    config = Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=50,
        pytorch_compile_mode=compile_mode,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        model = PI0Pytorch(config)

    weight_path = checkpoint_path
    if os.path.isdir(checkpoint_path):
        weight_path = os.path.join(checkpoint_path, "model.safetensors")

    safetensors.torch.load_model(model, weight_path)
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------

def prepare_inputs(
    caption: str,
    device: str = "cuda",
    image_size: int = 224,
    n_images: int = 3,
    state_dim: int = 14,
    action_dim: int = 32,
    max_token_len: int = 200,
):
    """Build dummy Observation tensors matching Pi0.5 expected format.

    Pi0.5 uses PaligemmaTokenizer with discrete_state_input=True:
      prompt format: "Task: <caption>, State: <discretized_state>;\\nAction: "

    To keep the benchmark consistent across models, we use the same caption
    and construct the tokenized prompt to match the expected length.
    """
    image_keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")

    images = {}
    image_masks = {}
    for i, key in enumerate(image_keys):
        img = torch.randn(1, 3, image_size, image_size, device=device, dtype=torch.float32)
        img = img.clamp(-1, 1)
        images[key] = img
        image_masks[key] = torch.tensor([i < n_images], dtype=torch.bool, device=device)

    state = torch.randn(1, action_dim, device=device, dtype=torch.float32).clamp(-1, 1)

    # Build tokenized prompt matching Pi0.5 format:
    # "Task: <caption>, State: <discretized_state>;\nAction: "
    state_np = state[0, :state_dim].cpu().numpy()
    discretized_state = np.digitize(state_np, bins=np.linspace(-1, 1, 257)[:-1]) - 1
    state_str = " ".join(map(str, discretized_state))
    full_prompt = f"Task: {caption}, State: {state_str};\nAction: "

    # Use sentencepiece tokenizer from openpi
    try:
        from openpi.models.tokenizer import PaligemmaTokenizer
        tokenizer = PaligemmaTokenizer(max_len=max_token_len)
        tokens_np, mask_np = tokenizer.tokenize(caption, state_np)
        tokenized_prompt = torch.from_numpy(tokens_np).unsqueeze(0).to(device)
        tokenized_prompt_mask = torch.from_numpy(mask_np).unsqueeze(0).to(device)
    except Exception:
        # Fallback: synthetic tokens with realistic length
        prompt_len = min(len(full_prompt.split()) * 3, max_token_len)
        tokens = torch.randint(1, 250000, (1, max_token_len), device=device, dtype=torch.int32)
        mask = torch.zeros(1, max_token_len, dtype=torch.bool, device=device)
        mask[0, :prompt_len] = True
        tokenized_prompt = tokens
        tokenized_prompt_mask = mask

    obs = SimpleObservation(
        images=images,
        image_masks=image_masks,
        state=state,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
    )

    return obs


# ---------------------------------------------------------------------------
# Benchmark routines
# ---------------------------------------------------------------------------

@torch.no_grad()
def benchmark_modules(
    model: PI0Pytorch,
    observation: SimpleObservation,
    warmup: int,
    repeats: int,
    num_steps: int,
    device: str,
):
    """Time each stage of Pi0.5 inference separately.

    Breakdown:
      - Vision encoder (SigLIP): image embedding calls inside embed_prefix
      - Prefix KV cache: full prefix forward (images + language) → past_key_values
      - Denoising loop: flow-matching Euler steps via action expert
      - End-to-end (no tokenize): total from observation to actions
    """
    total_runs = warmup + repeats
    vision_times, prefix_times, denoise_times, total_times = [], [], [], []

    for run_idx in range(total_runs):
        torch.cuda.synchronize()

        t_total = CudaTimer()
        t_prefix = CudaTimer()
        t_denoise = CudaTimer()
        vt = _SigLIPTimer(model)

        with t_total:
            # --- Preprocessing ---
            images, img_masks, lang_tokens, lang_masks, state = \
                model._preprocess_observation(observation, train=False)

            # --- Prefix encoding (SigLIP + language + KV cache) ---
            with t_prefix, vt:
                prefix_embs, prefix_pad_masks, prefix_att_masks = \
                    model.embed_prefix(images, img_masks, lang_tokens, lang_masks)

                from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
                prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
                prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
                prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)

                model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

                _, past_key_values = model.paligemma_with_expert.forward(
                    attention_mask=prefix_att_2d_masks_4d,
                    position_ids=prefix_position_ids,
                    past_key_values=None,
                    inputs_embeds=[prefix_embs, None],
                    use_cache=True,
                )

            # --- Denoising loop (flow-matching Euler steps) ---
            with t_denoise:
                bsize = state.shape[0]
                actions_shape = (bsize, model.config.action_horizon, model.config.action_dim)
                noise = model.sample_noise(actions_shape, device)

                dt = -1.0 / num_steps
                dt_t = torch.tensor(dt, dtype=torch.float32, device=device)
                x_t = noise
                time_val = torch.tensor(1.0, dtype=torch.float32, device=device)
                while time_val >= -dt_t / 2:
                    expanded_time = time_val.expand(bsize)
                    v_t = model.denoise_step(
                        state,
                        prefix_pad_masks,
                        past_key_values,
                        x_t,
                        expanded_time,
                    )
                    x_t = x_t + dt_t * v_t
                    time_val += dt_t

        if run_idx >= warmup:
            vision_times.append(vt.elapsed_ms)
            prefix_total_ms = t_prefix.elapsed_ms
            prefix_times.append(prefix_total_ms)
            denoise_times.append(t_denoise.elapsed_ms)
            total_times.append(t_total.elapsed_ms)

    return {
        "Vision encoder (SigLIP)": vision_times,
        "Prefix KV cache": prefix_times,
        "Denoising loop": denoise_times,
        "End-to-end (no tokenize)": total_times,
    }


def print_results(name: str, times_ms: list):
    arr = np.array(times_ms)
    print(
        f"  {name:<30s}  "
        f"mean={arr.mean():8.2f} ms  "
        f"std={arr.std():6.2f} ms  "
        f"min={arr.min():8.2f} ms  "
        f"max={arr.max():8.2f} ms  "
        f"({1000.0 / arr.mean():6.1f} Hz)"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pi0.5 E2E Benchmark")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint dir (containing model.safetensors)")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--image_size", type=int, default=224,
                        help="Pi0.5 native resolution is 224x224")
    parser.add_argument("--n_images", type=int, default=3,
                        help="Number of camera views (Pi0.5 uses 3: base + 2 wrist)")
    parser.add_argument("--num_steps", type=int, default=10,
                        help="Flow-matching denoising steps (default 10)")
    parser.add_argument("--state_dim", type=int, default=14,
                        help="Robot state dimension (RoboTwin=14, padded to 32)")
    parser.add_argument("--compile", type=str, default=None,
                        choices=["default", "reduce-overhead", "max-autotune",
                                 "max-autotune-no-cudagraphs"],
                        help="torch.compile mode (None = disabled for faster startup)")
    args = parser.parse_args()

    device = "cuda"

    print(f"Loading Pi0.5 model from {args.checkpoint} ...")
    model = load_model(args.checkpoint, device, compile_mode=args.compile)

    caption = "pick up the red cube and place it on the blue plate"
    print(f"Preparing dummy inputs ...")
    observation = prepare_inputs(
        caption=caption,
        device=device,
        image_size=args.image_size,
        n_images=args.n_images,
        state_dim=args.state_dim,
        action_dim=model.config.action_dim,
        max_token_len=model.config.max_token_len,
    )

    print(f"\nBenchmark config: warmup={args.warmup}, repeats={args.repeats}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Image size: {args.image_size}x{args.image_size}, n_images: {args.n_images}")
    print(f"Denoising steps: {args.num_steps}")
    print(f"Action horizon: {model.config.action_horizon}, action_dim: {model.config.action_dim}")
    print(f"Caption: \"{caption}\"")
    print(f"torch.compile: {args.compile or 'disabled'}")

    # ---- Standard mode ----
    print(f"\n{'=' * 80}")
    print("  Pi0.5 STANDARD MODE")
    print(f"{'=' * 80}")
    results = benchmark_modules(model, observation, args.warmup, args.repeats, args.num_steps, device)
    for name, times in results.items():
        print_results(name, times)

    # ---- torch.compile mode (if not already enabled) ----
    if args.compile is None:
        print(f"\n  Building torch.compile (max-autotune) version ...")
        model_compiled = load_model(args.checkpoint, device, compile_mode="max-autotune")

        # Warmup the compiled version (first runs trigger compilation)
        print(f"  Warming up compiled model (this may take a while) ...")
        compiled_results = benchmark_modules(
            model_compiled, observation, max(args.warmup, 5), args.repeats, args.num_steps, device
        )

        print(f"\n{'=' * 80}")
        print("  Pi0.5 torch.compile MODE (max-autotune)")
        print(f"{'=' * 80}")
        for name, times in compiled_results.items():
            print_results(name, times)

        # ---- Comparison ----
        print(f"\n{'=' * 80}")
        print("  COMPARISON (standard vs torch.compile)")
        print(f"{'=' * 80}")
        for name in results:
            orig_mean = np.mean(results[name])
            comp_mean = np.mean(compiled_results[name])
            speedup = orig_mean / comp_mean if comp_mean > 0 else float("inf")
            saved = orig_mean - comp_mean
            marker = "*" if speedup > 1.05 else " "
            print(
                f" {marker} {name:<30s}  "
                f"standard={orig_mean:8.2f} ms  compiled={comp_mean:8.2f} ms  "
                f"saved={saved:+8.2f} ms  speedup={speedup:.2f}x"
            )


if __name__ == "__main__":
    main()
