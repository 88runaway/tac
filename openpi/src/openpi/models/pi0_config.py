import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0
    from openpi.models.pi0_df import Pi0DF


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    pytorch_compile_mode: str | None = "max-autotune"

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)


@dataclasses.dataclass(frozen=True)
class Pi0DFConfig(_model.BaseModelConfig):
    """Pi0 / Pi0.5 with block-wise Diffusion Forcing.

    Shares the exact same parameters/weights as :class:`Pi0Config` (so pi05_base checkpoints
    load unchanged); the only differences are how the flow-matching timestep is sampled during
    training and scheduled during inference. See ``openpi.models.pi0_df.Pi0DF``.
    """

    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    pytorch_compile_mode: str | None = "max-autotune"

    # ── Diffusion forcing options ──────────────────────────────────────────────
    # Number of contiguous blocks the action chunk is split into. Noise level is uniform
    # within each block. Must divide action_horizon.
    num_blocks: int = 5
    # Probability, per training sample, of using block-wise independent noise levels
    # (diffusion forcing). With probability (1 - mix_prob) the whole chunk shares a single
    # noise level (standard flow matching), which keeps the model close to pi05_base.
    mix_prob: float = 0.5
    # How the per-block noise levels are sampled during training when the DF branch fires:
    #   "independent" : each block gets an independent random noise level (original behaviour).
    #   "monotone"    : a single global progress scalar is mapped to a fixed monotonically
    #                   increasing (earlier blocks cleaner) per-block schedule, identical to
    #                   the inference "blockwise" pyramid family. Aligns train/inference
    #                   distributions and avoids the unseen "mostly-clean tail" states.
    block_time_sampling: str = "independent"
    # Exponent for inverse-frequency loss reweighting under monotone schedule:
    #   w_k = (num_blocks / (k + 1)) ^ reweight_gamma,  normalised so mean=1.
    #
    # Controls the strength of per-block loss reweighting:
    #   gamma = 0.0  →  no reweighting (all blocks equal weight)
    #   gamma = 0.5  →  moderate (block 0/9 ratio ≈ 3.2:1)
    #   gamma = 1.0  →  full inverse-frequency (block 0/9 ratio = 10:1)
    #   gamma > 1.0  →  aggressive (even more weight on early blocks)
    #
    # Motivation: under monotone schedule, block k receives gradients only
    # (k+1)/num_blocks fraction of training steps.  Block 0 gets 10x fewer
    # gradients than block 9, causing undertrained first-block quality at
    # inference.  gamma=1.0 fully compensates; gamma<1.0 partially compensates.
    #
    # Has no effect when block_time_sampling=="independent" (already balanced).
    reweight_gamma: float = 1.0
    # Alpha parameter for the Beta(alpha, 1) distribution used to sample the
    # monotone progress scalar `phase` during training.
    #
    # Background: under monotone sampling, `phase ~ Uniform(0,1)` causes block k
    # to receive gradients only (k+1)/num_blocks of the time.  Block 0 is noisy
    # only when phase < 1/num_blocks, and its t values are then Uniform(0,1).
    # However, with num_steps=50 inference, block 0 uses t ∈ [0.2, 1.0], so a
    # large fraction of the training budget for block 0 already covers the
    # inference regime even with uniform phase.
    #
    # Beta(alpha, 1) has PDF ∝ phase^(alpha-1), skewing phase toward 0 (high t):
    #   alpha = 1.0  →  Uniform(0,1)  [no change; default for backward compat]
    #   alpha = 0.7  →  mild skew; block 0 gradient freq: 10% → 20%
    #   alpha = 0.5  →  moderate; block 0 gradient freq: 10% → 32%  (recommended)
    #   alpha = 0.3  →  strong;   block 0 gradient freq: 10% → 50%
    #
    # Effect with num_steps=50 inference:
    #   alpha=1.0 (uniform): block 0 effective coverage ≈  8%
    #   alpha=0.5:           block 0 effective coverage ≈ 28%  (3.5x improvement)
    #
    # Only affects block_time_sampling=="monotone"; ignored for "independent".
    phase_alpha: float = 1.0

    # ── Tactile injection (optional) ───────────────────────────────────────────
    # When enabled, a tactile encoder maps left/right rgb_marker images into
    # 16 tokens each (32 total) that are prepended to the action-expert suffix.
    # During training the model selects the tactile frame corresponding to the
    # number of clean blocks (derived from the monotone progress scalar).
    # During inference the tactile tokens are updated each time a new block
    # becomes clean and its actions are executed on the robot.
    use_tactile: bool = False
    tactile_tokens_per_finger: int = 16

    # Encoder backend:
    #   "resnet"  — ResNet-18 trained from scratch (single frame, 3-ch input).
    #               Original default; no extra dependencies.
    #   "sparsh"  — Sparsh DINO ViT-Small (two frames cat'd → 6-ch input).
    #               Requires converted weights at `sparsh_npz_path`.
    #               AttentionPool + proj are randomly init'd; ViT is pretrained.
    tactile_encoder_type: str = "resnet"

    # Path to .npz weights produced by convert_sparsh_weights.py.
    # Only used when tactile_encoder_type == "sparsh".
    sparsh_npz_path: str = "/data/zjb/ckpts/sparsh/dino/sparsh_dino_small_jax.npz"

    # When True, freeze the ViT backbone of SparshTactileEncoder during training.
    # The AttentionPool and projection layers are always trainable.
    # Useful for the first phase of training when data is limited.
    sparsh_freeze_backbone: bool = False

    # When True, the ViT register token (global DINO summary at index 0 of the
    # 197-token sequence) is extracted and prepended as an extra token per finger,
    # yielding (tactile_tokens_per_finger + 1) tokens per finger instead of
    # tactile_tokens_per_finger.  Total tactile tokens = (tactile_tokens_per_finger+1)*2.
    # Only applicable when tactile_encoder_type == "sparsh"; silently ignored otherwise.
    # The register token receives finger_emb but NOT spatial_emb; a dedicated
    # register_token_emb (zero-init, D-dim) is added to distinguish it from spatial tokens.
    use_tactile_register_token: bool = False

    # When True, both tactile encoders (resnet and sparsh) add learned positional
    # encodings to the output tokens:
    #   spatial_emb  — one vector per grid cell (num_tokens × D), row-major 4×4 grid.
    #                  Lets the model distinguish the spatial location of each patch.
    #   finger_emb   — one vector per finger (2 × D, index 0=left, 1=right).
    #                  Lets the model distinguish which hand the token belongs to.
    #
    # Setting False removes both parameter blocks from the model, useful for ablation.
    # This is a model-architecture choice saved with the checkpoint; it CANNOT be
    # changed at evaluation time without retraining.
    tactile_use_pos_emb: bool = True

    # ── Tactile expert (future tactile prediction via flow matching) ────────────
    # When enabled, adds an INDEPENDENT Transformer (third expert) that denoises
    # future tactile latent tokens alongside the action tokens. Both experts have
    # separate parameters but mutually attend each other via the shared attention
    # mask. The tactile expert predicts the next-block tactile observation. Its
    # noise schedule is synchronized with the action block becoming clean.
    use_tactile_expert: bool = False
    # Gemma variant for the tactile expert Transformer (independent params).
    # Must share head_dim/num_heads/num_kv_heads/depth with other experts.
    tactile_expert_variant: _gemma.Variant = "gemma_300m"
    # Number of tokens for the tactile expert stream (predicted future tactile).
    tactile_expert_num_tokens: int = 32
    # Loss weight for the tactile expert velocity matching loss.
    tactile_expert_loss_weight: float = 0.5
    # When True, the tactile expert tokens can only attend to action block 0
    # (the block that becomes clean first), not all action tokens.
    # This enforces a local attention pattern: tactile predicts future tactile
    # conditioned only on the immediately upcoming action block.
    tac_expert_local_attn: bool = False

    # When True (default), current tactile tokens in the action-expert suffix are
    # allowed to attend to all prefix tokens (image/language conditioning), which
    # is the standard cross-stream behaviour.
    # When False, tactile tokens are restricted to self-attend only (they cannot
    # attend to any prefix token), effectively making them a prefix-independent
    # tactile condition that interacts with action tokens only.
    tactile_attend_prefix: bool = True

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]
        assert self.action_horizon % self.num_blocks == 0, (
            f"action_horizon ({self.action_horizon}) must be divisible by num_blocks ({self.num_blocks})"
        )
        assert self.block_time_sampling in ("independent", "monotone"), (
            f"block_time_sampling must be 'independent' or 'monotone', got {self.block_time_sampling!r}"
        )

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0DF":
        from openpi.models.pi0_df import Pi0DF

        return Pi0DF(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
