"""Pi0 / Pi0.5 with block-wise Diffusion Forcing (DF).

This model extends the standard Pi0 flow-matching policy so that the flow-matching
*timestep* (i.e. the noise level) is defined **per action token** instead of being a
single scalar shared across the whole action chunk.

The action horizon is split into ``num_blocks`` contiguous blocks; within a block the
noise level is uniform. This is the discrete / block-wise variant of Diffusion Forcing
(Chen et al.) adapted to the Pi0 architecture:

  * Training: with probability ``mix_prob`` each block gets an *independent* noise level
    (block-wise diffusion forcing), otherwise the whole chunk shares a single noise level
    (standard flow matching). Mixing keeps the model close to the pretrained pi05_base
    behaviour and stabilises training.

  * Inference: ``sample_actions`` supports two schedules selected by ``infer_time_schedule``
      - ``"const"``    : identical to standard Pi0 (all tokens share the same time).
      - ``"blockwise"``: a diffusion-forcing pyramid schedule. Over a fixed ``num_steps`` outer
        steps, *all* blocks denoise simultaneously but at different rates so that earlier blocks
        reach the clean state first (block ``k`` becomes clean at ``(k+1)/num_blocks * num_steps``).
        This keeps a noise-level gradient across blocks at every step — the discrete / block-wise
        analogue of the Diffusion Forcing pyramid sampler — without inflating the step count.

The only architectural requirement beyond standard Pi0 is that the action-expert adaRMS
conditioning accepts a per-position tensor ``(b, ah, emb)``; the shared ``gemma.RMSNorm``
already handles both ``(b, emb)`` and ``(b, ah, emb)`` conditioning.

Tactile Expert (optional):
    When ``use_tactile_expert=True``, a THIRD gemma expert (independent Transformer
    weights) is instantiated to denoise future tactile latent tokens. The three experts
    share attention (mutual attention between action and tactile expert) but have
    completely separate QKV/FFN/Norm parameters.

RTC (real-time chunking) is intentionally **not** implemented here.
"""

import functools
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import tactile_encoder as _tactile_enc
from openpi.models import sparsh_encoder as _sparsh_enc
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision. See pi0.make_attn_mask for the full docstring."""
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0DF(_model.BaseModel):
    def __init__(self, config: "pi0_config.Pi0DFConfig", rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        # ── Build the multi-expert LLM ────────────────────────────────────────────
        self.use_tactile_expert = config.use_tactile_expert
        if self.use_tactile_expert:
            tactile_expert_config = _gemma.get_config(config.tactile_expert_variant)
            self._tactile_expert_width = tactile_expert_config.width
            llm_configs = [paligemma_config, action_expert_config, tactile_expert_config]
            use_adarms = [False, True, True] if config.pi05 else [False, False, False]
        else:
            self._tactile_expert_width = None
            llm_configs = [paligemma_config, action_expert_config]
            use_adarms = [False, True] if config.pi05 else [False, False]

        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=llm_configs,
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=use_adarms)
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        self.deterministic = True

        # ── Diffusion forcing config ─────────────────────────────────────────────
        self.num_blocks = config.num_blocks
        assert self.action_horizon % self.num_blocks == 0, (
            f"action_horizon ({self.action_horizon}) must be divisible by num_blocks ({self.num_blocks})"
        )
        self.block_size = self.action_horizon // self.num_blocks
        self.mix_prob = config.mix_prob
        assert 0.0 <= self.mix_prob <= 1.0, "mix_prob must be in [0, 1]"
        self.block_time_sampling = config.block_time_sampling
        assert self.block_time_sampling in ("independent", "monotone"), (
            f"block_time_sampling must be 'independent' or 'monotone', got {self.block_time_sampling!r}"
        )

        # ── Tactile encoder (optional) ────────────────────────────────────────────
        self.use_tactile = config.use_tactile
        self.tactile_tokens_per_finger = config.tactile_tokens_per_finger
        self.tactile_encoder_type = config.tactile_encoder_type
        if self.use_tactile:
            _use_reg_tok = (
                config.tactile_encoder_type == "sparsh"
                and getattr(config, "use_tactile_register_token", False)
            )
            if config.tactile_encoder_type == "sparsh":
                self.tactile_encoder = _sparsh_enc.SparshTactileEncoder.from_pretrained(
                    npz_path=config.sparsh_npz_path,
                    output_dim=action_expert_config.width,
                    num_tokens=self.tactile_tokens_per_finger,
                    use_register_token=_use_reg_tok,
                    rngs=rngs,
                )
                if config.sparsh_freeze_backbone:
                    # Backbone freezing is applied externally via the training
                    # freeze_filter (see train_df._build_freeze_filter).
                    logger.info("Pi0DF: sparsh_freeze_backbone=True — backbone will "
                                "be frozen by the training freeze_filter.")
            else:  # "resnet" (default)
                self.tactile_encoder = _tactile_enc.TactileResNetEncoder(
                    output_dim=action_expert_config.width,
                    num_tokens=self.tactile_tokens_per_finger,
                    rngs=rngs,
                )
            # Each finger outputs (tactile_tokens_per_finger + 1) tokens when
            # the register token is enabled, otherwise tactile_tokens_per_finger.
            _tokens_per_finger = self.tactile_tokens_per_finger + (1 if _use_reg_tok else 0)
            self.num_tactile_tokens = _tokens_per_finger * 2  # left + right

        # ── Tactile attention-to-prefix control ──────────────────────────────────
        # When False, current tactile tokens in the suffix are blocked from
        # attending to prefix tokens (image/language); they can only self-attend
        # and attend to action tokens.
        self.tactile_attend_prefix = config.tactile_attend_prefix

        # ── Tactile expert projections (independent Transformer via 3rd expert) ───
        if self.use_tactile_expert:
            assert self.use_tactile, "use_tactile_expert requires use_tactile=True"
            self.tactile_expert_num_tokens = config.tactile_expert_num_tokens
            self.tactile_expert_loss_weight = config.tactile_expert_loss_weight
            self.tac_expert_local_attn = config.tac_expert_local_attn
            tac_width = self._tactile_expert_width
            act_width = action_expert_config.width
            # Input: project noisy tactile latent (act_width) → tactile expert width
            self.tac_expert_in_proj = nnx.Linear(act_width, tac_width, rngs=rngs)
            # Time MLP for tactile expert adaRMS conditioning
            self.tac_expert_time_mlp_in = nnx.Linear(tac_width, tac_width, rngs=rngs)
            self.tac_expert_time_mlp_out = nnx.Linear(tac_width, tac_width, rngs=rngs)
            # Output: project tactile expert output (tac_width) → act_width for loss
            self.tac_expert_out_proj = nnx.Linear(tac_width, act_width, rngs=rngs)

        logger.info(
            f"Pi0DF: num_blocks={self.num_blocks}, block_size={self.block_size}, "
            f"mix_prob={self.mix_prob}, block_time_sampling={self.block_time_sampling}, "
            f"use_tactile={self.use_tactile}"
            + (f" [{config.tactile_encoder_type}]" if self.use_tactile else "")
            + (f", freeze_backbone={config.sparsh_freeze_backbone}"
               if self.use_tactile and config.tactile_encoder_type == "sparsh" else "")
            + (f", tactile_attend_prefix={self.tactile_attend_prefix}"
               f", num_tactile_tokens={self.num_tactile_tokens}"
               if self.use_tactile else "")
            + f", use_tactile_expert={self.use_tactile_expert}"
            + (f", tactile_expert_variant={config.tactile_expert_variant}" if self.use_tactile_expert else "")
        )

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            ar_mask += [False] * image_tokens.shape[1]

        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    def encode_tactile(
        self,
        tactile_left:  at.Float[at.Array, "b h w 3"],
        tactile_right: at.Float[at.Array, "b h w 3"],
        prev_tactile_left:  at.Float[at.Array, "b h w 3"] | None = None,
        prev_tactile_right: at.Float[at.Array, "b h w 3"] | None = None,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "b nt emb"]:
        """Encode left + right tactile images into ``num_tactile_tokens`` tokens.

        For the "sparsh" encoder the current and previous frames are concatenated
        along the channel dimension to form a 6-channel input (current‖prev).
        When ``prev_*`` is None (e.g. first block) the current frame is used as
        the previous frame, resulting in a zero-difference signal.

        For the "resnet" encoder the previous frames are ignored.
        """
        if self.tactile_encoder_type == "sparsh":
            # Fall back to current frame when no previous frame is available
            if prev_tactile_left is None:
                prev_tactile_left = tactile_left
            if prev_tactile_right is None:
                prev_tactile_right = tactile_right
            # Concatenate along channel dim: (b, h, w, 3) ‖ (b, h, w, 3) → (b, h, w, 6)
            left_inp  = jnp.concatenate([tactile_left,  prev_tactile_left],  axis=-1)
            right_inp = jnp.concatenate([tactile_right, prev_tactile_right], axis=-1)
            left_tok  = self.tactile_encoder(left_inp,  finger_idx=0)  # (b, 16, emb)
            right_tok = self.tactile_encoder(right_inp, finger_idx=1)  # (b, 16, emb)
        else:  # "resnet"
            left_tok  = self.tactile_encoder(tactile_left,  train=train)   # (b, 16, emb)
            right_tok = self.tactile_encoder(tactile_right, train=train)   # (b, 16, emb)
        return jnp.concatenate([left_tok, right_tok], axis=1)              # (b, 32, emb)

    @at.typecheck
    def embed_suffix(
        self,
        obs: _model.Observation,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, "b ah"],
        tactile_tokens: at.Float[at.Array, "b nt emb"] | None = None,
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b st emb"] | None,
    ]:
        """Embed the action-expert suffix (expert 1).

        Token layout: [current_tactile (condition)] [action_tokens]

        When tactile expert is enabled, action tokens share a causal block with the
        tactile expert (expert 2) via ar_mask. When disabled, action tokens form their
        own causal block (original Pi0DF behaviour).
        """
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            ar_mask += [True]

        # ── Current tactile tokens (condition) ────────────────────────────────
        if tactile_tokens is not None:
            nt = tactile_tokens.shape[1]
            tokens.append(tactile_tokens)
            input_mask.append(jnp.ones(tactile_tokens.shape[:2], dtype=jnp.bool_))
            ar_mask += [True] + ([False] * (nt - 1))

        # ── Action tokens ─────────────────────────────────────────────────────
        action_tokens = self.action_in_proj(noisy_actions)
        time_emb = jax.vmap(
            functools.partial(
                posemb_sincos, embedding_dim=self.action_in_proj.out_features, min_period=4e-3, max_period=4.0
            )
        )(timestep)
        if self.pi05:
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens

            # adaRMS conditioning: zeros for tactile condition + action time emb
            cond_parts = []
            if tactile_tokens is not None:
                cond_parts.append(jnp.zeros(
                    (tactile_tokens.shape[0], tactile_tokens.shape[1], time_emb.shape[-1]),
                    dtype=time_emb.dtype,
                ))
            cond_parts.append(time_emb)
            adarms_cond = jnp.concatenate(cond_parts, axis=1)
        else:
            time_tokens = time_emb
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None

        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # When tactile expert is enabled, action shares causal block with expert 2
        # (ar_mask=False → same cumsum → bidirectional attention with expert 2 tokens).
        # When disabled, action starts its own block (original behaviour).
        if self.use_tactile_expert:
            ar_mask += [True] + ([False] * (self.action_horizon - 1))
        else:
            ar_mask += [True] + ([False] * (self.action_horizon - 1))

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    def embed_tactile_expert(
        self,
        noisy_future_tactile: at.Float[at.Array, "b nft emb"],
        tactile_expert_timestep: at.Float[at.Array, "b nft"],
    ) -> tuple[
        at.Float[at.Array, "b nft tac_w"],
        at.Bool[at.Array, "b nft"],
        at.Bool[at.Array, " nft"],
        at.Float[at.Array, "b nft tac_w"],
    ]:
        """Embed tokens for the tactile expert (expert 2).

        Returns (tokens, input_mask, ar_mask, adarms_cond) for the third expert.
        ar_mask is all-False so tokens share the same causal block as action tokens
        (enabling bidirectional mutual attention).
        """
        nft = noisy_future_tactile.shape[1]
        b = noisy_future_tactile.shape[0]
        tac_width = self._tactile_expert_width

        # Project noisy tactile latent → tactile expert embedding space
        tac_tokens = self.tac_expert_in_proj(noisy_future_tactile)  # (b, nft, tac_width)

        input_mask = jnp.ones((b, nft), dtype=jnp.bool_)
        # All False: same cumsum as action tokens → bidirectional attention
        ar_mask = jnp.array([False] * nft)

        # adaRMS conditioning from timestep
        tac_time_emb = jax.vmap(
            functools.partial(
                posemb_sincos, embedding_dim=tac_width, min_period=4e-3, max_period=4.0
            )
        )(tactile_expert_timestep)
        tac_time_emb = self.tac_expert_time_mlp_in(tac_time_emb)
        tac_time_emb = nnx.swish(tac_time_emb)
        tac_time_emb = self.tac_expert_time_mlp_out(tac_time_emb)
        tac_time_emb = nnx.swish(tac_time_emb)

        return tac_tokens, input_mask, ar_mask, tac_time_emb

    def _expand_blocks(self, block_values: jax.Array) -> jax.Array:
        """Expand per-block values ``(b, num_blocks)`` to per-position ``(b, ah)``."""
        return jnp.repeat(block_values, self.block_size, axis=1)

    def _pyramid_block_time(self, rng: at.KeyArrayLike, batch: int) -> jax.Array:
        """Sample a fixed monotone (earlier blocks cleaner) per-position time ``(b, ah)``."""
        phase = jax.random.uniform(rng, (batch, 1))  # (b, 1) in [0, 1)
        k = jnp.arange(self.num_blocks)[None, :]  # (1, num_blocks)
        t_blocks = jnp.clip(1.0 - phase * self.num_blocks / (k + 1), 0.0, 1.0)  # (b, num_blocks)
        return self._expand_blocks(t_blocks)  # (b, ah)

    def _select_tactile_for_training(
        self,
        observation: _model.Observation,
        phase: at.Float[at.Array, "b 1"],
        use_block: at.Bool[at.Array, "b 1"],
        *,
        train: bool = False,
    ) -> tuple[at.Float[at.Array, "b nt emb"] | None, at.Float[at.Array, "b nt emb"] | None]:
        """Select & encode current tactile + future tactile target.

        Returns:
            current_tactile_tokens: Encoded current tactile (condition).
            future_tactile_target: Encoded next-block tactile (prediction target).
                None if tactile expert is disabled.
        """
        if not self.use_tactile or observation.tactile is None:
            return None, None

        tac_left = observation.tactile["left"]    # (b, num_blocks, H, W, 3)
        tac_right = observation.tactile["right"]  # (b, num_blocks, H, W, 3)
        b = tac_left.shape[0]

        if self.block_time_sampling == "monotone":
            c = jnp.floor(phase[:, 0] * self.num_blocks).astype(jnp.int32)
            c = jnp.clip(c, 0, self.num_blocks - 1)  # (b,)
            c = jnp.where(use_block[:, 0], c, 0)      # const branch → c=0
        else:
            c = jnp.zeros(b, dtype=jnp.int32)

        idx = jnp.arange(b)
        sel_left  = tac_left[idx, c]    # (b, H, W, 3)  current frame
        sel_right = tac_right[idx, c]   # (b, H, W, 3)

        # For Sparsh: also select the "previous block" frame (c-1, clamped to 0).
        # At block 0, prev == current → zero temporal difference (a safe neutral signal).
        if self.tactile_encoder_type == "sparsh":
            prev_c     = jnp.clip(c - 1, 0, self.num_blocks - 1)
            prev_left  = tac_left[idx, prev_c]    # (b, H, W, 3)
            prev_right = tac_right[idx, prev_c]   # (b, H, W, 3)
        else:
            prev_left = prev_right = None

        if self.use_tactile_expert:
            # Batch current + future into a single encoder forward to halve cost.
            c_next     = jnp.clip(c + 1, 0, self.num_blocks - 1)
            fut_left   = tac_left[idx, c_next]    # (b, H, W, 3)
            fut_right  = tac_right[idx, c_next]   # (b, H, W, 3)
            all_left   = jnp.concatenate([sel_left,  fut_left],  axis=0)   # (2b, H, W, 3)
            all_right  = jnp.concatenate([sel_right, fut_right], axis=0)   # (2b, H, W, 3)

            if self.tactile_encoder_type == "sparsh":
                # For future frame, its "previous" is the current frame (sel)
                all_prev_left  = jnp.concatenate([prev_left,  sel_left],  axis=0)  # (2b, H, W, 3)
                all_prev_right = jnp.concatenate([prev_right, sel_right], axis=0)  # (2b, H, W, 3)
                all_tokens = self.encode_tactile(
                    all_left, all_right, all_prev_left, all_prev_right, train=train
                )
            else:
                all_tokens = self.encode_tactile(all_left, all_right, train=train)

            current_tactile_tokens = all_tokens[:b]
            future_tactile_target  = all_tokens[b:]
        else:
            current_tactile_tokens = self.encode_tactile(
                sel_left, sel_right, prev_left, prev_right, train=train
            )
            future_tactile_target = None

        return current_tactile_tokens, future_tactile_target

    def _apply_tac_local_attn_mask(
        self,
        attn_mask: at.Bool[at.Array, "b q k"],
        prefix_len: int,
        suffix_len: int,
        nft: int,
        *,
        is_cached: bool,
    ) -> at.Bool[at.Array, "b q k"]:
        """Block tactile expert from attending to action block 1..N.

        Sequence layout:
          Full forward:   [prefix(p) | suffix(s) | tac_expert(nft)]
          Cached forward: [suffix(s) | tac_expert(nft)] (prefix in kv_cache)
                           key cols:  [prefix(p) | suffix(s) | tac_expert(nft)]

        action tokens are the last ``self.action_horizon`` tokens of suffix.
        block 0 covers the first ``self.block_size`` action tokens.
        """
        ah = self.action_horizon
        bs = self.block_size
        q_len = attn_mask.shape[1]
        k_len = attn_mask.shape[2]

        if is_cached:
            # query positions in the current (non-cached) part
            tac_q_start = suffix_len
            tac_q_end = suffix_len + nft
            # key positions include cached prefix first
            act_block1_k = prefix_len + (suffix_len - ah + bs)
            act_end_k = prefix_len + suffix_len
        else:
            tac_q_start = prefix_len + suffix_len
            tac_q_end = tac_q_start + nft
            act_block1_k = prefix_len + (suffix_len - ah + bs)
            act_end_k = prefix_len + suffix_len

        q_idx = jnp.arange(q_len)
        k_idx = jnp.arange(k_len)
        # True where this is a tac_expert query position
        is_tac_q = (q_idx >= tac_q_start) & (q_idx < tac_q_end)    # (q,)
        # True where this is an action block_1..N key position
        is_block1_k = (k_idx >= act_block1_k) & (k_idx < act_end_k)  # (k,)
        # Block positions that are both tac query AND block_1+ key
        block = is_tac_q[:, None] & is_block1_k[None, :]              # (q, k)
        return attn_mask & ~block[None, :, :]  # broadcast over batch

    def _apply_tactile_no_prefix_mask(
        self,
        attn_mask: at.Bool[at.Array, "b q k"],
        prefix_len: int,
        suffix_len: int,
        *,
        is_cached: bool,
    ) -> at.Bool[at.Array, "b q k"]:
        """Block current tactile tokens (suffix head) from attending to prefix tokens.

        Sequence layout:
          Full forward:   [prefix(p) | suffix(s)]   key cols same
          Cached forward: [suffix(s)]  query,  key cols: [prefix(p) | suffix(s)]

        Tactile tokens occupy the first ``self.num_tactile_tokens`` positions of the
        suffix (pi05=True, so no state token prepended).
        """
        nt = self.num_tactile_tokens
        q_len = attn_mask.shape[1]
        k_len = attn_mask.shape[2]

        if is_cached:
            # query: current part starts at 0; tactile at 0..nt-1
            tac_q_start = 0
            tac_q_end = nt
            # key: [prefix(p) | suffix(s)] — prefix occupies 0..prefix_len-1
            prefix_k_end = prefix_len
        else:
            # query: tactile starts after prefix
            tac_q_start = prefix_len
            tac_q_end = prefix_len + nt
            # key: prefix occupies 0..prefix_len-1
            prefix_k_end = prefix_len

        q_idx = jnp.arange(q_len)
        k_idx = jnp.arange(k_len)
        is_tac_q  = (q_idx >= tac_q_start) & (q_idx < tac_q_end)   # (q,)
        is_prefix_k = k_idx < prefix_k_end                            # (k,)
        block = is_tac_q[:, None] & is_prefix_k[None, :]              # (q, k)
        return attn_mask & ~block[None, :, :]  # broadcast over batch

    def _forward_3expert(
        self,
        prefix_tokens, prefix_mask, prefix_ar_mask,
        suffix_tokens, suffix_mask, suffix_ar_mask, action_adarms,
        tac_expert_tokens=None, tac_expert_mask=None, tac_expert_ar_mask=None, tac_adarms=None,
        *,
        kv_cache=None,
        use_kv_cache: bool = False,
    ):
        """Run the multi-expert LLM forward pass.

        Handles both full forward (training) and cached-prefix forward (inference).

        Returns:
            suffix_out: output of expert 1 (action expert suffix).
            tac_expert_out: output of expert 2 (tactile expert), or None.
            kv_cache: updated kv_cache (or None).
        """
        if not use_kv_cache:
            # Full forward: prefix + suffix + (optional) tactile expert
            if tac_expert_tokens is not None:
                input_mask = jnp.concatenate([prefix_mask, suffix_mask, tac_expert_mask], axis=1)
                ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask, tac_expert_ar_mask], axis=0)
            else:
                input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
                ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
            attn_mask = make_attn_mask(input_mask, ar_mask)
            # Apply local attention: restrict tac_expert to only attend block 0
            if tac_expert_tokens is not None and self.use_tactile_expert and self.tac_expert_local_attn:
                attn_mask = self._apply_tac_local_attn_mask(
                    attn_mask,
                    prefix_len=prefix_tokens.shape[1],
                    suffix_len=suffix_tokens.shape[1],
                    nft=tac_expert_tokens.shape[1],
                    is_cached=False,
                )
            # Block tactile tokens from attending to prefix when tactile_attend_prefix=False
            if self.use_tactile and not self.tactile_attend_prefix:
                attn_mask = self._apply_tactile_no_prefix_mask(
                    attn_mask,
                    prefix_len=prefix_tokens.shape[1],
                    suffix_len=suffix_tokens.shape[1],
                    is_cached=False,
                )
            positions = jnp.cumsum(input_mask, axis=1) - 1

            if tac_expert_tokens is not None:
                outputs, new_kv = self.PaliGemma.llm(
                    [prefix_tokens, suffix_tokens, tac_expert_tokens],
                    mask=attn_mask, positions=positions,
                    adarms_cond=[None, action_adarms, tac_adarms],
                )
                _, suffix_out, tac_expert_out = outputs
            else:
                outputs, new_kv = self.PaliGemma.llm(
                    [prefix_tokens, suffix_tokens],
                    mask=attn_mask, positions=positions,
                    adarms_cond=[None, action_adarms],
                )
                _, suffix_out = outputs
                tac_expert_out = None
            return suffix_out, tac_expert_out, new_kv
        else:
            # Cached forward: suffix + (optional) tactile expert attend to prefix via kv_cache
            if tac_expert_tokens is not None:
                current_mask = jnp.concatenate([suffix_mask, tac_expert_mask], axis=1)
                current_ar = jnp.concatenate([suffix_ar_mask, tac_expert_ar_mask], axis=0)
            else:
                current_mask = suffix_mask
                current_ar = suffix_ar_mask
            current_attn = make_attn_mask(current_mask, current_ar)
            prefix_attend = einops.repeat(prefix_mask, "b p -> b s p", s=current_mask.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attend, current_attn], axis=-1)
            # Apply local attention in cached forward
            if tac_expert_tokens is not None and self.use_tactile_expert and self.tac_expert_local_attn:
                # prefix_len: full_attn_mask key dim = prefix_len + suffix_len + nft
                _nft = tac_expert_tokens.shape[1]
                _slen = suffix_tokens.shape[1]
                _plen = full_attn_mask.shape[2] - _slen - _nft
                full_attn_mask = self._apply_tac_local_attn_mask(
                    full_attn_mask,
                    prefix_len=_plen,
                    suffix_len=_slen,
                    nft=_nft,
                    is_cached=True,
                )
            # Block tactile tokens from attending to prefix when tactile_attend_prefix=False
            if self.use_tactile and not self.tactile_attend_prefix:
                _slen = suffix_tokens.shape[1]
                _plen = full_attn_mask.shape[2] - _slen - (
                    tac_expert_tokens.shape[1] if tac_expert_tokens is not None else 0
                )
                full_attn_mask = self._apply_tactile_no_prefix_mask(
                    full_attn_mask,
                    prefix_len=_plen,
                    suffix_len=_slen,
                    is_cached=True,
                )
            positions_local = (
                jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(current_mask, axis=-1) - 1
            )

            if tac_expert_tokens is not None:
                outputs, new_kv = self.PaliGemma.llm(
                    [None, suffix_tokens, tac_expert_tokens],
                    mask=full_attn_mask, positions=positions_local,
                    kv_cache=kv_cache,
                    adarms_cond=[None, action_adarms, tac_adarms],
                )
                _, suffix_out, tac_expert_out = outputs
            else:
                outputs, new_kv = self.PaliGemma.llm(
                    [None, suffix_tokens],
                    mask=full_attn_mask, positions=positions_local,
                    kv_cache=kv_cache,
                    adarms_cond=[None, action_adarms],
                )
                _, suffix_out = outputs
                tac_expert_out = None
            return suffix_out, tac_expert_out, new_kv

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng, block_rng, type_rng, phase_rng, tac_noise_rng = jax.random.split(rng, 7)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        b, ah, ad = actions.shape
        noise = jax.random.normal(noise_rng, actions.shape)

        # standard flow-matching time
        time_scalar = jax.random.beta(time_rng, 1.5, 1, (b, 1)) * 0.999 + 0.001  # (b, 1)
        time_const = jnp.broadcast_to(time_scalar, (b, ah))  # (b, ah)

        # block-wise diffusion forcing time
        phase = jax.random.uniform(phase_rng, (b, 1))
        if self.block_time_sampling == "monotone":
            k = jnp.arange(self.num_blocks)[None, :]
            t_blocks = jnp.clip(1.0 - phase * self.num_blocks / (k + 1), 0.0, 1.0)
            time_block_full = self._expand_blocks(t_blocks)
        else:
            time_blocks = jax.random.beta(block_rng, 1.5, 1, (b, self.num_blocks)) * 0.999 + 0.001
            time_block_full = self._expand_blocks(time_blocks)

        use_block = jax.random.bernoulli(type_rng, self.mix_prob, (b, 1))  # (b, 1)
        time = jnp.where(use_block, time_block_full, time_const)  # (b, ah)

        x_t = time[..., None] * noise + (1 - time[..., None]) * actions
        u_t = noise - actions

        # ── Tactile selection & encoding ──────────────────────────────────────
        tactile_tokens, future_tactile_target = self._select_tactile_for_training(
            observation, phase, use_block, train=train
        )

        # ── Tactile expert: noisy future tactile for flow matching ────────────
        tac_expert_tokens = None
        tac_expert_mask = None
        tac_expert_ar_mask = None
        tac_adarms = None
        tac_u_t = None
        tac_expert_timestep = None
        if self.use_tactile_expert and future_tactile_target is not None:
            nft = future_tactile_target.shape[1]  # num_tactile_tokens (32)
            emb_dim = future_tactile_target.shape[2]
            tac_noise_rng, tac_time_rng = jax.random.split(tac_noise_rng)
            tac_noise = jax.random.normal(tac_noise_rng, (b, nft, emb_dim))

            # Tactile expert timestep synchronized with current block
            if self.block_time_sampling == "monotone":
                c = jnp.floor(phase[:, 0] * self.num_blocks).astype(jnp.int32)
                c = jnp.clip(c, 0, self.num_blocks - 1)
                tac_t_scalar = t_blocks[jnp.arange(b), c][:, None]  # (b, 1)
            else:
                tac_t_scalar = jax.random.beta(
                    tac_time_rng, 1.5, 1, (b, 1)
                ) * 0.999 + 0.001
            tac_t_scalar = jnp.where(use_block, tac_t_scalar, time_scalar)

            tac_expert_timestep = jnp.broadcast_to(tac_t_scalar, (b, nft))  # (b, nft)
            noisy_future_tactile = (
                tac_t_scalar[..., None] * tac_noise +
                (1 - tac_t_scalar[..., None]) * future_tactile_target
            )
            tac_u_t = tac_noise - future_tactile_target  # target velocity

            # Embed for expert 2
            tac_expert_tokens, tac_expert_mask, tac_expert_ar_mask, tac_adarms = (
                self.embed_tactile_expert(noisy_future_tactile, tac_expert_timestep)
            )

        # ── Forward pass ──────────────────────────────────────────────────────
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, action_adarms = self.embed_suffix(
            observation, x_t, time, tactile_tokens=tactile_tokens,
        )

        suffix_out, tac_expert_out, _ = self._forward_3expert(
            prefix_tokens, prefix_mask, prefix_ar_mask,
            suffix_tokens, suffix_mask, suffix_ar_mask, action_adarms,
            tac_expert_tokens, tac_expert_mask, tac_expert_ar_mask, tac_adarms,
        )

        # ── Action velocity loss ──────────────────────────────────────────────
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
        per_token_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)  # (b, ah)
        active_mask = (time > 5e-4).astype(per_token_loss.dtype)  # (b, ah)
        action_loss = per_token_loss * active_mask
        action_loss_scalar = jnp.mean(action_loss)  # pure action loss for logging

        # ── Tactile expert velocity loss ──────────────────────────────────────
        tac_loss_scalar = jnp.zeros(())  # scalar 0 when expert disabled
        if self.use_tactile_expert and tac_expert_out is not None and tac_u_t is not None:
            tac_v_t = self.tac_expert_out_proj(tac_expert_out)  # (b, nft, act_width)
            tac_per_token_loss = jnp.mean(jnp.square(tac_v_t - tac_u_t), axis=-1)  # (b, nft)
            tac_active = (tac_expert_timestep > 5e-4).astype(tac_per_token_loss.dtype)
            tac_loss = jnp.mean(tac_per_token_loss * tac_active, axis=-1, keepdims=True)  # (b, 1)
            tac_loss_scalar = jnp.mean(tac_loss)
            action_loss = action_loss + self.tactile_expert_loss_weight * tac_loss

        # combined_loss is used for gradient; aux carries individual scalar components for logging.
        # action_loss_scalar: pure action FM loss (before adding tac)
        # tac_loss_scalar: raw tactile FM loss (before weighting by tactile_expert_loss_weight)
        aux = {
            "action_loss": action_loss_scalar,
            "tac_loss": tac_loss_scalar,
        }
        return action_loss, aux

    def _blockwise_time_schedule(self, num_steps: int, num_blocks: int | None = None) -> jax.Array:
        """Build a block-wise diffusion-forcing pyramid schedule.

        Returns an array of shape ``(num_steps + 1, ah)`` with values in ``[0, 1]``.
        """
        nb = num_blocks if num_blocks is not None else self.num_blocks
        bs = self.action_horizon // nb
        m = jnp.arange(num_steps + 1)[:, None]  # (num_steps+1, 1)
        k = jnp.arange(nb)[None, :]              # (1, nb)
        clean_step = (k + 1) * num_steps / nb    # (1, nb)
        t_block = jnp.clip(1.0 - m / clean_step, 0.0, 1.0)  # (num_steps+1, nb)
        t_schedule = jnp.repeat(t_block, bs, axis=1)         # (num_steps+1, ah)
        return t_schedule

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        infer_time_schedule: str = "const",
        num_blocks: int | None = None,
        tactile_tokens: at.Float[at.Array, "b nt emb"] | None = None,
    ) -> _model.Actions:
        """Sample actions with optional pre-encoded tactile tokens.

        When ``use_tactile_expert`` is enabled, a parallel tactile expert stream
        (independent Transformer) runs alongside action denoising. The tactile expert
        noise is re-initialized at each block boundary during blockwise inference.
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        # Prefill KV cache with prefix only
        if self.use_tactile_expert:
            prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
            positions = jnp.cumsum(prefix_mask, axis=1) - 1
            _, kv_cache = self.PaliGemma.llm(
                [prefix_tokens, None, None], mask=prefix_attn_mask, positions=positions,
            )
        else:
            prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
            positions = jnp.cumsum(prefix_mask, axis=1) - 1
            _, kv_cache = self.PaliGemma.llm(
                [prefix_tokens, None], mask=prefix_attn_mask, positions=positions,
            )

        # Initialize tactile expert noise if enabled
        tac_expert_noise = None
        if self.use_tactile_expert and tactile_tokens is not None:
            rng, tac_rng = jax.random.split(rng)
            nft = self.num_tactile_tokens
            emb_dim = tactile_tokens.shape[-1]
            tac_expert_noise = jax.random.normal(tac_rng, (batch_size, nft, emb_dim))

        def forward_velocity(x_t, time_full, tac_tok, tac_x_t=None, tac_time=None):
            """Single denoising step: compute action velocity and optionally tactile velocity."""
            suffix_tokens, suffix_mask, suffix_ar_mask, action_adarms = self.embed_suffix(
                observation, x_t, time_full, tactile_tokens=tac_tok,
            )
            tac_expert_tokens_local = None
            tac_expert_mask_local = None
            tac_expert_ar_mask_local = None
            tac_adarms_local = None
            if tac_x_t is not None and self.use_tactile_expert:
                tac_expert_tokens_local, tac_expert_mask_local, tac_expert_ar_mask_local, tac_adarms_local = (
                    self.embed_tactile_expert(tac_x_t, tac_time)
                )

            suffix_out, tac_expert_out, _ = self._forward_3expert(
                prefix_tokens, prefix_mask, prefix_ar_mask,
                suffix_tokens, suffix_mask, suffix_ar_mask, action_adarms,
                tac_expert_tokens_local, tac_expert_mask_local, tac_expert_ar_mask_local, tac_adarms_local,
                kv_cache=kv_cache, use_kv_cache=True,
            )
            action_v = self.action_out_proj(suffix_out[:, -self.action_horizon:])
            tac_v = None
            if tac_expert_out is not None:
                tac_v = self.tac_expert_out_proj(tac_expert_out)
            return action_v, tac_v

        if infer_time_schedule == "const":
            dt = -1.0 / num_steps

            if tac_expert_noise is not None:
                def step(carry):
                    x_t, tac_x_t, time = carry
                    time_full = jnp.broadcast_to(time, (batch_size, self.action_horizon))
                    tac_time_full = jnp.broadcast_to(time, (batch_size, self.num_tactile_tokens))
                    action_v, tac_v = forward_velocity(x_t, time_full, tactile_tokens, tac_x_t, tac_time_full)
                    return x_t + dt * action_v, tac_x_t + dt * tac_v, time + dt

                def cond(carry):
                    _, _, time = carry
                    return time >= -dt / 2

                x_0, _, _ = jax.lax.while_loop(cond, step, (noise, tac_expert_noise, 1.0))
            else:
                def step(carry):
                    x_t, time = carry
                    time_full = jnp.broadcast_to(time, (batch_size, self.action_horizon))
                    action_v, _ = forward_velocity(x_t, time_full, tactile_tokens)
                    return x_t + dt * action_v, time + dt

                def cond(carry):
                    _, time = carry
                    return time >= -dt / 2

                x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
            return x_0

        if infer_time_schedule == "blockwise":
            t_schedule = self._blockwise_time_schedule(num_steps, num_blocks)
            t_schedule = jnp.broadcast_to(t_schedule[:, None, :], (t_schedule.shape[0], batch_size, self.action_horizon))
            dt_schedule = t_schedule[1:] - t_schedule[:-1]
            t_starts = t_schedule[:-1]

            nb = num_blocks if num_blocks is not None else self.num_blocks
            steps_per_block = num_steps // nb

            if tac_expert_noise is not None:
                def step_adaptive(carry, step_params):
                    x_t, tac_x_t, step_idx = carry
                    t_curr, dt_curr = step_params
                    # Tactile expert time: the CURRENT noise level of tac_x_t.
                    # tac_x_t starts at t=1 (pure noise) and steps towards t=0.
                    # Euler: t_k = 1 - k/N where k = step_idx % steps_per_block.
                    tac_t = jnp.clip(
                        1.0 - (step_idx % steps_per_block) / steps_per_block, 0.0, 1.0
                    )
                    tac_time_full = jnp.broadcast_to(tac_t, (batch_size, self.num_tactile_tokens))
                    action_v, tac_v = forward_velocity(x_t, t_curr, tactile_tokens, tac_x_t, tac_time_full)
                    x_next = x_t + dt_curr[..., None] * action_v
                    # Tactile expert Euler step
                    tac_dt = -1.0 / steps_per_block
                    tac_x_next = tac_x_t + tac_dt * tac_v
                    # Re-initialize tactile noise at block boundaries
                    at_boundary = ((step_idx + 1) % steps_per_block == 0)
                    tac_x_next = jnp.where(
                        at_boundary,
                        jax.random.normal(
                            jax.random.fold_in(rng, step_idx),
                            tac_x_t.shape
                        ),
                        tac_x_next,
                    )
                    return (x_next, tac_x_next, step_idx + 1), None

                (x_0, _, _), _ = jax.lax.scan(
                    step_adaptive, (noise, tac_expert_noise, 0), (t_starts, dt_schedule)
                )
            else:
                def step_adaptive(x_t, step_params):
                    t_curr, dt_curr = step_params
                    action_v, _ = forward_velocity(x_t, t_curr, tactile_tokens)
                    return x_t + dt_curr[..., None] * action_v, None

                x_0, _ = jax.lax.scan(step_adaptive, noise, (t_starts, dt_schedule))
            return x_0

        raise ValueError(f"Invalid infer_time_schedule: {infer_time_schedule!r} (expected 'const' or 'blockwise')")

    # ── Interactive block-level inference with tactile feedback ────────────

    def prepare_interactive_inference(
        self, rng: at.KeyArrayLike, observation: _model.Observation,
    ):
        """Prepare prefix KV-cache and initial noise for interactive inference.

        Returns ``(noise, prefix_tokens, prefix_mask, kv_cache)`` that are
        passed to ``denoise_segment``.
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        batch_size = observation.state.shape[0]
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        if self.use_tactile_expert:
            _, kv_cache = self.PaliGemma.llm(
                [prefix_tokens, None, None], mask=prefix_attn_mask, positions=positions,
            )
        else:
            _, kv_cache = self.PaliGemma.llm(
                [prefix_tokens, None], mask=prefix_attn_mask, positions=positions,
            )
        return noise, observation, prefix_tokens, prefix_mask, kv_cache

    def denoise_segment(
        self,
        x_t: at.Float[at.Array, "b ah ad"],
        t_starts_seg: at.Float[at.Array, "steps b ah"],
        dt_seg: at.Float[at.Array, "steps b ah"],
        observation: _model.Observation,
        prefix_tokens,
        prefix_mask,
        kv_cache,
        tactile_tokens: at.Float[at.Array, "b nt emb"] | None,
        tac_expert_noise: at.Float[at.Array, "b nft emb"] | None = None,
    ) -> at.Float[at.Array, "b ah ad"]:
        """Run a segment of denoising steps with fixed ``tactile_tokens``.

        This is the inner loop used by the interactive inference server:
        run a fixed number of steps with the current tactile, then return
        the partially-denoised actions so the caller can execute newly-clean
        blocks and obtain fresh tactile observations.

        When ``tac_expert_noise`` is provided and ``use_tactile_expert`` is True,
        the tactile expert (independent Transformer) runs in parallel, denoising
        from noise to predicted future tactile within this segment.
        """
        num_seg_steps = t_starts_seg.shape[0]
        prefix_ar_mask = jnp.zeros(prefix_tokens.shape[1], dtype=jnp.bool_)

        def _step(carry, params):
            x_t_local, tac_x_t, step_idx = carry
            t_curr, dt_curr = params

            suffix_tokens, suffix_mask, suffix_ar_mask, action_adarms = self.embed_suffix(
                observation, x_t_local, t_curr, tactile_tokens=tactile_tokens,
            )

            tac_expert_tokens_local = None
            tac_expert_mask_local = None
            tac_expert_ar_mask_local = None
            tac_adarms_local = None
            if tac_x_t is not None and self.use_tactile_expert:
                tac_t = jnp.clip(1.0 - step_idx / num_seg_steps, 0.0, 1.0)
                tac_time = jnp.broadcast_to(tac_t, tac_x_t.shape[:2])
                tac_expert_tokens_local, tac_expert_mask_local, tac_expert_ar_mask_local, tac_adarms_local = (
                    self.embed_tactile_expert(tac_x_t, tac_time)
                )

            suffix_out, tac_expert_out, _ = self._forward_3expert(
                prefix_tokens, prefix_mask, prefix_ar_mask,
                suffix_tokens, suffix_mask, suffix_ar_mask, action_adarms,
                tac_expert_tokens_local, tac_expert_mask_local, tac_expert_ar_mask_local, tac_adarms_local,
                kv_cache=kv_cache, use_kv_cache=True,
            )

            # Action Euler step
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
            x_t_new = x_t_local + dt_curr[..., None] * v_t

            # Tactile expert Euler step
            tac_x_t_new = tac_x_t
            if tac_x_t is not None and self.use_tactile_expert and tac_expert_out is not None:
                tac_v = self.tac_expert_out_proj(tac_expert_out)
                tac_dt = -1.0 / num_seg_steps
                tac_x_t_new = tac_x_t + tac_dt * tac_v

            return (x_t_new, tac_x_t_new, step_idx + 1), None

        init_carry = (x_t, tac_expert_noise, 0)
        (x_out, _, _), _ = jax.lax.scan(_step, init_carry, (t_starts_seg, dt_seg))
        return x_out
