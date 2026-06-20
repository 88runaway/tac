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
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
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

        # This attribute gets automatically set by model.train() and model.eval().
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
        if self.use_tactile:
            self.tactile_encoder = _tactile_enc.TactileResNetEncoder(
                output_dim=action_expert_config.width,
                num_tokens=self.tactile_tokens_per_finger,
                rngs=rngs,
            )
            self.num_tactile_tokens = self.tactile_tokens_per_finger * 2  # left + right

        logger.info(
            f"Pi0DF: num_blocks={self.num_blocks}, block_size={self.block_size}, "
            f"mix_prob={self.mix_prob}, block_time_sampling={self.block_time_sampling}, "
            f"use_tactile={self.use_tactile}"
        )

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
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
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    def encode_tactile(
        self,
        tactile_left: at.Float[at.Array, "b h w 3"],
        tactile_right: at.Float[at.Array, "b h w 3"],
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "b nt emb"]:
        """Encode left + right tactile images into ``num_tactile_tokens`` tokens."""
        left_tok = self.tactile_encoder(tactile_left, train=train)    # (b, 16, emb)
        right_tok = self.tactile_encoder(tactile_right, train=train)  # (b, 16, emb)
        return jnp.concatenate([left_tok, right_tok], axis=1)         # (b, 32, emb)

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
        """Embed the suffix with a *per-position* timestep ``(b, ah)``.

        When ``tactile_tokens`` is provided (shape ``(b, num_tactile_tokens, emb)``),
        they are prepended before the action tokens in the suffix and form their own
        causal block. The resulting block structure is ``prefix < tactile < action``:
        tactile attends only the prefix and itself (never the noisy action tokens, so
        its representation is stable and cacheable), while action attends the prefix,
        the tactile tokens and itself. The prefix cannot attend back to the suffix.

        The adaRMS conditioning is zero-padded for tactile positions (no time-dependent
        modulation) and set to the per-action time embedding for action positions.
        """
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            ar_mask += [True]

        # ── Tactile tokens (prepended before action tokens) ───────────────────
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

            # Build adaRMS conditioning: zeros for tactile, time_emb for actions
            if tactile_tokens is not None:
                tac_cond = jnp.zeros(
                    (tactile_tokens.shape[0], tactile_tokens.shape[1], time_emb.shape[-1]),
                    dtype=time_emb.dtype,
                )
                adarms_cond = jnp.concatenate([tac_cond, time_emb], axis=1)
            else:
                adarms_cond = time_emb
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
        # Action tokens always start a new causal block. With tactile present this
        # places them *after* the tactile block (cumsum: prefix=0, tactile=1, action=2),
        # so action can attend prefix+tactile+action while tactile attends only
        # prefix+tactile (never the noisy action tokens).
        ar_mask += [True] + ([False] * (self.action_horizon - 1))

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    def _expand_blocks(self, block_values: jax.Array) -> jax.Array:
        """Expand per-block values ``(b, num_blocks)`` to per-position ``(b, ah)``."""
        return jnp.repeat(block_values, self.block_size, axis=1)

    def _pyramid_block_time(self, rng: at.KeyArrayLike, batch: int) -> jax.Array:
        """Sample a fixed monotone (earlier blocks cleaner) per-position time ``(b, ah)``.

        A single global progress scalar ``p ~ U(0, 1)`` is drawn per batch element and mapped
        to a monotonically-increasing per-block noise level via the same formula used by the
        inference ``blockwise`` pyramid schedule (see ``_blockwise_time_schedule``):

            ``t_k = clip(1 - p * num_blocks / (k + 1), 0, 1)``

        At ``p = 0`` all blocks are pure noise (``t = 1``); as ``p → 1`` earlier blocks reach the
        clean state first, with block ``k`` becoming clean at ``p = (k + 1) / num_blocks``. Drawing
        ``p`` uniformly matches the (linear) progress the inference sampler sweeps through, so the
        training and inference per-block time distributions coincide.
        """
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
    ) -> at.Float[at.Array, "b nt emb"] | None:
        """Select & encode the tactile frame matching the monotone progress ``p``.

        For the monotone DF branch, ``c = floor(p * num_blocks)`` gives the number
        of clean blocks; ``tactile[:, c]`` is the observation after executing ``c``
        blocks.  For the const branch (``use_block == False``) ``c = 0`` (initial
        observation).  For the ``independent`` sampling mode ``c = 0`` as well (no
        well-defined monotone ordering).
        """
        if not self.use_tactile or observation.tactile is None:
            return None

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
        sel_left = tac_left[idx, c]    # (b, H, W, 3)
        sel_right = tac_right[idx, c]  # (b, H, W, 3)
        return self.encode_tactile(sel_left, sel_right, train=train)

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng, block_rng, type_rng, phase_rng = jax.random.split(rng, 6)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        b, ah, ad = actions.shape
        noise = jax.random.normal(noise_rng, actions.shape)

        # standard flow-matching time: one scalar per batch, broadcast across the chunk
        time_scalar = jax.random.beta(time_rng, 1.5, 1, (b, 1)) * 0.999 + 0.001  # (b, 1)
        time_const = jnp.broadcast_to(time_scalar, (b, ah))  # (b, ah)

        # block-wise diffusion forcing time
        phase = jax.random.uniform(phase_rng, (b, 1))  # save phase for tactile selection
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
        tactile_tokens = self._select_tactile_for_training(
            observation, phase, use_block, train=train
        )

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, time, tactile_tokens=tactile_tokens,
        )
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        per_token_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)  # (b, ah)

        active_mask = (time > 5e-4).astype(per_token_loss.dtype)  # (b, ah)
        return per_token_loss * active_mask

    def _blockwise_time_schedule(self, num_steps: int, num_blocks: int | None = None) -> jax.Array:
        """Build a block-wise diffusion-forcing pyramid schedule.

        All blocks denoise *simultaneously* over a fixed ``num_steps`` outer steps, but at
        different rates so that earlier blocks reach the clean state first, maintaining a
        noise-level gradient across blocks at every step.

        Concretely, block ``k`` (0-indexed) descends linearly from ``t = 1`` (pure noise) to
        ``t = 0`` (clean) over its first ``(k + 1) / num_blocks`` fraction of the schedule, i.e.
        it becomes clean at outer step ``(k + 1) / num_blocks * num_steps``. With ``num_blocks``
        blocks this means block 0 is clean after ``num_steps / num_blocks`` steps, block 1 after
        ``2 * num_steps / num_blocks`` steps, and the last block exactly at ``num_steps``.

        Args:
            num_steps: total outer denoising steps.
            num_blocks: number of blocks. Overrides ``self.num_blocks`` when provided (e.g. set
                via ``block_size`` in the eval config). Must divide ``self.action_horizon``.

        Returns an array of shape ``(num_steps + 1, ah)`` with values in ``[0, 1]``.
        """
        nb = num_blocks if num_blocks is not None else self.num_blocks
        bs = self.action_horizon // nb
        m = jnp.arange(num_steps + 1)[:, None]  # (num_steps+1, 1)
        k = jnp.arange(nb)[None, :]              # (1, nb)
        # outer step at which block k reaches clean (t == 0)
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

        For standard ``const`` / ``blockwise`` schedules, ``tactile_tokens``
        (if provided) remain fixed throughout the denoising.  For interactive
        block-level tactile feedback, use ``sample_actions_interactive`` instead.
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def forward_velocity(x_t, time, tac_tok):
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, time, tactile_tokens=tac_tok,
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask_local = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask_local, suffix_attn_mask], axis=-1)
            positions_local = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions_local,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            return self.action_out_proj(suffix_out[:, -self.action_horizon :])

        if infer_time_schedule == "const":
            dt = -1.0 / num_steps

            def step(carry):
                x_t, time = carry
                time_full = jnp.broadcast_to(time, (batch_size, self.action_horizon))
                v_t = forward_velocity(x_t, time_full, tactile_tokens)
                return x_t + dt * v_t, time + dt

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

            def step_adaptive(x_t, step_params):
                t_curr, dt_curr = step_params
                v_t = forward_velocity(x_t, t_curr, tactile_tokens)
                x_next = x_t + dt_curr[..., None] * v_t
                return x_next, None

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
    ) -> at.Float[at.Array, "b ah ad"]:
        """Run a segment of denoising steps with fixed ``tactile_tokens``.

        This is the inner loop used by the interactive inference server:
        run a fixed number of steps with the current tactile, then return
        the partially-denoised actions so the caller can execute newly-clean
        blocks and obtain fresh tactile observations.
        """
        def _step(x_t, params):
            t_curr, dt_curr = params
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, t_curr, tactile_tokens=tactile_tokens,
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask_local = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask_local, suffix_attn_mask], axis=-1)
            positions_local = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions_local,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            return x_t + dt_curr[..., None] * v_t, None

        x_out, _ = jax.lax.scan(_step, x_t, (t_starts_seg, dt_seg))
        return x_out
