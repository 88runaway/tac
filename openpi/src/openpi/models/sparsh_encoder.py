"""Sparsh DINO ViT-Small tactile encoder in Flax NNX.

Architecture: timm-style ViT-Small/16 (matches the actual safetensors keys)
  - patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4
  - register_tokens  (1, 1, 384)  — single learnable "CLS-like" token
  - pos_embed: 2D SinCos from frequency_bands (2, 96)
  - LayerScale (ls1.gamma, ls2.gamma) in every Transformer block
  - in_channels=6  (current + previous tactile frame, 83 ms apart)
  - Input:  (B, H, W, 6) float32 in [0, 1]  (NHWC, JAX convention)
  - Output: (B, 197, 384)  — register token at [0] + 196 patch tokens

Pretrained weights: /data/zjb/ckpts/sparsh/dino/dino_vitsmall.safetensors
Converted weights:  /data/zjb/ckpts/sparsh/dino/sparsh_dino_small_jax.npz
  (produced by convert_sparsh_weights.py)

─────────────────────────────────────────────────────────────────────────────
Typical usage (full encoder with attention pooling):

    encoder = SparshTactileEncoder.from_pretrained(
        npz_path="/data/zjb/ckpts/sparsh/dino/sparsh_dino_small_jax.npz",
        output_dim=1024,          # action-expert width
        rngs=nnx.Rngs(0),
    )
    tokens = encoder(tactile_pair)   # (B, 16, 1024)
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)

# ── Architecture constants (ViT-Small/16) ────────────────────────────────────
EMBED_DIM   = 384
DEPTH       = 12
NUM_HEADS   = 6
HEAD_DIM    = EMBED_DIM // NUM_HEADS   # 64
MLP_HIDDEN  = EMBED_DIM * 4            # 1536
PATCH_SIZE  = 16
IMAGE_SIZE  = 224
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 14×14 = 196
SEQ_LEN     = NUM_PATCHES + 1                  # 197  (register_token + patches)
IN_CHANNELS = 6
LN_EPS      = 1e-6


# ── Helper: 2D SinCos position embedding ─────────────────────────────────────

def _sincos2d_from_freq_bands(
    frequency_bands: jax.Array,   # (2, D//4)
    grid_h: int = 14,
    grid_w: int = 14,
) -> jax.Array:
    """Compute absolute 2D SinCos position embedding from stored frequency bands.

    Matches timm's 2D Fourier PE forward pass:
        emb = [sin(h·ω_h), cos(h·ω_h), sin(w·ω_w), cos(w·ω_w)]
    where ω_h = frequency_bands[0], ω_w = frequency_bands[1].

    Returns:
        (1, grid_h*grid_w, embed_dim) — one embedding per patch.
    """
    freq_h = frequency_bands[0]   # (D//4,)
    freq_w = frequency_bands[1]   # (D//4,)

    hs = jnp.arange(grid_h, dtype=jnp.float32)
    ws = jnp.arange(grid_w, dtype=jnp.float32)

    h_ang = jnp.outer(hs, freq_h)   # (H, D//4)
    w_ang = jnp.outer(ws, freq_w)   # (W, D//4)

    h_ang_grid = jnp.broadcast_to(h_ang[:, None, :], (grid_h, grid_w, h_ang.shape[-1]))
    w_ang_grid = jnp.broadcast_to(w_ang[None, :, :], (grid_h, grid_w, w_ang.shape[-1]))

    emb = jnp.concatenate([
        jnp.sin(h_ang_grid),
        jnp.cos(h_ang_grid),
        jnp.sin(w_ang_grid),
        jnp.cos(w_ang_grid),
    ], axis=-1)                                    # (H, W, D)
    emb = emb.reshape(grid_h * grid_w, EMBED_DIM) # (H*W, D)
    return emb[None]                               # (1, H*W, D)


# ══════════════════════════════════════════════════════════════════════════════
# ViT sub-modules
# ══════════════════════════════════════════════════════════════════════════════

class _Attention(nnx.Module):
    def __init__(self, dim: int, num_heads: int, *, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.qkv  = nnx.Linear(dim, dim * 3, use_bias=True, rngs=rngs)
        self.proj = nnx.Linear(dim, dim,     use_bias=True, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        B, N, C = x.shape
        H, D = self.num_heads, self.head_dim
        qkv = self.qkv(x).reshape(B, N, 3, H, D)
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))    # (3, B, H, N, D)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ jnp.swapaxes(k, -2, -1)) * self.scale
        attn = jax.nn.softmax(attn, axis=-1)
        x = (attn @ v)
        x = jnp.transpose(x, (0, 2, 1, 3)).reshape(B, N, C)
        return self.proj(x)


class _MLP(nnx.Module):
    def __init__(self, in_features: int, hidden_features: int, *, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(in_features,   hidden_features, use_bias=True, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_features, in_features,   use_bias=True, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.fc2(jax.nn.gelu(self.fc1(x), approximate=False))


class _Block(nnx.Module):
    """Transformer block with pre-norm + LayerScale residual connections."""

    def __init__(self, dim: int, num_heads: int, mlp_hidden: int, *, rngs: nnx.Rngs):
        self.norm1     = nnx.LayerNorm(dim, epsilon=LN_EPS, rngs=rngs)
        self.attn      = _Attention(dim, num_heads, rngs=rngs)
        self.ls1_gamma = nnx.Param(jnp.ones(dim))   # LayerScale, pretrained values loaded later
        self.norm2     = nnx.LayerNorm(dim, epsilon=LN_EPS, rngs=rngs)
        self.mlp       = _MLP(dim, mlp_hidden, rngs=rngs)
        self.ls2_gamma = nnx.Param(jnp.ones(dim))

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x + self.ls1_gamma.value * self.attn(self.norm1(x))
        x = x + self.ls2_gamma.value * self.mlp(self.norm2(x))
        return x


# ══════════════════════════════════════════════════════════════════════════════
# ViT-Small backbone
# ══════════════════════════════════════════════════════════════════════════════

class SparshViTSmall(nnx.Module):
    """ViT-Small backbone for Sparsh DINO tactile encoder.

    Key architectural differences from plain DINO ViT-S:
      - register_tokens  (not cls_token)
      - 2D SinCos PE computed from stored frequency_bands (not learned 1D PE)
      - LayerScale (ls1.gamma, ls2.gamma) in every block

    Usage (pretrained):
        model = SparshViTSmall.load_pretrained(npz_path, rngs=nnx.Rngs(0))
        tokens = model(tactile_pair)   # (B, 197, 384)
    """

    def __init__(self, *, rngs: nnx.Rngs):
        # Patch embedding: Conv2D(6 → 384, 16×16 stride 16)
        self.patch_embed_proj = nnx.Conv(
            IN_CHANNELS, EMBED_DIM,
            kernel_size=(PATCH_SIZE, PATCH_SIZE),
            strides=(PATCH_SIZE, PATCH_SIZE),
            padding="VALID",
            use_bias=True,
            rngs=rngs,
        )
        self.register_tokens = nnx.Param(jnp.zeros((1, 1, EMBED_DIM)))
        self.freq_bands      = nnx.Param(jnp.zeros((2, EMBED_DIM // 4)))

        for i in range(DEPTH):
            setattr(self, f"block_{i}", _Block(EMBED_DIM, NUM_HEADS, MLP_HIDDEN, rngs=rngs))

        self.norm = nnx.LayerNorm(EMBED_DIM, epsilon=LN_EPS, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Args:
            x: (B, H, W, 6) float32 in [0, 1].
        Returns:
            (B, 197, 384) — register token at [:, 0, :] + 196 patch tokens.
        """
        B   = x.shape[0]
        grid = IMAGE_SIZE // PATCH_SIZE   # 14

        x = self.patch_embed_proj(x)                  # (B, 14, 14, 384)
        x = x.reshape(B, NUM_PATCHES, EMBED_DIM)      # (B, 196, 384)

        # Add 2D SinCos position embedding (patches only)
        x = x + _sincos2d_from_freq_bands(
            self.freq_bands.value, grid_h=grid, grid_w=grid
        )                                              # (B, 196, 384)

        # Prepend register token
        reg = jnp.broadcast_to(self.register_tokens.value, (B, 1, EMBED_DIM))
        x   = jnp.concatenate([reg, x], axis=1)       # (B, 197, 384)

        for i in range(DEPTH):
            x = getattr(self, f"block_{i}")(x)

        return self.norm(x)                            # (B, 197, 384)

    @classmethod
    def load_pretrained(
        cls,
        npz_path: str | Path,
        *,
        rngs: nnx.Rngs,
        dtype: str = "float32",
    ) -> "SparshViTSmall":
        """Load converted .npz weights produced by convert_sparsh_weights.py."""
        model = cls(rngs=rngs)
        _load_vit_weights(model, npz_path, dtype=dtype)
        return model


# ══════════════════════════════════════════════════════════════════════════════
# Shared weight-loading helper (used by both backbone-only and full encoder)
# ══════════════════════════════════════════════════════════════════════════════

def _load_vit_weights(
    model: SparshViTSmall,
    npz_path: str | Path,
    *,
    dtype: str = "float32",
) -> None:
    """Load .npz pretrained weights into an existing SparshViTSmall instance.

    Modifies `model` in-place.  Raises FileNotFoundError / KeyError on problems.
    """
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Converted weights not found: {npz_path}\n"
            "Run:  python policy/Pi05_openpi_DF/convert_sparsh_weights.py convert"
        )

    w  = np.load(str(npz_path))
    dt = jnp.bfloat16 if dtype == "bfloat16" else jnp.float32

    def _a(key: str) -> jax.Array:
        if key not in w:
            raise KeyError(f"Key '{key}' not found in {npz_path}.  Available: {sorted(w.files)}")
        return jnp.array(w[key], dtype=dt)

    model.patch_embed_proj.kernel.value = _a("patch_embed_proj_kernel")
    model.patch_embed_proj.bias.value   = _a("patch_embed_proj_bias")
    model.register_tokens.value         = _a("register_tokens")
    model.freq_bands.value              = _a("freq_bands")

    for i in range(DEPTH):
        blk = getattr(model, f"block_{i}")
        p   = f"block_{i}"
        blk.norm1.scale.value      = _a(f"{p}_norm1_scale")
        blk.norm1.bias.value       = _a(f"{p}_norm1_bias")
        blk.attn.qkv.kernel.value  = _a(f"{p}_attn_qkv_kernel")
        blk.attn.qkv.bias.value    = _a(f"{p}_attn_qkv_bias")
        blk.attn.proj.kernel.value = _a(f"{p}_attn_proj_kernel")
        blk.attn.proj.bias.value   = _a(f"{p}_attn_proj_bias")
        blk.ls1_gamma.value        = _a(f"{p}_ls1_gamma")
        blk.norm2.scale.value      = _a(f"{p}_norm2_scale")
        blk.norm2.bias.value       = _a(f"{p}_norm2_bias")
        blk.mlp.fc1.kernel.value   = _a(f"{p}_mlp_fc1_kernel")
        blk.mlp.fc1.bias.value     = _a(f"{p}_mlp_fc1_bias")
        blk.mlp.fc2.kernel.value   = _a(f"{p}_mlp_fc2_kernel")
        blk.mlp.fc2.bias.value     = _a(f"{p}_mlp_fc2_bias")
        blk.ls2_gamma.value        = _a(f"{p}_ls2_gamma")

    model.norm.scale.value = _a("norm_scale")
    model.norm.bias.value  = _a("norm_bias")

    n_params = sum(np.prod(w[k].shape) for k in w.files if not k.startswith("__"))
    logger.info(
        "[SparshViTSmall] Loaded %s  (%.1fM params, dtype=%s)",
        npz_path.name, n_params / 1e6, dtype,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Attention Pooling  (197 ViT tokens → num_queries spatial tokens)
# ══════════════════════════════════════════════════════════════════════════════

class AttentionPool(nnx.Module):
    """Perceiver-style cross-attention pooling.

    Compresses a long sequence of encoder tokens into a fixed number of
    query-driven output tokens via multi-head cross-attention:

        Q  =  learned_queries  (1, num_queries, D)
        K  =  k_proj(LayerNorm(encoder_out))
        V  =  v_proj(LayerNorm(encoder_out))
        out = MHA(Q, K, V) + Q   [residual stabilises fine-tuning]
    """

    def __init__(
        self,
        embed_dim:   int,
        num_queries: int,
        num_heads:   int,
        *,
        rngs: nnx.Rngs,
    ):
        assert embed_dim % num_heads == 0
        self.num_queries = num_queries
        self.num_heads   = num_heads
        self.head_dim    = embed_dim // num_heads
        self.scale       = self.head_dim ** -0.5

        self.queries  = nnx.Param(
            jax.random.normal(rngs.params(), (1, num_queries, embed_dim)) * 0.02
        )
        self.q_proj   = nnx.Linear(embed_dim, embed_dim, use_bias=True, rngs=rngs)
        self.k_proj   = nnx.Linear(embed_dim, embed_dim, use_bias=True, rngs=rngs)
        self.v_proj   = nnx.Linear(embed_dim, embed_dim, use_bias=True, rngs=rngs)
        self.out_proj = nnx.Linear(embed_dim, embed_dim, use_bias=True, rngs=rngs)
        self.norm_q   = nnx.LayerNorm(embed_dim, epsilon=LN_EPS, rngs=rngs)
        self.norm_kv  = nnx.LayerNorm(embed_dim, epsilon=LN_EPS, rngs=rngs)

    def __call__(self, encoder_out: jax.Array) -> jax.Array:
        """
        Args:
            encoder_out: (B, S, D) — ViT output tokens (S = 197).
        Returns:
            (B, num_queries, D) pooled tokens.
        """
        B, S, D = encoder_out.shape
        Nq = self.num_queries
        H, Dh = self.num_heads, self.head_dim

        queries = jnp.broadcast_to(self.queries.value, (B, Nq, D))   # (B, Nq, D)

        q  = self.q_proj(self.norm_q(queries))    # (B, Nq, D)
        kv = self.norm_kv(encoder_out)
        k  = self.k_proj(kv)                      # (B, S, D)
        v  = self.v_proj(kv)                      # (B, S, D)

        def _split(t: jax.Array, N: int) -> jax.Array:
            return t.reshape(B, N, H, Dh).transpose(0, 2, 1, 3)   # (B, H, N, Dh)

        q, k, v = _split(q, Nq), _split(k, S), _split(v, S)

        attn = (q @ jnp.swapaxes(k, -2, -1)) * self.scale   # (B, H, Nq, S)
        attn = jax.nn.softmax(attn, axis=-1)

        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, Nq, D)  # (B, Nq, D)
        out = self.out_proj(out)
        return out + queries   # residual


# ══════════════════════════════════════════════════════════════════════════════
# Full encoder: ViT backbone + AttentionPool + Linear proj
# Drop-in replacement for TactileResNetEncoder
# ══════════════════════════════════════════════════════════════════════════════

class SparshTactileEncoder(nnx.Module):
    """Sparsh DINO ViT-Small + Attention Pooling tactile encoder.

    Drop-in replacement for ``TactileResNetEncoder`` — same input/output
    contract, but richer features from a pretrained ViT backbone.

    Input:  (B, H, W, 6)  — current frame (ch 0-2) ‖ prev frame (ch 3-5)
    Output: (B, num_tokens, output_dim)   default num_tokens=16 (4×4 grid)

    Architecture:
        SparshViTSmall  →  (B, 197, 384)
        AttentionPool   →  (B, 16,  384)
        Linear          →  (B, 16,  output_dim)

    The ViT backbone is loaded from pretrained weights; AttentionPool and
    Linear projection are randomly initialised (trained from scratch on robot data).

    Usage:
        encoder = SparshTactileEncoder.from_pretrained(
            npz_path="/data/zjb/ckpts/sparsh/dino/sparsh_dino_small_jax.npz",
            output_dim=1024,
            rngs=nnx.Rngs(0),
        )
        tokens = encoder(x)   # x: (B, 224, 224, 6) → (B, 16, 1024)
    """

    def __init__(
        self,
        output_dim:  int,
        num_tokens:  int = 16,
        pool_heads:  int = 6,
        *,
        rngs: nnx.Rngs,
    ):
        self.num_tokens = num_tokens
        self.output_dim = output_dim

        self.backbone = SparshViTSmall(rngs=rngs)
        self.pool     = AttentionPool(EMBED_DIM, num_tokens, pool_heads, rngs=rngs)
        self.proj     = nnx.Linear(EMBED_DIM, output_dim, use_bias=True, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Args:
            x: (B, H, W, 6) float32 in [0, 1].
        Returns:
            (B, num_tokens, output_dim) token embeddings.
        """
        tokens = self.backbone(x)    # (B, 197, 384)
        tokens = self.pool(tokens)   # (B, num_tokens, 384)
        return self.proj(tokens)     # (B, num_tokens, output_dim)

    @classmethod
    def from_pretrained(
        cls,
        npz_path:   str | Path,
        output_dim: int,
        num_tokens: int = 16,
        pool_heads: int = 6,
        *,
        rngs:  nnx.Rngs,
        dtype: str = "float32",
    ) -> "SparshTactileEncoder":
        """Create encoder with pretrained ViT backbone.

        AttentionPool and proj are randomly initialised (need training).

        Args:
            npz_path:   Path to .npz from convert_sparsh_weights.py.
            output_dim: Action-expert feature width (e.g. 1024).
            num_tokens: Spatial tokens per finger (default 16).
            pool_heads: Attention heads in pooling layer (default 6).
            rngs:       Flax RNG state.
            dtype:      "float32" or "bfloat16".
        """
        encoder = cls(output_dim, num_tokens, pool_heads, rngs=rngs)
        _load_vit_weights(encoder.backbone, npz_path, dtype=dtype)
        logger.info("[SparshTactileEncoder] backbone loaded; pool+proj randomly init.")
        return encoder
