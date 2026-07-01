"""ResNet-18 tactile encoder in Flax NNX.

Takes a single tactile ``rgb_marker`` image ``(B, H, W, 3)`` and produces
``num_tokens`` feature vectors ``(B, num_tokens, output_dim)`` that can be
concatenated with the action expert tokens in Pi0DF.

Architecture (reference: ``UniVTAC/T-Rex/qwen_vla/DeformAE.py`` and
``UniVTAC/policy/encoder/network.py``):

    Stem  (Conv7x7-s2 → GN → ReLU → MaxPool-s2) → 56×56
    Layer1  (2 × BasicBlock, 64  ch)              → 56×56
    Layer2  (2 × BasicBlock, 128 ch, stride 2)     → 28×28
    Layer3  (2 × BasicBlock, 256 ch, stride 2)     → 14×14
    Layer4  (2 × BasicBlock, 512 ch, stride 2)     → 7×7
    Pad 7→8, AvgPool 2×2                           → 4×4
    Reshape                                        → 16 × 512
    Linear                                         → 16 × output_dim

For both left and right fingers this gives 32 tokens total.
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.shared import array_typing as at


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _num_groups(ch: int, max_groups: int = 32) -> int:
    """Largest group count <= max_groups that divides ``ch`` (fallback 1)."""
    for g in range(min(max_groups, ch), 0, -1):
        if ch % g == 0:
            return g
    return 1


class _ConvGN(nnx.Module):
    """Conv → GroupNorm (no activation).

    GroupNorm is used instead of BatchNorm: it normalizes per-sample (no batch
    dependence, no running statistics), so it behaves identically in training and
    inference and is stable at batch size 1 and under the functional nnx loop.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int,
        stride: int = 1,
        padding: int | str = 0,
        *,
        rngs: nnx.Rngs,
    ):
        if isinstance(padding, int):
            pad = ((padding, padding), (padding, padding))
        else:
            pad = padding
        self.conv = nnx.Conv(
            in_ch, out_ch,
            kernel_size=(kernel, kernel),
            strides=(stride, stride),
            padding=pad,
            use_bias=False,
            rngs=rngs,
        )
        self.gn = nnx.GroupNorm(out_ch, num_groups=_num_groups(out_ch), epsilon=1e-5, rngs=rngs)

    def __call__(self, x, *, train: bool = False):  # noqa: ARG002 (train kept for API parity)
        return self.gn(self.conv(x))


class _BasicBlock(nnx.Module):
    """ResNet BasicBlock (two 3×3 convolutions with skip connection)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, *, rngs: nnx.Rngs):
        self.conv1 = _ConvGN(in_ch, out_ch, 3, stride=stride, padding=1, rngs=rngs)
        self.conv2 = _ConvGN(out_ch, out_ch, 3, stride=1, padding=1, rngs=rngs)
        self.need_proj = (stride != 1 or in_ch != out_ch)
        if self.need_proj:
            self.proj = _ConvGN(in_ch, out_ch, 1, stride=stride, padding=0, rngs=rngs)

    def __call__(self, x, *, train: bool = False):
        identity = x
        out = nnx.relu(self.conv1(x, train=train))
        out = self.conv2(out, train=train)
        if self.need_proj:
            identity = self.proj(x, train=train)
        return nnx.relu(out + identity)


class _ResLayer(nnx.Module):
    """A ResNet stage of ``num_blocks`` BasicBlocks.

    Blocks are stored as string-named attributes (``block_0``, ``block_1``, ...)
    rather than a Python list so the nnx param tree uses string keys (integer
    keys break ``flax.traverse_util.flatten_dict(..., sep="/")`` in the openpi
    weight loader).
    """

    def __init__(self, in_ch: int, out_ch: int, num_blocks: int, stride: int, *, rngs: nnx.Rngs):
        self.num_blocks = num_blocks
        setattr(self, "block_0", _BasicBlock(in_ch, out_ch, stride=stride, rngs=rngs))
        for i in range(1, num_blocks):
            setattr(self, f"block_{i}", _BasicBlock(out_ch, out_ch, stride=1, rngs=rngs))

    def __call__(self, x, *, train: bool = False):
        for i in range(self.num_blocks):
            x = getattr(self, f"block_{i}")(x, train=train)
        return x


# ---------------------------------------------------------------------------
# Main encoder
# ---------------------------------------------------------------------------

class TactileResNetEncoder(nnx.Module):
    """ResNet-18 backbone that maps a tactile image to spatial tokens.

    Positional encodings (spatial + finger identity) match those of
    ``SparshTactileEncoder`` so both encoders produce tokens in the same
    embedding space convention.

    Parameters
    ----------
    output_dim : int
        Dimension of each output token (typically the action-expert width,
        e.g. 1024 for ``gemma_300m``).
    num_tokens : int
        Number of spatial tokens per image.  Default 16 (4×4 grid).
    """

    def __init__(
        self,
        output_dim: int,
        num_tokens: int = 16,
        use_pos_emb: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        self.output_dim = output_dim
        self.num_tokens = num_tokens
        self.use_pos_emb = use_pos_emb
        self._grid_h = self._grid_w = int(num_tokens ** 0.5)
        assert self._grid_h * self._grid_w == num_tokens

        # Stem
        self.stem_conv = nnx.Conv(
            3, 64, kernel_size=(7, 7), strides=(2, 2),
            padding=((3, 3), (3, 3)), use_bias=False, rngs=rngs,
        )
        self.stem_gn = nnx.GroupNorm(64, num_groups=_num_groups(64), epsilon=1e-5, rngs=rngs)

        # ResNet layers (2 blocks each, standard ResNet-18)
        self.layer1 = _ResLayer(64,  64,  2, stride=1, rngs=rngs)
        self.layer2 = _ResLayer(64,  128, 2, stride=2, rngs=rngs)
        self.layer3 = _ResLayer(128, 256, 2, stride=2, rngs=rngs)
        self.layer4 = _ResLayer(256, 512, 2, stride=2, rngs=rngs)

        # Project 512-d spatial features → output_dim
        self.proj = nnx.Linear(512, output_dim, rngs=rngs)

        # Learned positional encodings (same convention as SparshTactileEncoder).
        # Only created when use_pos_emb=True; omitted entirely when False (ablation).
        if use_pos_emb:
            self.spatial_emb = nnx.Param(jnp.zeros((num_tokens, output_dim)))
            self.finger_emb  = nnx.Param(jnp.zeros((2, output_dim)))

    @at.typecheck
    def __call__(
        self,
        x: at.Float[at.Array, "b h w 3"],
        finger_idx: int = 0,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "b n d"]:
        """Encode a batch of tactile images into spatial tokens.

        Args:
            x:          ``(B, H, W, 3)`` float32 images in ``[0, 1]``.
            finger_idx: 0 = left finger, 1 = right finger.
            train:      Unused (GroupNorm has no train/eval modes; kept for API parity).

        Returns:
            ``(B, num_tokens, output_dim)`` token embeddings.
        """
        # Stem: Conv → GroupNorm → ReLU → MaxPool
        x = self.stem_conv(x)
        x = self.stem_gn(x)
        x = nnx.relu(x)
        # MaxPool 3×3 stride 2 padding 1  (NHWC)
        x = jax.lax.reduce_window(
            x, -jnp.inf, jax.lax.max,
            window_dimensions=(1, 3, 3, 1),
            window_strides=(1, 2, 2, 1),
            padding=((0, 0), (1, 1), (1, 1), (0, 0)),
        )

        # ResNet body
        x = self.layer1(x, train=train)
        x = self.layer2(x, train=train)
        x = self.layer3(x, train=train)
        x = self.layer4(x, train=train)
        # x: (B, 7, 7, 512)  for 224×224 input

        # Adaptive avg-pool → (grid_h, grid_w)
        b, h, w, c = x.shape
        x = jax.image.resize(x, (b, self._grid_h, self._grid_w, c), method="linear")

        # Flatten spatial → tokens, then project
        x = x.reshape(b, self.num_tokens, c)   # (B, 16, 512)
        tokens = self.proj(x)                  # (B, 16, output_dim)

        if self.use_pos_emb:
            # Spatial position encoding (4×4 grid, row-major)
            tokens = tokens + self.spatial_emb.value          # (16, D) broadcast over B
            # Finger identity encoding (left=0 / right=1)
            tokens = tokens + self.finger_emb.value[finger_idx]  # (D,) broadcast

        return tokens


# ============================================================================
# UniVTAC pre-trained ResNet-18 encoder (BatchNorm, matches original training)
# ============================================================================

class _ConvBN(nnx.Module):
    """Conv → BatchNorm (no activation).

    Uses BatchNorm instead of GroupNorm to match the torchvision ResNet-18
    architecture used in the UniVTAC pre-training pipeline.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int,
        stride: int = 1,
        padding: int | str = 0,
        *,
        rngs: nnx.Rngs,
    ):
        if isinstance(padding, int):
            pad = ((padding, padding), (padding, padding))
        else:
            pad = padding
        self.conv = nnx.Conv(
            in_ch, out_ch,
            kernel_size=(kernel, kernel),
            strides=(stride, stride),
            padding=pad,
            use_bias=False,
            rngs=rngs,
        )
        self.bn = nnx.BatchNorm(out_ch, momentum=0.1, epsilon=1e-5, rngs=rngs)

    def __call__(self, x, *, train: bool = False):
        return self.bn(self.conv(x), use_running_average=not train)


class _BasicBlockBN(nnx.Module):
    """ResNet BasicBlock with BatchNorm (two 3×3 convolutions + skip connection)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, *, rngs: nnx.Rngs):
        self.conv1 = _ConvBN(in_ch, out_ch, 3, stride=stride, padding=1, rngs=rngs)
        self.conv2 = _ConvBN(out_ch, out_ch, 3, stride=1, padding=1, rngs=rngs)
        self.need_proj = (stride != 1 or in_ch != out_ch)
        if self.need_proj:
            self.proj = _ConvBN(in_ch, out_ch, 1, stride=stride, padding=0, rngs=rngs)

    def __call__(self, x, *, train: bool = False):
        identity = x
        out = nnx.relu(self.conv1(x, train=train))
        out = self.conv2(out, train=train)
        if self.need_proj:
            identity = self.proj(x, train=train)
        return nnx.relu(out + identity)


class _ResLayerBN(nnx.Module):
    """A ResNet stage of ``num_blocks`` BasicBlocks with BatchNorm.

    Blocks are stored as string-named attributes (``block_0``, ``block_1``, ...)
    for compatibility with the openpi weight-loader flatten convention.
    """

    def __init__(self, in_ch: int, out_ch: int, num_blocks: int, stride: int, *, rngs: nnx.Rngs):
        self.num_blocks = num_blocks
        setattr(self, "block_0", _BasicBlockBN(in_ch, out_ch, stride=stride, rngs=rngs))
        for i in range(1, num_blocks):
            setattr(self, f"block_{i}", _BasicBlockBN(out_ch, out_ch, stride=1, rngs=rngs))

    def __call__(self, x, *, train: bool = False):
        for i in range(self.num_blocks):
            x = getattr(self, f"block_{i}")(x, train=train)
        return x


class TactileUniVTACEncoder(nnx.Module):
    """ResNet-18 tactile encoder with BatchNorm, initialized from UniVTAC pre-trained weights.

    Architecture is identical to torchvision ResNet-18 up through layer4 (BatchNorm
    throughout), then replaced the global-average-pool + FC head with a 4×4 spatial
    pool → linear projection to produce ``num_tokens`` spatial token embeddings per
    finger.  This matches the UniVTAC training setup so pre-trained conv weights
    transfer faithfully.

    Typical usage (spatial 16-token mode)::

        encoder = TactileUniVTACEncoder.from_pretrained(
            npz_path="/data/zjb/ckpts/univtac_encoder/univtac_resnet18_jax.npz",
            output_dim=1024,
            rngs=nnx.Rngs(0),
        )
        tokens = encoder(img, finger_idx=0)   # (B, 16, 1024)

    Stacked-fc global-token mode (stack_fc=True, num_tokens=1)::

        encoder = TactileUniVTACEncoder.from_pretrained(
            npz_path="...",
            output_dim=1024,
            num_tokens=1,
            stack_fc=True,
            rngs=nnx.Rngs(0),
        )
        tokens = encoder(img, finger_idx=0)   # (B, 1, 1024)

    Parameters
    ----------
    output_dim : int
        Width of each output token (action-expert width, e.g. 1024 for gemma_300m).
    num_tokens : int
        Number of tokens per image.  Default 16 (4×4 spatial grid).
        Must be 1 when ``stack_fc=True``.
    use_pos_emb : bool
        Whether to add learned finger-identity (and spatial, if num_tokens>1)
        positional embeddings.
    stack_fc : bool
        When True, use the original UniVTAC global fc head (GlobalAvgPool →
        pretrained fc 512→512 → new proj 512→output_dim) instead of the
        spatial-resize path.  Requires num_tokens=1.  The ``fc`` weights are
        loaded from the converted .npz; ``proj`` is randomly initialised.
    """

    def __init__(
        self,
        output_dim: int,
        num_tokens: int = 16,
        use_pos_emb: bool = True,
        stack_fc: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.output_dim = output_dim
        self.num_tokens = num_tokens
        self.use_pos_emb = use_pos_emb
        self.stack_fc = stack_fc

        if stack_fc:
            assert num_tokens == 1, (
                "TactileUniVTACEncoder: stack_fc=True requires num_tokens=1, "
                f"got num_tokens={num_tokens}"
            )
        else:
            self._grid_h = self._grid_w = int(num_tokens ** 0.5)
            assert self._grid_h * self._grid_w == num_tokens, (
                f"num_tokens={num_tokens} must be a perfect square"
            )

        # Stem: Conv7×7 → BN → (ReLU + MaxPool applied in __call__)
        self.stem_conv = nnx.Conv(
            3, 64,
            kernel_size=(7, 7), strides=(2, 2),
            padding=((3, 3), (3, 3)), use_bias=False, rngs=rngs,
        )
        self.stem_bn = nnx.BatchNorm(64, momentum=0.1, epsilon=1e-5, rngs=rngs)

        # ResNet-18 body (BatchNorm, 2 blocks each — matches torchvision exactly)
        self.layer1 = _ResLayerBN(64,  64,  2, stride=1, rngs=rngs)
        self.layer2 = _ResLayerBN(64,  128, 2, stride=2, rngs=rngs)
        self.layer3 = _ResLayerBN(128, 256, 2, stride=2, rngs=rngs)
        self.layer4 = _ResLayerBN(256, 512, 2, stride=2, rngs=rngs)

        if stack_fc:
            # Pretrained global fc (512→512): loaded from UniVTAC checkpoint
            self.fc   = nnx.Linear(512, 512, use_bias=True, rngs=rngs)
            # New random projection (512→output_dim): fine-tuned with policy
            self.proj = nnx.Linear(512, output_dim, rngs=rngs)
        else:
            # Spatial projection head: 512→output_dim (random init, no pretrained fc)
            self.proj = nnx.Linear(512, output_dim, rngs=rngs)

        # Learned positional encodings (zero-init)
        if use_pos_emb:
            # spatial_emb: (num_tokens, D) — one vector per grid cell
            # For num_tokens=1 this degenerates to a single learned bias per finger,
            # which is still harmless and keeps the API uniform.
            self.spatial_emb = nnx.Param(jnp.zeros((num_tokens, output_dim)))
            self.finger_emb  = nnx.Param(jnp.zeros((2, output_dim)))

    @at.typecheck
    def __call__(
        self,
        x: at.Float[at.Array, "b h w 3"],
        finger_idx: int = 0,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "b n d"]:
        """Encode a batch of tactile images into token embeddings.

        Args:
            x:          ``(B, H, W, 3)`` float32 images in ``[0, 1]``.
            finger_idx: 0 = left finger, 1 = right finger.
            train:      If True, use batch statistics (BatchNorm training mode).
                        If False (default), use running statistics (eval mode).

        Returns:
            ``(B, num_tokens, output_dim)`` token embeddings.
            When stack_fc=True, num_tokens=1 (global token).
        """
        # Stem: Conv → BN → ReLU → MaxPool
        x = self.stem_bn(self.stem_conv(x), use_running_average=not train)
        x = nnx.relu(x)
        x = jax.lax.reduce_window(
            x, -jnp.inf, jax.lax.max,
            window_dimensions=(1, 3, 3, 1),
            window_strides=(1, 2, 2, 1),
            padding=((0, 0), (1, 1), (1, 1), (0, 0)),
        )

        # ResNet body
        x = self.layer1(x, train=train)
        x = self.layer2(x, train=train)
        x = self.layer3(x, train=train)
        x = self.layer4(x, train=train)
        # x: (B, 7, 7, 512) for 224×224 input

        if self.stack_fc:
            # ── Global-token path (stack_fc=True, num_tokens=1) ──────────────
            # GlobalAvgPool → pretrained fc (512→512) → new proj (512→output_dim)
            b = x.shape[0]
            x = x.mean(axis=(1, 2))       # (B, 512)  global average pool
            x = self.fc(x)                # (B, 512)  pretrained fc, no activation
            tokens = self.proj(x)         # (B, output_dim)
            tokens = tokens[:, None, :]   # (B, 1, output_dim)
        else:
            # ── Spatial-token path (stack_fc=False, num_tokens=grid²) ────────
            b, h, w, c = x.shape
            x = jax.image.resize(x, (b, self._grid_h, self._grid_w, c), method="linear")
            x = x.reshape(b, self.num_tokens, c)  # (B, N, 512)
            tokens = self.proj(x)                 # (B, N, output_dim)

        if self.use_pos_emb:
            tokens = tokens + self.spatial_emb.value            # (N, D) broadcast over B
            tokens = tokens + self.finger_emb.value[finger_idx] # (D,) broadcast

        return tokens

    @classmethod
    def from_pretrained(
        cls,
        npz_path: str,
        output_dim: int,
        num_tokens: int = 16,
        use_pos_emb: bool = True,
        stack_fc: bool = False,
        *,
        rngs: nnx.Rngs,
        dtype: str = "float32",
    ) -> "TactileUniVTACEncoder":
        """Create encoder with UniVTAC pre-trained ResNet-18 backbone.

        Loads conv + BN backbone weights (and, when stack_fc=True, the pretrained
        fc weights) from the converted .npz file.
        The ``proj`` head and (if use_pos_emb) ``spatial_emb`` + ``finger_emb``
        are randomly / zero-initialised and must be fine-tuned on robot data.

        Args:
            npz_path:   Path to .npz produced by convert_univtac_encoder_weights.py.
            output_dim: Action-expert feature width (e.g. 1024 for gemma_300m).
            num_tokens: Tokens per finger.  Use 16 (spatial) or 1 (global).
                        Must be 1 when stack_fc=True.
            use_pos_emb: If True (default), add spatial_emb + finger_emb.
            stack_fc:   If True, use GlobalAvgPool → pretrained fc (512→512) →
                        new proj (512→output_dim).  Requires num_tokens=1.
                        The fc weights are loaded from the .npz.
            rngs:       Flax RNG state.
            dtype:      "float32" or "bfloat16".
        """
        import logging
        from pathlib import Path

        logger = logging.getLogger(__name__)

        npz_path = Path(npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(
                f"Converted weights not found: {npz_path}\n"
                "Run:  python policy/Pi05_openpi_DF/convert_univtac_encoder_weights.py convert"
            )

        encoder = cls(
            output_dim, num_tokens,
            use_pos_emb=use_pos_emb,
            stack_fc=stack_fc,
            rngs=rngs,
        )
        _load_univtac_weights(encoder, npz_path, dtype=dtype)
        mode = "stack_fc (GlobalAvgPool→fc→proj, 1 token)" if stack_fc else f"spatial ({num_tokens} tokens)"
        logger.info(
            "[TactileUniVTACEncoder] loaded from %s | mode=%s | proj+pos_emb random/zero-init",
            npz_path, mode,
        )
        return encoder


def _load_univtac_weights(
    model: TactileUniVTACEncoder,
    npz_path,
    *,
    dtype: str = "float32",
) -> None:
    """Load converted .npz weights into an existing TactileUniVTACEncoder.

    Modifies ``model`` in-place.

    Always loaded (conv + BN backbone):
        stem_conv, stem_bn, layer{1-4}/block_{0,1}/conv{1,2} + BN + proj BN

    Loaded only when model.stack_fc=True:
        fc (pretrained global head 512→512)

    Never loaded (randomly / zero initialised, trained with policy):
        proj (512→output_dim), spatial_emb, finger_emb
    """
    import numpy as np
    from pathlib import Path

    w  = np.load(str(npz_path))
    dt = jnp.bfloat16 if dtype == "bfloat16" else jnp.float32

    def _a(key: str) -> jax.Array:
        if key not in w:
            raise KeyError(
                f"Key '{key}' not found in {npz_path}.\n"
                f"Available: {sorted(w.files)}"
            )
        return jnp.array(w[key], dtype=dt)

    # ── Stem ────────────────────────────────────────────────────────────────────
    model.stem_conv.kernel.value = _a("stem_conv_kernel")
    model.stem_bn.scale.value    = _a("stem_bn_scale")
    model.stem_bn.bias.value     = _a("stem_bn_bias")
    model.stem_bn.mean.value     = _a("stem_bn_mean")
    model.stem_bn.var.value      = _a("stem_bn_var")

    # ── ResNet layers 1-4, blocks 0-1 ──────────────────────────────────────────
    # layer_cfg: (layer_name, num_blocks, has_downsample_on_block0)
    layer_cfgs = [
        ("layer1", 2, False),
        ("layer2", 2, True),
        ("layer3", 2, True),
        ("layer4", 2, True),
    ]
    for layer_name, num_blocks, has_proj in layer_cfgs:
        layer = getattr(model, layer_name)
        for b in range(num_blocks):
            blk = getattr(layer, f"block_{b}")
            pfx = f"{layer_name}_block{b}"

            blk.conv1.conv.kernel.value = _a(f"{pfx}_conv1_kernel")
            blk.conv1.bn.scale.value    = _a(f"{pfx}_conv1_bn_scale")
            blk.conv1.bn.bias.value     = _a(f"{pfx}_conv1_bn_bias")
            blk.conv1.bn.mean.value     = _a(f"{pfx}_conv1_bn_mean")
            blk.conv1.bn.var.value      = _a(f"{pfx}_conv1_bn_var")

            blk.conv2.conv.kernel.value = _a(f"{pfx}_conv2_kernel")
            blk.conv2.bn.scale.value    = _a(f"{pfx}_conv2_bn_scale")
            blk.conv2.bn.bias.value     = _a(f"{pfx}_conv2_bn_bias")
            blk.conv2.bn.mean.value     = _a(f"{pfx}_conv2_bn_mean")
            blk.conv2.bn.var.value      = _a(f"{pfx}_conv2_bn_var")

            if b == 0 and has_proj:
                blk.proj.conv.kernel.value = _a(f"{pfx}_proj_kernel")
                blk.proj.bn.scale.value    = _a(f"{pfx}_proj_bn_scale")
                blk.proj.bn.bias.value     = _a(f"{pfx}_proj_bn_bias")
                blk.proj.bn.mean.value     = _a(f"{pfx}_proj_bn_mean")
                blk.proj.bn.var.value      = _a(f"{pfx}_proj_bn_var")

    # ── Pretrained global fc head (only when stack_fc=True) ─────────────────────
    if model.stack_fc:
        model.fc.kernel.value = _a("fc_kernel")
        model.fc.bias.value   = _a("fc_bias")
