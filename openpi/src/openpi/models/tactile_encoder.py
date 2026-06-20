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

    Parameters
    ----------
    output_dim : int
        Dimension of each output token (typically the action-expert width,
        e.g. 1024 for ``gemma_300m``).
    num_tokens : int
        Number of spatial tokens per image.  Default 16 (4×4 grid).
    """

    def __init__(self, output_dim: int, num_tokens: int = 16, *, rngs: nnx.Rngs):
        self.output_dim = output_dim
        self.num_tokens = num_tokens
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

    @at.typecheck
    def __call__(
        self,
        x: at.Float[at.Array, "b h w 3"],
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "b n d"]:
        """Encode a batch of tactile images into spatial tokens.

        Args:
            x: ``(B, H, W, 3)``  float32 images in ``[0, 1]``.
            train: Unused (kept for API parity); GroupNorm has no train/eval modes.

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

        # Flatten spatial → tokens
        x = x.reshape(b, self.num_tokens, c)   # (B, 16, 512)
        x = self.proj(x)                       # (B, 16, output_dim)
        return x
