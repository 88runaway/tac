#!/usr/bin/env python3
"""Convert UniVTAC ResNet-18 tactile encoder weights: PyTorch .pth → JAX .npz.

The UniVTAC encoder is a torchvision ResNet-18 trained with multi-task
supervision (RGB/marker/depth/pose reconstruction) on tactile rgb_marker images.
We extract the backbone conv + BatchNorm weights (layer1-4) and discard:
  - backbone.fc.*           (global-vector head, replaced by spatial projection)
  - decoders.*              (reconstruction decoders, not needed)

Transposition rules (PyTorch → JAX):
  Conv2D  kernel:  (O, I, H, W)  →  (H, W, I, O)   [np.transpose(w, (2,3,1,0))]
  BN weight/bias/running: no transposition (channel-wise 1-D vectors)

Output .npz key convention (consumed by TactileUniVTACEncoder.from_pretrained):
  stem_conv_kernel            (7, 7, 3, 64)
  stem_bn_{scale,bias,mean,var}  (64,)
  layer{L}_block{B}_conv{1,2}_kernel          (kH, kW, Cin, Cout)
  layer{L}_block{B}_conv{1,2}_bn_{scale,...}  (Cout,)
  layer{L}_block0_proj_kernel                 (1, 1, Cin, Cout)   [layer2-4 only]
  layer{L}_block0_proj_bn_{scale,...}         (Cout,)

Usage:
    # Requires torch + torchvision (use a PyTorch conda environment, e.g. tac)
    conda activate tac   # or any env that has torch

    cd /data/zjb/UniVTAC

    # Inspect source keys (optional)
    python policy/Pi05_openpi_DF/convert_univtac_encoder_weights.py inspect

    # Convert + validate shapes
    python policy/Pi05_openpi_DF/convert_univtac_encoder_weights.py convert

    # Custom paths
    python policy/Pi05_openpi_DF/convert_univtac_encoder_weights.py convert \\
        --src /data/zjb/ckpts/univtac_encoder/checkpoints/encoder.pth \\
        --dst /data/zjb/ckpts/univtac_encoder/univtac_resnet18_jax.npz
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_SRC = "/data/zjb/ckpts/univtac_encoder/checkpoints/encoder.pth"
DEFAULT_DST = "/data/zjb/ckpts/univtac_encoder/univtac_resnet18_jax.npz"

# Expected shapes (Cin, Cout) — used for validation
# (layer, block, conv_idx): (in_ch, out_ch, kernel)
CONV_SPEC = {
    # stem
    "stem": (3, 64, 7),
    # layer1 — no downsample
    ("layer1", 0, "conv1"): (64, 64, 3),
    ("layer1", 0, "conv2"): (64, 64, 3),
    ("layer1", 1, "conv1"): (64, 64, 3),
    ("layer1", 1, "conv2"): (64, 64, 3),
    # layer2 — block 0 has downsample
    ("layer2", 0, "conv1"): (64,  128, 3),
    ("layer2", 0, "conv2"): (128, 128, 3),
    ("layer2", 0, "proj"):  (64,  128, 1),
    ("layer2", 1, "conv1"): (128, 128, 3),
    ("layer2", 1, "conv2"): (128, 128, 3),
    # layer3 — block 0 has downsample
    ("layer3", 0, "conv1"): (128, 256, 3),
    ("layer3", 0, "conv2"): (256, 256, 3),
    ("layer3", 0, "proj"):  (128, 256, 1),
    ("layer3", 1, "conv1"): (256, 256, 3),
    ("layer3", 1, "conv2"): (256, 256, 3),
    # layer4 — block 0 has downsample
    ("layer4", 0, "conv1"): (256, 512, 3),
    ("layer4", 0, "conv2"): (512, 512, 3),
    ("layer4", 0, "proj"):  (256, 512, 1),
    ("layer4", 1, "conv1"): (512, 512, 3),
    ("layer4", 1, "conv2"): (512, 512, 3),
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. Load PyTorch checkpoint
# ══════════════════════════════════════════════════════════════════════════════

def _load_pth(path: str | Path) -> dict[str, np.ndarray]:
    """Load a PyTorch .pth state_dict and return as {key: numpy_array}."""
    try:
        import torch
    except ImportError:
        log.error(
            "PyTorch is required for this script.\n"
            "Activate a PyTorch conda environment, e.g.:\n"
            "  conda activate tac\n"
            "  python policy/Pi05_openpi_DF/convert_univtac_encoder_weights.py convert"
        )
        sys.exit(1)

    log.info("Loading %s …", path)
    sd = torch.load(str(path), map_location="cpu", weights_only=True)
    # state_dict may be wrapped inside a dict (e.g. {'model': sd})
    if isinstance(sd, dict) and "model" in sd and not any(
        k.startswith("backbone") for k in sd
    ):
        sd = sd["model"]
    out = {}
    for k, v in sd.items():
        try:
            out[k] = v.numpy()
        except Exception:
            out[k] = v.float().numpy()
    log.info("  → %d keys loaded.", len(out))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 2. Build NPZ dict
# ══════════════════════════════════════════════════════════════════════════════

def _conv_kernel(w: dict, pt_key: str) -> np.ndarray:
    """Fetch a conv weight and transpose (O,I,H,W) → (H,W,I,O)."""
    arr = w[pt_key]
    assert arr.ndim == 4, f"Expected 4-D conv weight at '{pt_key}', got {arr.shape}"
    return np.transpose(arr, (2, 3, 1, 0)).astype(np.float32)


def _bn_fields(w: dict, pt_prefix: str, dst_prefix: str) -> dict[str, np.ndarray]:
    """Extract BN {scale, bias, mean, var} from the PyTorch state dict."""
    return {
        f"{dst_prefix}_scale": w[f"{pt_prefix}.weight"].astype(np.float32),
        f"{dst_prefix}_bias":  w[f"{pt_prefix}.bias"].astype(np.float32),
        f"{dst_prefix}_mean":  w[f"{pt_prefix}.running_mean"].astype(np.float32),
        f"{dst_prefix}_var":   w[f"{pt_prefix}.running_var"].astype(np.float32),
    }


def _build_npz(w: dict) -> dict[str, np.ndarray]:
    """Convert a PyTorch state dict to the NPZ key convention."""
    out: dict[str, np.ndarray] = {}

    # ── Stem ────────────────────────────────────────────────────────────────────
    out["stem_conv_kernel"] = _conv_kernel(w, "backbone.conv1.weight")
    out.update(_bn_fields(w, "backbone.bn1", "stem_bn"))

    # ── ResNet layers 1-4 ───────────────────────────────────────────────────────
    layer_cfgs = [
        ("layer1", 2, False),
        ("layer2", 2, True),
        ("layer3", 2, True),
        ("layer4", 2, True),
    ]
    for layer_name, num_blocks, has_proj in layer_cfgs:
        for b in range(num_blocks):
            pt_blk = f"backbone.{layer_name}.{b}"
            dst_pfx = f"{layer_name}_block{b}"

            out[f"{dst_pfx}_conv1_kernel"] = _conv_kernel(w, f"{pt_blk}.conv1.weight")
            out.update(_bn_fields(w, f"{pt_blk}.bn1", f"{dst_pfx}_conv1_bn"))

            out[f"{dst_pfx}_conv2_kernel"] = _conv_kernel(w, f"{pt_blk}.conv2.weight")
            out.update(_bn_fields(w, f"{pt_blk}.bn2", f"{dst_pfx}_conv2_bn"))

            if b == 0 and has_proj:
                out[f"{dst_pfx}_proj_kernel"] = _conv_kernel(
                    w, f"{pt_blk}.downsample.0.weight"
                )
                out.update(_bn_fields(w, f"{pt_blk}.downsample.1", f"{dst_pfx}_proj_bn"))

    # ── Global fc head (backbone.fc, for stack_fc=True mode) ────────────────────
    # PyTorch Linear weight shape: (out_features, in_features) = (512, 512)
    # JAX NNX Linear kernel shape: (in_features, out_features) = (512, 512) — transpose
    out["fc_kernel"] = w["backbone.fc.weight"].T.astype(np.float32)   # (512, 512)
    out["fc_bias"]   = w["backbone.fc.bias"].astype(np.float32)        # (512,)

    return out


# ══════════════════════════════════════════════════════════════════════════════
# 3. Shape validation
# ══════════════════════════════════════════════════════════════════════════════

def _validate(npz: dict[str, np.ndarray]) -> bool:
    ok = True

    def _check(key: str, expected: tuple):
        nonlocal ok
        if key not in npz:
            log.error("  MISSING key: %s", key)
            ok = False
            return
        got = npz[key].shape
        if got != expected:
            log.error("  SHAPE MISMATCH %s: expected %s, got %s", key, expected, got)
            ok = False
        else:
            log.info("  ✓  %-45s  %s", key, got)

    # stem
    _check("stem_conv_kernel", (7, 7, 3, 64))
    for s in ("scale", "bias", "mean", "var"):
        _check(f"stem_bn_{s}", (64,))

    # layers
    ch = {1: 64, 2: 128, 3: 256, 4: 512}
    in_ch = {1: 64, 2: 64, 3: 128, 4: 256}
    for li in range(1, 5):
        C = ch[li]
        for b in range(2):
            pfx = f"layer{li}_block{b}"
            cin = in_ch[li] if b == 0 else C
            _check(f"{pfx}_conv1_kernel", (3, 3, cin, C))
            _check(f"{pfx}_conv2_kernel", (3, 3, C, C))
            for which in ("conv1", "conv2"):
                for s in ("scale", "bias", "mean", "var"):
                    _check(f"{pfx}_{which}_bn_{s}", (C,))
            if b == 0 and li > 1:
                cin_proj = in_ch[li]
                _check(f"{pfx}_proj_kernel", (1, 1, cin_proj, C))
                for s in ("scale", "bias", "mean", "var"):
                    _check(f"{pfx}_proj_bn_{s}", (C,))

    # global fc head
    _check("fc_kernel", (512, 512))
    _check("fc_bias",   (512,))

    return ok


# ══════════════════════════════════════════════════════════════════════════════
# 4. Commands
# ══════════════════════════════════════════════════════════════════════════════

def cmd_inspect(args):
    w = _load_pth(args.src)
    print(f"\n{'Key':<60}  {'Shape'}")
    print("-" * 80)
    for k, v in sorted(w.items()):
        shape = v.shape if hasattr(v, "shape") else type(v).__name__
        print(f"{k:<60}  {shape}")


def cmd_convert(args):
    w   = _load_pth(args.src)
    npz = _build_npz(w)

    log.info("\n── Validating shapes ──────────────────────────────────────────")
    ok = _validate(npz)
    if not ok:
        log.error("Validation failed — aborting.")
        sys.exit(1)

    dst = Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(dst), **npz)
    log.info("\n✓  Saved %d arrays to %s  (%.1f MB)", len(npz), dst, dst.stat().st_size / 1e6)
    log.info(
        "Next steps:\n"
        "  1. Set tactile_encoder_type: 'univtac' in your training YAML\n"
        "  2. Set univtac_encoder_path: '%s'\n"
        "  3. Run training: python policy/Pi05_openpi_DF/train_df.py --task <task> --gpu 0",
        dst,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 5. CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Convert UniVTAC ResNet-18 encoder: PyTorch .pth → JAX .npz",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_inspect = sub.add_parser("inspect", help="Print all keys in the .pth file")
    p_inspect.add_argument("--src", default=DEFAULT_SRC, help="Source .pth path")

    p_convert = sub.add_parser("convert", help="Convert .pth → .npz")
    p_convert.add_argument("--src", default=DEFAULT_SRC, help="Source .pth path")
    p_convert.add_argument("--dst", default=DEFAULT_DST, help="Destination .npz path")

    args = parser.parse_args()
    if args.command == "inspect":
        cmd_inspect(args)
    elif args.command == "convert":
        cmd_convert(args)


if __name__ == "__main__":
    main()
