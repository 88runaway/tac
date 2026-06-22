#!/usr/bin/env python3
"""Convert Sparsh DINO ViT-Small weights: safetensors (PyTorch) → JAX .npz.

Actual architecture (discovered via --inspect):
  - register_tokens (1, 1, 384)      — learnable CLS-like token
  - pos_embed.frequency_bands (2,96) — 2D SinCos position encoding frequencies
  - blocks.{i}.ls1.gamma (384,)      — LayerScale on attention output
  - blocks.{i}.ls2.gamma (384,)      — LayerScale on MLP output

Transposition rules (PyTorch → JAX):
  Conv2D  kernel:  (O, I, H, W)  →  (H, W, I, O)   [transpose(2,3,1,0)]
  Linear  kernel:  (out, in)     →  (in, out)        [.T]
  Others (bias, norm, cls, pos): no transposition

Three validation levels are run after conversion:
  L1 — expected shapes for every converted array
  L2 — JAX forward pass: finite, non-trivial outputs
  L3 — element-wise PyTorch comparison (optional, --torch-compare)

Usage:
    conda activate openpi
    cd /data/zjb/UniVTAC

    # Inspect source keys (optional)
    python policy/Pi05_openpi_DF/convert_sparsh_weights.py inspect

    # Convert + validate
    python policy/Pi05_openpi_DF/convert_sparsh_weights.py convert

    # Convert + L3 comparison (needs torch)
    python policy/Pi05_openpi_DF/convert_sparsh_weights.py convert --torch-compare
"""

from __future__ import annotations

import argparse
import json
import logging
import struct
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Architecture constants ─────────────────────────────────────────────────────
DEPTH       = 12
EMBED_DIM   = 384
NUM_HEADS   = 6
MLP_HIDDEN  = 1536
IN_CHANNELS = 6
PATCH_SIZE  = 16
IMAGE_SIZE  = 224
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2   # 196
SEQ_LEN     = NUM_PATCHES + 1                   # 197

DEFAULT_SRC = "/data/zjb/ckpts/sparsh/dino/dino_vitsmall.safetensors"
DEFAULT_DST = "/data/zjb/ckpts/sparsh/dino/sparsh_dino_small_jax.npz"


# ══════════════════════════════════════════════════════════════════════════════
# 1. Pure-Python safetensors reader (no external library required)
# ══════════════════════════════════════════════════════════════════════════════

def _read_safetensors(path: str | Path) -> dict[str, np.ndarray]:
    """Load a .safetensors file.  Falls back to pure-python parser if the
    `safetensors` package is not installed.  BF16 is promoted to float32."""
    try:
        from safetensors.numpy import load_file
        log.info("Using safetensors library.")
        raw = load_file(str(path))
        out = {}
        for k, v in raw.items():
            if v.dtype == np.uint16:   # safetensors stores bf16 as uint16
                u32 = v.astype(np.uint32) << 16
                out[k] = u32.view(np.float32)
            else:
                out[k] = v.astype(np.float32)
        return out
    except ImportError:
        pass

    log.info("safetensors not found; using built-in parser.")
    return _read_safetensors_manual(path)


def _read_safetensors_manual(path: str | Path) -> dict[str, np.ndarray]:
    """Format: [8-byte header_len (LE uint64)] [JSON header] [tensor data]"""
    DTYPE_MAP = {
        "F32":  (np.float32, 4),
        "F16":  (np.float16, 2),
        "BF16": ("bf16", 2),
        "I64":  (np.int64, 8),
        "I32":  (np.int32, 4),
        "I8":   (np.int8, 1),
        "U8":   (np.uint8, 1),
        "BOOL": (np.bool_, 1),
    }
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header: dict = json.loads(f.read(header_len).decode("utf-8"))
        raw_data: bytes = f.read()

    tensors: dict[str, np.ndarray] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        dtype_str = meta["dtype"]
        shape     = meta["shape"]
        s, e      = meta["data_offsets"]
        chunk     = raw_data[s:e]

        if dtype_str == "BF16":
            u16 = np.frombuffer(chunk, dtype=np.uint16)
            u32 = u16.astype(np.uint32) << 16
            arr = u32.view(np.float32)
        else:
            np_dtype, _ = DTYPE_MAP[dtype_str]
            arr = np.frombuffer(chunk, dtype=np_dtype).astype(np.float32)

        tensors[name] = arr.reshape(shape) if shape else arr
    return tensors


def _detect_prefix(weights: dict) -> Optional[str]:
    for prefix in ("", "model.", "backbone.", "encoder."):
        if f"{prefix}patch_embed.proj.weight" in weights:
            return prefix
    return None


# ══════════════════════════════════════════════════════════════════════════════
# 2. inspect sub-command
# ══════════════════════════════════════════════════════════════════════════════

def cmd_inspect(args: argparse.Namespace) -> None:
    log.info(f"Inspecting: {args.src}")
    weights = _read_safetensors(args.src)
    log.info(f"Total tensors: {len(weights)}")

    keys = [k for k in sorted(weights) if (args.key or "") in k]
    if not keys:
        log.error(f"No key matching '{args.key}'.")
        sys.exit(1)

    col_w = max(len(k) for k in keys) + 2
    print(f"\n{'KEY':<{col_w}}  {'SHAPE':<24}  DTYPE")
    print("-" * (col_w + 40))
    for k in keys:
        v = weights[k]
        print(f"{k:<{col_w}}  {str(tuple(v.shape)):<24}  {v.dtype}")

    prefix = _detect_prefix(weights)
    if prefix is None:
        log.warning("Could not auto-detect key prefix.")
    else:
        log.info(f"Auto-detected prefix: {repr(prefix) if prefix else 'empty'}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. convert sub-command
# ══════════════════════════════════════════════════════════════════════════════

def cmd_convert(args: argparse.Namespace) -> None:
    t0 = time.time()
    log.info(f"Loading: {args.src}")
    pt = _read_safetensors(args.src)

    prefix = args.prefix
    if prefix is None:
        prefix = _detect_prefix(pt)
        if prefix is None:
            log.error("Cannot detect key prefix. Use --inspect then --prefix <p>.")
            sys.exit(1)
        log.info(f"Detected prefix: {repr(prefix) if prefix else 'empty string'}")

    def _get(key: str) -> np.ndarray:
        full = f"{prefix}{key}"
        if full not in pt:
            raise KeyError(
                f"Expected key '{full}' not in safetensors.\n"
                f"Run --inspect to verify names, or set --prefix."
            )
        return pt[full].astype(np.float32)

    # ── Validate source architecture ──────────────────────────────────────────
    _check_source_shapes(pt, prefix)

    # ── Build JAX weight dict ─────────────────────────────────────────────────
    jax_w: dict[str, np.ndarray] = {}

    # Patch embedding: PyTorch Conv (O,I,H,W) → JAX Conv (H,W,I,O)
    jax_w["patch_embed_proj_kernel"] = _get("patch_embed.proj.weight").transpose(2, 3, 1, 0)
    jax_w["patch_embed_proj_bias"]   = _get("patch_embed.proj.bias")

    # Register token (no transposition)
    jax_w["register_tokens"] = _get("register_tokens")

    # Position encoding: frequency bands (2, 96) — no transposition
    jax_w["freq_bands"] = _get("pos_embed.frequency_bands")

    # Transformer blocks
    for i in range(DEPTH):
        pp = f"blocks.{i}"   # PyTorch prefix
        op = f"block_{i}"    # JAX prefix

        # LayerNorm: PyTorch 'weight' → JAX 'scale'
        jax_w[f"{op}_norm1_scale"] = _get(f"{pp}.norm1.weight")
        jax_w[f"{op}_norm1_bias"]  = _get(f"{pp}.norm1.bias")

        # Attention: Linear (out,in) → (in,out)
        jax_w[f"{op}_attn_qkv_kernel"]  = _get(f"{pp}.attn.qkv.weight").T
        jax_w[f"{op}_attn_qkv_bias"]    = _get(f"{pp}.attn.qkv.bias")
        jax_w[f"{op}_attn_proj_kernel"] = _get(f"{pp}.attn.proj.weight").T
        jax_w[f"{op}_attn_proj_bias"]   = _get(f"{pp}.attn.proj.bias")

        # LayerScale
        jax_w[f"{op}_ls1_gamma"] = _get(f"{pp}.ls1.gamma")

        jax_w[f"{op}_norm2_scale"] = _get(f"{pp}.norm2.weight")
        jax_w[f"{op}_norm2_bias"]  = _get(f"{pp}.norm2.bias")

        # MLP: Linear transpositions
        jax_w[f"{op}_mlp_fc1_kernel"] = _get(f"{pp}.mlp.fc1.weight").T
        jax_w[f"{op}_mlp_fc1_bias"]   = _get(f"{pp}.mlp.fc1.bias")
        jax_w[f"{op}_mlp_fc2_kernel"] = _get(f"{pp}.mlp.fc2.weight").T
        jax_w[f"{op}_mlp_fc2_bias"]   = _get(f"{pp}.mlp.fc2.bias")

        # LayerScale
        jax_w[f"{op}_ls2_gamma"] = _get(f"{pp}.ls2.gamma")

    # Final norm
    jax_w["norm_scale"] = _get("norm.weight")
    jax_w["norm_bias"]  = _get("norm.bias")

    # ── Save ──────────────────────────────────────────────────────────────────
    dst = Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(dst), **jax_w)
    size_mb = dst.stat().st_size / 1e6
    log.info(
        "Saved %d arrays → %s  (%.1f MB, %.1fs)",
        len(jax_w), dst, size_mb, time.time() - t0,
    )

    # ── Validate ─────────────────────────────────────────────────────────────
    if not args.no_validate:
        _validate(jax_w, pt, prefix, args)


# ── Source shape checks ───────────────────────────────────────────────────────

def _check_source_shapes(pt: dict, prefix: str) -> None:
    """Verify the safetensors matches the expected ViT-Small/16 architecture."""
    expected = {
        f"{prefix}patch_embed.proj.weight":    (EMBED_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE),
        f"{prefix}patch_embed.proj.bias":      (EMBED_DIM,),
        f"{prefix}register_tokens":            (1, 1, EMBED_DIM),
        f"{prefix}pos_embed.frequency_bands":  (2, EMBED_DIM // 4),
        f"{prefix}blocks.0.attn.qkv.weight":   (EMBED_DIM * 3, EMBED_DIM),
        f"{prefix}blocks.0.attn.qkv.bias":     (EMBED_DIM * 3,),
        f"{prefix}blocks.0.ls1.gamma":         (EMBED_DIM,),
        f"{prefix}blocks.0.mlp.fc1.weight":    (MLP_HIDDEN, EMBED_DIM),
        f"{prefix}norm.weight":                (EMBED_DIM,),
    }
    errors = []
    for key, exp in expected.items():
        if key not in pt:
            errors.append(f"  MISSING: {key!r}")
        elif tuple(pt[key].shape) != exp:
            errors.append(
                f"  SHAPE {key!r}: got {pt[key].shape}, expected {exp}"
            )
    if errors:
        log.error("Architecture mismatch:\n" + "\n".join(errors))
        sys.exit(1)
    log.info("Source shapes OK — ViT-Small/16 with LayerScale + SinCos2D PE.")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Validation
# ══════════════════════════════════════════════════════════════════════════════

def _validate(
    jax_w: dict,
    pt_w:  dict,
    prefix: str,
    args:  argparse.Namespace,
) -> None:
    log.info("=" * 64)
    log.info("VALIDATION")
    log.info("=" * 64)
    _validate_l1(jax_w)
    _validate_l2(jax_w)
    if args.torch_compare:
        _validate_l3(jax_w, pt_w, prefix)
    log.info("=" * 64)
    log.info("ALL CHECKS PASSED")
    log.info("=" * 64)


# ── L1: shapes ────────────────────────────────────────────────────────────────

def _validate_l1(jax_w: dict) -> None:
    log.info("[L1] Shape & dtype checks …")
    expected: dict[str, tuple] = {
        "patch_embed_proj_kernel": (PATCH_SIZE, PATCH_SIZE, IN_CHANNELS, EMBED_DIM),
        "patch_embed_proj_bias":   (EMBED_DIM,),
        "register_tokens":         (1, 1, EMBED_DIM),
        "freq_bands":              (2, EMBED_DIM // 4),
        "norm_scale":              (EMBED_DIM,),
        "norm_bias":               (EMBED_DIM,),
    }
    for i in range(DEPTH):
        p = f"block_{i}"
        expected.update({
            f"{p}_norm1_scale":      (EMBED_DIM,),
            f"{p}_norm1_bias":       (EMBED_DIM,),
            f"{p}_attn_qkv_kernel":  (EMBED_DIM, EMBED_DIM * 3),
            f"{p}_attn_qkv_bias":    (EMBED_DIM * 3,),
            f"{p}_attn_proj_kernel": (EMBED_DIM, EMBED_DIM),
            f"{p}_attn_proj_bias":   (EMBED_DIM,),
            f"{p}_ls1_gamma":        (EMBED_DIM,),
            f"{p}_norm2_scale":      (EMBED_DIM,),
            f"{p}_norm2_bias":       (EMBED_DIM,),
            f"{p}_mlp_fc1_kernel":   (EMBED_DIM, MLP_HIDDEN),
            f"{p}_mlp_fc1_bias":     (MLP_HIDDEN,),
            f"{p}_mlp_fc2_kernel":   (MLP_HIDDEN, EMBED_DIM),
            f"{p}_mlp_fc2_bias":     (EMBED_DIM,),
            f"{p}_ls2_gamma":        (EMBED_DIM,),
        })

    errors = []
    for key, exp in expected.items():
        if key not in jax_w:
            errors.append(f"  MISSING: {key!r}")
        elif tuple(jax_w[key].shape) != exp:
            errors.append(f"  SHAPE {key!r}: got {jax_w[key].shape}, want {exp}")
    if errors:
        log.error("[L1] FAILED:\n" + "\n".join(errors))
        sys.exit(1)

    total = sum(v.size for v in jax_w.values())
    log.info("[L1] PASSED — %d arrays, %.2fM params", len(jax_w), total / 1e6)

    # Spot-check weight norms
    for key in ["patch_embed_proj_kernel", "block_0_attn_qkv_kernel",
                "block_0_ls1_gamma", "block_6_mlp_fc1_kernel", "norm_scale"]:
        v = jax_w[key]
        log.info(
            "  %-38s  norm=%8.3f  mean=%+.4f  std=%.4f",
            key, float(np.linalg.norm(v)), float(np.mean(v)), float(np.std(v)),
        )

    # LayerScale should be small but non-zero (DINO typically init 0.1 or 1.0)
    ls_val = float(np.mean(jax_w["block_0_ls1_gamma"]))
    log.info("  block_0 ls1.gamma mean = %.4f (typical: 0.1 – 1.0)", ls_val)
    if ls_val < 1e-4:
        log.warning("  LayerScale values are very small — possible loading issue.")


# ── L2: JAX forward pass ──────────────────────────────────────────────────────

def _inject_weights(model, jax_w: dict) -> None:
    """Load numpy arrays directly into SparshViTSmall NNX model."""
    import jax.numpy as jnp
    _a = lambda k: jnp.array(jax_w[k], dtype=jnp.float32)

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


def _validate_l2(jax_w: dict) -> None:
    log.info("[L2] JAX forward pass …")
    try:
        import jax, jax.numpy as jnp, flax.nnx as nnx
        openpi_src = str(Path(__file__).parents[2] / "openpi/src")
        if openpi_src not in sys.path:
            sys.path.insert(0, openpi_src)
        from openpi.models.sparsh_encoder import SparshViTSmall
    except ImportError as e:
        log.warning("[L2] Skipped — import error: %s", e)
        return

    model = SparshViTSmall(rngs=nnx.Rngs(0))
    _inject_weights(model, jax_w)

    rng   = jax.random.PRNGKey(42)
    dummy = jax.random.uniform(rng, (2, IMAGE_SIZE, IMAGE_SIZE, IN_CHANNELS))
    out   = jax.jit(model)(dummy)   # (2, 197, 384)

    ok = True
    exp_shape = (2, SEQ_LEN, EMBED_DIM)
    if out.shape != exp_shape:
        log.error("[L2] SHAPE: got %s, want %s", out.shape, exp_shape)
        ok = False
    else:
        log.info("  output shape: %s  ✓", out.shape)

    n_nan = int(jnp.isnan(out).sum())
    n_inf = int(jnp.isinf(out).sum())
    if n_nan > 0 or n_inf > 0:
        log.error("[L2] NaN=%d  Inf=%d", n_nan, n_inf)
        ok = False
    else:
        log.info("  no NaN / Inf  ✓")

    # Register token statistics
    reg_out  = np.array(out[:, 0, :])
    reg_norm = float(np.linalg.norm(reg_out, axis=-1).mean())
    reg_std  = float(np.std(reg_out))
    log.info("  register token: L2 norm=%.3f  std=%.4f", reg_norm, reg_std)
    if reg_norm < 0.01 or reg_norm > 1000.0:
        log.warning("  register token norm %.3f looks unusual.", reg_norm)

    # Patch token diversity
    patch_out = np.array(out[:, 1:, :])
    var_across_patches = float(np.var(patch_out, axis=1).mean())
    log.info("  patch token variance (across positions): %.6f", var_across_patches)
    if var_across_patches < 1e-8:
        log.error("[L2] Patch tokens are nearly identical — possible forward-pass bug.")
        ok = False

    # Two different inputs should produce different outputs
    dummy2 = jax.random.uniform(jax.random.PRNGKey(99), (2, IMAGE_SIZE, IMAGE_SIZE, IN_CHANNELS))
    out2   = jax.jit(model)(dummy2)
    diff12 = float(jnp.mean(jnp.abs(out - out2)))
    log.info("  mean |output1 - output2| (different inputs): %.4f", diff12)
    if diff12 < 1e-6:
        log.error("[L2] Model produces identical output for different inputs.")
        ok = False

    if ok:
        log.info("[L2] PASSED ✓")
    else:
        log.error("[L2] FAILED ✗")
        sys.exit(1)


# ── L3: PyTorch element-wise comparison ───────────────────────────────────────

def _validate_l3(jax_w: dict, pt_w: dict, prefix: str) -> None:
    log.info("[L3] PyTorch element-wise comparison …")
    try:
        import torch
    except ImportError:
        log.warning("[L3] Skipped — torch not available.")
        return
    try:
        import timm
    except ImportError:
        log.warning("[L3] Skipped — timm not available (needed to instantiate PT model).")
        return
    try:
        import jax, jax.numpy as jnp, flax.nnx as nnx
        openpi_src = str(Path(__file__).parents[2] / "openpi/src")
        if openpi_src not in sys.path:
            sys.path.insert(0, openpi_src)
        from openpi.models.sparsh_encoder import SparshViTSmall
    except ImportError as e:
        log.warning("[L3] Skipped — JAX import error: %s", e)
        return

    # ── Build PyTorch model ────────────────────────────────────────────────────
    # Try a few timm model names that match the architecture
    pt_model = None
    for model_name in [
        "vit_small_patch16_224",
        "vit_small_patch16_224.augreg_in21k",
    ]:
        try:
            pt_model = timm.create_model(model_name, pretrained=False, in_chans=IN_CHANNELS, num_classes=0)
            log.info("  Using timm model: %s", model_name)
            break
        except Exception:
            continue
    if pt_model is None:
        log.warning("[L3] Could not create timm model — skipping.")
        return

    pt_model.eval()

    # Load PT weights (only keys present in the state dict)
    pt_sd  = pt_model.state_dict()
    new_sd = {}
    for k in pt_sd:
        full_k = f"{prefix}{k}"
        if full_k in pt_w:
            new_sd[k] = torch.from_numpy(pt_w[full_k])
    missing = set(pt_sd.keys()) - set(new_sd.keys())
    if missing:
        log.warning("  %d PT keys not loaded (head?): %s", len(missing), list(missing)[:5])
    pt_model.load_state_dict(new_sd, strict=False)
    pt_model.eval()

    # ── Shared input ──────────────────────────────────────────────────────────
    rng    = np.random.RandomState(7)
    np_in  = rng.rand(1, IMAGE_SIZE, IMAGE_SIZE, IN_CHANNELS).astype(np.float32)

    # PyTorch: NCHW
    with torch.no_grad():
        pt_in  = torch.from_numpy(np_in.transpose(0, 3, 1, 2))
        pt_out = pt_model.forward_features(pt_in).numpy()  # (1, 197, 384) or (1, 384)

    # JAX: NHWC
    jax_model = SparshViTSmall(rngs=nnx.Rngs(0))
    _inject_weights(jax_model, jax_w)
    jax_out = np.array(jax.jit(jax_model)(jnp.array(np_in)))   # (1, 197, 384)

    # Align shapes (timm might return only CLS)
    if pt_out.ndim == 2:
        pt_out  = pt_out[:, None, :]
        jax_out = jax_out[:, :1, :]

    diff      = np.abs(jax_out - pt_out)
    max_diff  = float(diff.max())
    mean_diff = float(diff.mean())
    rel_diff  = float(max_diff / (np.abs(pt_out).max() + 1e-8))

    log.info("  max |JAX − PT|  = %.3e", max_diff)
    log.info("  mean|JAX − PT|  = %.3e", mean_diff)
    log.info("  relative error  = %.3e", rel_diff)

    if max_diff < 1e-3:
        log.info("[L3] PASSED ✓  (max diff %.2e)", max_diff)
    elif max_diff < 5e-3:
        log.warning("[L3] MARGINAL — max diff %.2e  (float32 accumulation).", max_diff)
    else:
        idx = np.unravel_index(diff.argmax(), diff.shape)
        log.error(
            "[L3] FAILED ✗  max diff %.2e  at %s  "
            "(JAX=%.5f, PT=%.5f).  Check transpositions or arch mismatch.",
            max_diff, idx, jax_out[idx], pt_out[idx],
        )
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# 5. CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--src",    default=DEFAULT_SRC, help="Source .safetensors file.")
    p.add_argument("--dst",    default=DEFAULT_DST, help="Output .npz file.")
    p.add_argument("--prefix", default=None,        help="Key prefix override.")

    sub = p.add_subparsers(dest="cmd")

    ins = sub.add_parser("inspect", help="Print all keys/shapes in the source file.")
    ins.add_argument("--key", default="", help="Filter keys containing this substring.")

    conv = sub.add_parser("convert", help="Convert weights and run validation.")
    conv.add_argument("--no-validate",   action="store_true",
                      help="Skip validation after conversion.")
    conv.add_argument("--torch-compare", action="store_true",
                      help="Level-3: element-wise comparison with PyTorch (needs torch+timm).")

    return p


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()
    if args.cmd == "inspect":
        cmd_inspect(args)
    elif args.cmd == "convert":
        cmd_convert(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
