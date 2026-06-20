import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_univtac_example() -> dict:
    """Creates a random input example for the UniVTAC policy."""
    return {
        "observation/state": np.random.rand(8).astype(np.float32),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _parse_tactile(arr) -> np.ndarray:
    """Normalize a (stack of) tactile image(s) to ``(num_blocks, H, W, 3)`` float32 in [0, 1].

    Accepts channel-first ``(T, C, H, W)`` / ``(C, H, W)`` (LeRobot video convention)
    or channel-last ``(T, H, W, C)`` / ``(H, W, C)``, and uint8 or float inputs.
    """
    arr = np.asarray(arr)
    if arr.ndim == 3:  # single frame → add block axis
        arr = arr[None]
    # Move channel-first (..., 3, H, W) to channel-last (..., H, W, 3) if needed.
    if arr.shape[-3] == 3 and arr.shape[-1] != 3:
        arr = np.moveaxis(arr, -3, -1)
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)
    return arr


@dataclasses.dataclass(frozen=True)
class UniVTACInputs(transforms.DataTransformFn):
    """Converts UniVTAC observations to OpenPI model input format.

    UniVTAC data format (after repack):
      - observation/image: head camera (uint8 or float32, HWC or CHW)
      - observation/wrist_image: wrist camera (optional)
      - observation/state: 8-dim float32 (7 arm joints + 1 gripper)
      - actions: 8-dim float32 (next-step target qpos)
      - prompt: task instruction string
      - observation/tactile_left: (optional) tactile images (num_blocks, H, W, 3)
      - observation/tactile_right: (optional) tactile images (num_blocks, H, W, 3)
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])

        has_wrist = "observation/wrist_image" in data
        if has_wrist:
            wrist_image = _parse_image(data["observation/wrist_image"])
        else:
            wrist_image = np.zeros_like(base_image)

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": (
                    np.True_
                    if has_wrist or self.model_type == _model.ModelType.PI0_FAST
                    else np.False_
                ),
                "right_wrist_0_rgb": (
                    np.True_
                    if self.model_type == _model.ModelType.PI0_FAST
                    else np.False_
                ),
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        # Tactile images at block boundaries for DF tactile training.
        # Loaded via delta_timestamps as a video feature → LeRobot returns
        # (num_blocks, C, H, W) float32 in [0, 1] (channel-first), same convention
        # as the head camera. Normalize to (num_blocks, H, W, 3) float32 in [0, 1].
        has_tac = (
            "observation/tactile_left" in data and "observation/tactile_right" in data
        )
        if has_tac:
            inputs["tactile"] = {
                "left": _parse_tactile(data["observation/tactile_left"]),
                "right": _parse_tactile(data["observation/tactile_right"]),
            }

        return inputs


@dataclasses.dataclass(frozen=True)
class UniVTACOutputs(transforms.DataTransformFn):
    """Converts OpenPI model output back to UniVTAC action format.

    Extracts the first 8 dimensions from the model's 32-dim action output.
    """

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :8])}
