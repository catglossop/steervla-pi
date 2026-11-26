"""Transforms for the ABC bimanual sim (scene camera + two wrist cameras, 14-D joint state/actions)."""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

ACTION_DIM = 14
# ABC camera name -> openpi image slot.
CAMERA_SLOTS = {"top": "base_0_rgb", "left_wrist": "left_wrist_0_rgb", "right_wrist": "right_wrist_0_rgb"}


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:  # CHW (as the ABC sim emits) -> HWC
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class AbcInputs(transforms.DataTransformFn):
    """Raw ABC observation -> openpi model inputs. Used for training and inference.

    Expects ``observation/images/{top,left_wrist,right_wrist}`` (HWC or CHW uint8), ``observation/state`` (14,),
    optional ``actions`` (chunk, 14), ``prompt`` and ``task_group``. A camera whose
    ``observation/image_mask/<cam>`` is False (see :class:`AbcObservationDropout`) is masked out of attention.
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        images, masks = {}, {}
        for cam, slot in CAMERA_SLOTS.items():
            images[slot] = _parse_image(data[f"observation/images/{cam}"])
            masks[slot] = np.asarray(data.get(f"observation/image_mask/{cam}", True), dtype=np.bool_)
        inputs = {"state": np.asarray(data["observation/state"], np.float32), "image": images, "image_mask": masks}
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], np.float32)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        if "action_prefix_len" in data:
            inputs["action_prefix_len"] = np.asarray(data["action_prefix_len"], np.int32)
        if "task_group" in data:
            # Carried on Observation.dataset_id so evals can group metrics by task.
            inputs["dataset_id"] = np.asarray(data["task_group"], np.int32)
        return inputs


@dataclasses.dataclass(frozen=True)
class AbcOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :ACTION_DIM])}


@dataclasses.dataclass(frozen=True)
class AbcObservationDropout(transforms.DataTransformFn):
    """Training-only conditioning dropout with an independent probability per input.

    A dropped camera is zeroed and its image mask set False (the model never attends to it); a dropped
    prompt becomes the empty string, which trains the unconditional (task-free) policy evaluated with
    ``prompt=""``. Put this in ``repack_transforms`` so it never runs at inference.
    """

    prompt: float = 0.0
    top: float = 0.0
    left_wrist: float = 0.0
    right_wrist: float = 0.0

    def __call__(self, data: dict) -> dict:
        rng = _rng()
        data = dict(data)
        if "prompt" in data and rng.random() < self.prompt:
            data["prompt"] = ""
        for cam in CAMERA_SLOTS:
            if rng.random() < getattr(self, cam):
                key = f"observation/images/{cam}"
                data[key] = np.zeros_like(np.asarray(data[key]))
                data[f"observation/image_mask/{cam}"] = np.False_
        return data


_RNG: np.random.Generator | None = None


def _rng() -> np.random.Generator:
    # One generator per process (data-loader workers are separate processes), seeded from OS entropy.
    global _RNG  # noqa: PLW0603
    if _RNG is None:
        _RNG = np.random.default_rng()
    return _RNG
