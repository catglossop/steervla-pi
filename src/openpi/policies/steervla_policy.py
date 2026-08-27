import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_steervla_example() -> dict:
    """Creates a random input example for the SteerVLA policy."""
    return {
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/state": np.random.rand(8).astype(np.float32),
        "subtask": "",
        "reasoning": "",
        "prompt": "The car is driving on a highway.",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def normalize_ego_state(
    state: np.ndarray,
    *,
    include_ego_history: bool = True,
    proprio_norm: bool = True,
) -> np.ndarray:
    """Process the ego state from the nuScenes format.

    The raw state is interleaved [speed, course] pairs. This function centers
    course angles to (-180, 180], optionally normalizes speed (/20) and
    course (/180), and returns the last N history states or just current speed.
    """
    state = np.asarray(state, dtype=np.float32)
    num_pairs = state.shape[-1] // 2
    reshaped = state.reshape(*state.shape[:-1], num_pairs, 2)
    speeds = reshaped[..., 0]
    courses = reshaped[..., 1]

    courses = (courses % 360.0 + 360.0) % 360.0
    courses = np.where(courses > 180.0, courses - 360.0, courses)

    if proprio_norm:
        speeds = speeds / 20.0
        courses = courses / 180.0

    stacked = np.stack([speeds, courses], axis=-1)
    flat = stacked.reshape(*state.shape[:-1], -1)

    if not include_ego_history:
        return flat[..., -2:]
    # Last 4 history states = 8 values (speed, course) * 4
    return flat[..., -8:]


def normalize_actions(
    actions: np.ndarray,
    *,
    include_xy_action: bool = False,
    global_course: np.ndarray | None = None,
) -> np.ndarray:
    """Normalize nuScenes actions (delta_speed, course, optional xy) to [-1, 1]."""
    delta_speed_norm = 10.0
    delta_xy_norm = 15.0

    speed_deltas = actions[..., 0] / delta_speed_norm
    courses = (actions[..., 1] % 360.0 + 360.0) % 360.0
    courses = np.where(courses > 180.0, courses - 360.0, courses)
    normalized_courses = courses / 180.0

    result = np.stack([speed_deltas, normalized_courses], axis=-1)

    if include_xy_action and actions.shape[-1] >= 4 and global_course is not None:
        global_xy_deltas = actions[..., 2:4]
        yaw_rad = np.deg2rad(global_course[..., np.newaxis])
        c, s = np.cos(yaw_rad), np.sin(yaw_rad)
        x_ego = c * global_xy_deltas[..., 0] + s * global_xy_deltas[..., 1]
        y_ego = -s * global_xy_deltas[..., 0] + c * global_xy_deltas[..., 1]
        ego_xy = np.stack([x_ego, y_ego], axis=-1) / delta_xy_norm
        result = np.concatenate([result, ego_xy], axis=-1)

    return result


@dataclasses.dataclass(frozen=True)
class SteerVLAInputs(transforms.DataTransformFn):
    model_type: _model.ModelType
    speed_in_prompt: bool = True
    include_ego_history: bool = True
    proprio_norm: bool = True
    # Camera streams to emit. ``None`` keeps the historical behaviour of emitting all three
    # PaliGemma slots, with the two wrist slots filled with zeros and masked off. Driving is
    # single-camera, so those two are pure padding: masked-off image tokens are excluded from
    # attention but still cost a SigLIP forward and 2x256 dead prefix positions per sample.
    # Set ``("base_0_rgb",)`` to drop them. Must match ``Pi0CoTConfig.image_keys``.
    image_keys: tuple[str, ...] | None = None

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        state = normalize_ego_state(
            np.asarray(data["observation/state"], dtype=np.float32),
            include_ego_history=self.include_ego_history,
            proprio_norm=self.proprio_norm,
        )

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, np.zeros_like(base_image), np.zeros_like(base_image))
                image_masks = (np.True_, np.False_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (base_image, np.zeros_like(base_image), np.zeros_like(base_image))
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        image_dict = dict(zip(names, images, strict=True))
        image_mask_dict = dict(zip(names, image_masks, strict=True))

        if self.image_keys is not None:
            missing = set(self.image_keys) - set(image_dict)
            if missing:
                raise ValueError(
                    f"image_keys {sorted(missing)} are not produced for model_type={self.model_type}; "
                    f"available: {sorted(image_dict)}"
                )
            image_dict = {k: image_dict[k] for k in self.image_keys}
            image_mask_dict = {k: image_mask_dict[k] for k in self.image_keys}

        inputs = {
            "state": state,
            "image": image_dict,
            "image_mask": image_mask_dict,
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "action_loss_mask" in data:
            inputs["action_loss_mask"] = np.asarray(data["action_loss_mask"], dtype=np.bool_)
        if "cot_loss_mask" in data:
            inputs["cot_loss_mask"] = np.asarray(data["cot_loss_mask"], dtype=np.bool_)
        if "dataset_id" in data:
            inputs["dataset_id"] = np.asarray(data["dataset_id"], dtype=np.int32)

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")

            if self.speed_in_prompt and "observation/current_speed" in data:
                speed = float(data["observation/current_speed"])
                prompt = f"The current speed is {speed} m/s. {prompt}"

            inputs["prompt"] = prompt

        for cot_key in ("reasoning", "subtask"):
            if cot_key in data:
                val = data[cot_key]
                if isinstance(val, bytes):
                    val = val.decode("utf-8")
                inputs[cot_key] = val

        return inputs


@dataclasses.dataclass(frozen=True)
class SteerVLAOutputs(transforms.DataTransformFn):
    action_dim: int = 2
    enable_cot: bool = False
    def __call__(self, data: dict) -> dict:
        # if self.enable_cot:
        #     return {
        #         "actions": np.asarray(data["actions"][:, :self.action_dim]),
        #         "subtask": data["subtask"],
        #         "reasoning": data["reasoning"],
        #     }
        # else:
        return {"actions": np.asarray(data["actions"][:, :self.action_dim])}
