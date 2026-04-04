"""
Convert a nuScenes driving dataset (in RLDS format) to LeRobot format for fine-tuning.

This is useful for smaller subsets of the data. For training on the full dataset,
use the RLDS data loader directly (see steervla_rlds_dataset.py).

Usage:
uv run --group rlds examples/steervla/convert_nuscenes_to_lerobot.py \
    --rlds_data_dir /path/to/rlds/data \
    --dataset_name nuscenes_dataset_img512_0910

If you want to push your dataset to the Hugging Face Hub:
uv run --group rlds examples/steervla/convert_nuscenes_to_lerobot.py \
    --rlds_data_dir /path/to/rlds/data --push_to_hub
"""

import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm
import tyro


REPO_NAME = "your_hf_username/my_steervla_dataset"

FPS = 2
ACTION_HORIZON = FPS * 3
INCLUDE_EGO_HISTORY = True
PROPRIO_NORM = True
SPEED_IN_PROMPT = True


def normalize_course(courses: np.ndarray) -> np.ndarray:
    courses = (courses % 360.0 + 360.0) % 360.0
    return np.where(courses > 180.0, courses - 360.0, courses)


def process_state(raw_state: np.ndarray) -> np.ndarray:
    """Process ego state from interleaved [speed, course] pairs."""
    num_pairs = raw_state.shape[-1] // 2
    reshaped = raw_state.reshape(num_pairs, 2)
    speeds = reshaped[:, 0]
    courses = normalize_course(reshaped[:, 1])

    if PROPRIO_NORM:
        speeds = speeds / 20.0
        courses = courses / 180.0

    stacked = np.stack([speeds, courses], axis=-1).flatten()
    if INCLUDE_EGO_HISTORY:
        return stacked[-8:].astype(np.float32)
    return stacked[-2:-1].astype(np.float32)


def process_actions(action_chunk: np.ndarray) -> np.ndarray:
    """Normalize action chunk (delta_speed, course) to [-1, 1]."""
    speed_deltas = action_chunk[:, 0] / 10.0
    courses = normalize_course(action_chunk[:, 1]) / 180.0
    actions = np.stack([speed_deltas, courses], axis=-1)
    return actions[:ACTION_HORIZON].astype(np.float32)


def main(
    rlds_data_dir: str,
    dataset_name: str = "nuscenes_dataset_img512_0910",
    dataset_version: str = "1.0.0",
    max_episodes: int | None = None,
    *,
    push_to_hub: bool = False,
):
    import tensorflow as tf
    import tensorflow_datasets as tfds
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

    tf.config.set_visible_devices([], "GPU")

    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    state_dim = 8 if INCLUDE_EGO_HISTORY else 1
    action_dim = 2

    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="vehicle",
        fps=FPS,
        features={
            "image": {
                "dtype": "image",
                "shape": (512, 512, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    builder = tfds.builder(dataset_name, data_dir=rlds_data_dir, version=dataset_version)
    tfds_dataset = builder.as_dataset(split="train")

    episode_count = 0
    for episode in tqdm(tfds_dataset, desc="Converting episodes"):
        if max_episodes is not None and episode_count >= max_episodes:
            break

        steps = list(episode["steps"])
        if len(steps) == 0:
            continue

        language_instruction = steps[0].get("language_instruction", b"").numpy()
        if isinstance(language_instruction, bytes):
            language_instruction = language_instruction.decode("utf-8")

        for step in steps:
            obs = step["observation"]
            front_image = obs["front_image"].numpy()
            if front_image.dtype != np.uint8:
                front_image = tf.io.decode_image(front_image, expand_animations=False, dtype=tf.uint8).numpy()

            raw_state = obs["state"].numpy()
            state = process_state(raw_state)

            action_chunk = step["action_chunk"].numpy()
            actions_flat = process_actions(action_chunk)

            prompt = language_instruction
            if SPEED_IN_PROMPT:
                current_speed = raw_state[-2]
                prompt = f"The current speed is {current_speed} m/s. {prompt}"

            for t in range(min(ACTION_HORIZON, actions_flat.shape[0])):
                dataset.add_frame(
                    {
                        "image": front_image,
                        "state": state,
                        "actions": actions_flat[t],
                        "task": prompt,
                    }
                )

        dataset.save_episode()
        episode_count += 1

    print(f"Converted {episode_count} episodes")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["steervla", "nuscenes", "driving"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
