"""Randomly sample subtasks from a SteerVLA SimLingo RLDS dataset.

Example:
    python scripts/sample_subtasks.py \\
        --config-name pi05_steervla_cot_simplified_reasoning \\
        --dataset-name simlingo_dataset_all_img512_1116 \\
        --num-samples 100 \\
        --output sampled_subtasks.json
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import random
from typing import Literal

import tensorflow_datasets as tfds
import tqdm
import tyro

import openpi.training.config as _config


def _decode(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


@dataclasses.dataclass
class Args:
    config_name: str = "pi05_steervla_cot_simplified_reasoning"
    dataset_name: str = "simlingo_dataset_all_img512_1116"
    num_samples: int = 100
    max_steps: int = 750_000
    seed: int = 0
    split: Literal["train", "val", "all"] = "train"
    output: pathlib.Path = pathlib.Path("sampled_subtasks.json")


def main(args: Args) -> None:
    config = _config.get_config(args.config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is None:
        raise ValueError(f"Config {args.config_name!r} does not define an RLDS data directory.")

    subtask_key = data_config.steervla_cot_subtask_key
    reasoning_key = data_config.steervla_cot_reasoning_key
    rng = random.Random(args.seed)

    splits = ["train", "val"] if args.split == "all" else [args.split]
    reservoir: list[dict] = []
    steps_seen = 0

    for split_name in splits:
        builder = tfds.builder(args.dataset_name, data_dir=data_config.rlds_data_dir)
        dataset = builder.as_dataset(
            split=split_name,
            shuffle_files=True,
            read_config=tfds.ReadConfig(shuffle_seed=args.seed),
        )

        progress = tqdm.tqdm(total=args.max_steps - steps_seen, desc=f"Reading {split_name}", unit="step")
        for episode_index, episode in enumerate(dataset):
            for step_index, step in enumerate(episode["steps"]):
                if steps_seen >= args.max_steps:
                    break

                item = {
                    "split": split_name,
                    "episode_index": episode_index,
                    "step_index": step_index,
                    "subtask": _decode(step[subtask_key].numpy()),
                    "reasoning": _decode(step[reasoning_key].numpy()),
                    "prompt": _decode(step["prompt"].numpy()),
                    "routing_command": _decode(step["routing_command"].numpy()),
                }

                if len(reservoir) < args.num_samples:
                    reservoir.append(item)
                else:
                    replace_index = rng.randint(0, steps_seen)
                    if replace_index < args.num_samples:
                        reservoir[replace_index] = item

                steps_seen += 1
                progress.update(1)

            if steps_seen >= args.max_steps:
                break
        progress.close()

    if len(reservoir) < args.num_samples:
        raise ValueError(
            f"Collected {len(reservoir)} subtasks from {steps_seen} steps, "
            f"fewer than requested {args.num_samples}."
        )

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "config_name": args.config_name,
        "dataset_name": args.dataset_name,
        "split": args.split,
        "subtask_key": subtask_key,
        "reasoning_key": reasoning_key,
        "num_samples": len(reservoir),
        "steps_seen": steps_seen,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "samples": reservoir,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    text_path = output_path.with_suffix(".txt")
    text_path.write_text("\n".join(sample["subtask"] for sample in reservoir) + "\n", encoding="utf-8")

    print(f"Sampled {len(reservoir)} subtasks from {steps_seen} steps (max {args.max_steps})")
    print(f"Saved subtasks to {output_path}")
    print(f"Saved plain-text subtasks to {text_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
