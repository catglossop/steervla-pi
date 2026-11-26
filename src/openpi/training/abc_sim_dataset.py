"""ABC simulation dataset (https://github.com/amazon-far/abc) converted to LeRobot shards.

Raw episodes (``<root>/<task>/episode_<uuid>/episode.mcap``) are converted once by
``examples/abc_sim/convert_abc_sim_to_lerobot.py`` into many LeRobot datasets ("shards") under
``<lerobot_root>/{train,val}_NNNN``. This module defines the task list and prompt rule shared by the
converter, training and eval, and the dataset that stitches the shards together.

Data layout, following ABC's own exporter (``abc_minimal/export_mcap.py``):
  * 30 Hz fixed clock, causal floor resampling of every stream.
  * state / actions: 14-D ``[L joint1..6, L gripper, R joint1..6, R gripper]``, absolute joint targets.
  * cameras: ``top`` (scene), ``left_wrist``, ``right_wrist``, stored at native 224x168.
  * action chunk: 30 steps (1 s), as in ABC.
"""

from collections.abc import Sequence
import dataclasses
import json
import pathlib

import numpy as np

# Task directory names in the raw release, in a fixed order: ``task_group`` indexes into this list.
TASKS: tuple[str, ...] = (
    "ball_tray_balancing",
    "conveyor_pick",
    "count_into_opaque_box",
    "grab_clutter",
    "hang_the_mug_on_the_mug_rack",
    "inhand_transfer_the_item_to_other_side",
    "lego_blocks_sorting",
    "load_the_plates_into_the_dish_rack",
    "multi_drawer_search",
    "nuts_bolts_sorting",
    "pouring_beads",
    "put_relative",
    "put_the_plastic_bottles_in_the_bin",
    "set_up_chess_pieces_on_the_board",
    "spell_abc",
    "spell_agi",
    "spell_cat",
    "spell_dog",
    "spell_fish",
    "spell_xdof",
    "spell_yam",
    "sweep_away_paper_scraps_from_the_table",
    "throw_plastic_bottles_in_bin",
    "turn_the_mug_right_side_up",
)

# Tasks whose real instruction varies per episode; it lives in randomization.json -> metadata.prompt
# (and abc_sim's env sets the same directive on reset).
DIRECTIVE_TASKS = ("put_relative", "grab_clutter", "count_into_opaque_box", "multi_drawer_search")

CAMERAS = ("top", "left_wrist", "right_wrist")
ACTION_DIM = 14
CHUNK_LENGTH = 30  # 30 steps @ 30 Hz = 1 s
FPS = 30


def task_prompt(task: str, directive: str | None = None) -> str:
    """Prompt for an episode: the per-episode directive if there is one, else the task name with spaces."""
    if directive:
        return directive.strip()
    return task.replace("_", " ")


def episode_directive(episode_dir: pathlib.Path) -> str | None:
    path = episode_dir / "randomization.json"
    if not path.exists():
        return None
    prompt = json.loads(path.read_text()).get("metadata", {}).get("prompt")
    return prompt or None


def is_val_episode(episode_name: str, val_permille: int) -> bool:
    """Deterministic held-out split from the episode uuid (stable across machines and reruns)."""
    import hashlib

    return int(hashlib.sha1(episode_name.encode()).hexdigest()[:8], 16) % 1000 < val_permille


@dataclasses.dataclass(frozen=True)
class AbcSimLoaderConfig:
    """Options for :class:`AbcSimDataset`, carried on ``DataConfig.abc_sim``."""

    # Directory holding the converted LeRobot shards (train_NNNN / val_NNNN).
    lerobot_root: str = "/raid/users/cglossop/datasets/abc_sim_lerobot"
    # Restrict to these task names (None = all of TASKS).
    tasks: Sequence[str] | None = None
    chunk_length: int = CHUNK_LENGTH


class AbcSimDataset:
    """Random-access dataset over all converted shards of one split.

    Each item is one frame with its 30-step action chunk:
    ``{observation/images/{top,left_wrist,right_wrist}: uint8 HWC, observation/state: (14,),
    actions: (chunk, 14), prompt: str, task_group: int}``. Chunks never run past the episode end (ABC uses
    ``len - chunk + 1`` start frames per episode, no padding).
    """

    def __init__(self, config: AbcSimLoaderConfig, *, split: str = "train"):
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        root = pathlib.Path(config.lerobot_root)
        shard_dirs = sorted(p for p in root.glob(f"{split}_*") if (p / "DONE").exists())
        if not shard_dirs:
            raise FileNotFoundError(f"No finished {split} shards under {root}; run the converter first.")
        wanted = None if config.tasks is None else {TASKS.index(t) for t in config.tasks}
        delta = {"action": [i / FPS for i in range(config.chunk_length)]}
        self._shards = []
        self._index: list[np.ndarray] = []  # per shard: valid global frame indices
        for shard_dir in shard_dirs:
            ds = LeRobotDataset(repo_id=f"abc_sim/{shard_dir.name}", root=shard_dir, delta_timestamps=delta)
            ep_from = ds.episode_data_index["from"].numpy()
            ep_to = ds.episode_data_index["to"].numpy()
            groups = np.asarray(ds.hf_dataset["task_group"]).reshape(-1)
            valid = []
            for start, end in zip(ep_from, ep_to, strict=True):
                last = end - config.chunk_length + 1  # ABC: no padding past the episode end
                if last > start and (wanted is None or int(groups[start]) in wanted):
                    valid.append(np.arange(start, last))
            if valid:
                self._shards.append(ds)
                self._index.append(np.concatenate(valid))
        self._offsets = np.cumsum([0] + [len(i) for i in self._index])

    def __len__(self) -> int:
        return int(self._offsets[-1])

    def __getitem__(self, index) -> dict:
        index = int(index)
        shard = int(np.searchsorted(self._offsets, index, side="right") - 1)
        item = self._shards[shard][int(self._index[shard][index - self._offsets[shard]])]

        def image(key):  # LeRobot returns float CHW in [0, 1]
            return (np.asarray(item[f"observation.images.{key}"]).transpose(1, 2, 0) * 255).round().astype(np.uint8)

        return {
            **{f"observation/images/{cam}": image(cam) for cam in CAMERAS},
            "observation/state": np.asarray(item["observation.state"], np.float32),
            "actions": np.asarray(item["action"], np.float32),
            "prompt": item["task"],
            "task_group": np.int32(np.asarray(item["task_group"]).reshape(())),
        }
