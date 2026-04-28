"""
RLDS-based data loader for SteerVLA driving datasets.

Supports both nuScenes and SimLingo dataset formats from bigvision-palivla-drive.
The loader handles multiple weighted datasets, ego state processing, action normalization,
and various language instruction modes.
"""

from collections.abc import Sequence
import dataclasses
from enum import Enum, auto
import logging

from openpi.training.rlds_multihost import shard_dataset_then_batch


class DatasetFormat(Enum):
    """Supported RLDS dataset formats."""

    NUSCENES = auto()
    SIMLINGO = auto()


class OutputActionFormat(Enum):
    """SimLingo action output formats."""

    DELTA_SPEED_T_DELTA_COURSE_T_DELTA_COURSE_SPACE = "delta_speed_t_delta_course_t_delta_course_space"
    DELTA_XY_T_DELTA_XY_SPACE = "delta_xy_t_delta_xy_space"
    DELTA_XY_T_DELTA_COURSE_SPACE = "delta_xy_t_delta_course_space"


class LangLabelType(Enum):
    """SimLingo language label types."""

    COMMENTARY = "commentary"
    GEMINI_SHORTER = "gemini_shorter"
    GEMINI_LONGER = "gemini_longer"
    ROUTING_COMMAND = "routing_command"


@dataclasses.dataclass
class SteerVLARLDSDataset:
    name: str
    weight: float = 1.0
    version: str | None = None


class SteerVLARldsDataset:
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        datasets: Sequence[SteerVLARLDSDataset],
        *,
        dataset_format: DatasetFormat = DatasetFormat.NUSCENES,
        shuffle: bool = True,
        action_chunk_size: int = 6,
        include_ego_history: bool = True,
        include_xy_action: bool = False,
        speed_in_prompt: bool = True,
        proprio_norm: bool = True,
        # SimLingo-specific options
        output_action_format: OutputActionFormat = OutputActionFormat.DELTA_XY_T_DELTA_XY_SPACE,
        lang_label_type: LangLabelType = LangLabelType.ROUTING_COMMAND,
        routing_command_in_prompt: bool = False,
        add_suffix_to_prompt: bool = False,
        enable_cot: bool = False,
        shuffle_buffer_size: int = 50_000,
        num_parallel_reads: int = -1,
        num_parallel_calls: int = -1,
        image_size: int = 512,
    ):
        import dlimp as dl
        import tensorflow as tf
        import tensorflow_datasets as tfds

        tf.config.set_visible_devices([], "GPU")

        # Weights are normalized internally by sample_from_datasets; no need to sum to 1.0.
        total_weight = sum(d.weight for d in datasets)
        assert total_weight > 0, "Total dataset weight must be positive"
        normalized_weights = [d.weight / total_weight for d in datasets]

        def _build_nuscenes_restructure(traj_map_tf=tf):
            """Build the nuScenes restructure function."""

            def restructure(traj):
                traj_len = traj_map_tf.shape(traj["action_chunk"])[0]

                raw_state = traj["observation"]["state"]
                state_dim = raw_state.shape[-1]
                num_pairs = state_dim // 2
                reshaped_state = traj_map_tf.reshape(raw_state, [traj_len, num_pairs, 2])
                state_speeds = reshaped_state[:, :, 0]
                state_courses = reshaped_state[:, :, 1]

                state_courses = (state_courses % 360.0 + 360.0) % 360.0
                state_courses = traj_map_tf.where(state_courses > 180.0, state_courses - 360.0, state_courses)

                if proprio_norm:
                    state_speeds = state_speeds / 20.0
                    state_courses = state_courses / 180.0

                stacked_state = traj_map_tf.stack([state_speeds, state_courses], axis=-1)
                num_features = traj_map_tf.shape(stacked_state)[-1] * traj_map_tf.shape(stacked_state)[-2]
                flat_state = traj_map_tf.reshape(stacked_state, [traj_len, num_features])

                ego_state = flat_state[:, -8:] if include_ego_history else flat_state[:, -2:-1]
                current_speed = traj["observation"]["state"][:, -2]

                delta_speed_norm = 10.0
                speed_deltas = traj["action_chunk"][..., 0] / delta_speed_norm
                action_courses = (traj["action_chunk"][..., 1] % 360.0 + 360.0) % 360.0
                action_courses = traj_map_tf.where(action_courses > 180.0, action_courses - 360.0, action_courses)
                normalized_courses = action_courses / 180.0
                actions = traj_map_tf.concat([speed_deltas[..., None], normalized_courses[..., None]], axis=-1)

                if include_xy_action:
                    delta_xy_norm = 15.0
                    global_course = traj["global_course"]
                    global_xy_deltas = traj["action_chunk"][..., 2:4]
                    yaw_rad = traj_map_tf.cast(3.14159265358979 / 180.0, traj_map_tf.float32) * global_course[..., None]
                    c = traj_map_tf.cos(yaw_rad)
                    s = traj_map_tf.sin(yaw_rad)
                    x_ego = c * global_xy_deltas[..., 0] + s * global_xy_deltas[..., 1]
                    y_ego = -s * global_xy_deltas[..., 0] + c * global_xy_deltas[..., 1]
                    ego_xy = traj_map_tf.stack([x_ego, y_ego], axis=-1) / delta_xy_norm
                    actions = traj_map_tf.concat([actions, ego_xy], axis=-1)

                instruction = traj["language_instruction"]
                if speed_in_prompt:
                    speed_str = traj_map_tf.strings.as_string(current_speed)
                    speed_prompt = traj_map_tf.strings.join(["The current speed is ", speed_str, " m/s. "])
                    instruction = traj_map_tf.strings.join([speed_prompt, instruction], separator="")

                front_image = traj["observation"]["front_image"]

                return {
                    "actions": actions,
                    "observation": {
                        "image": front_image,
                        "state": ego_state,
                        "current_speed": current_speed,
                    },
                    "prompt": instruction,
                }

            return restructure

        def _build_simlingo_restructure():
            """Build the SimLingo restructure function."""

            def restructure(traj):

                traj_len = tf.shape(traj["speed"])[0]
                current_speed_og = traj["speed"]
                
                state_speeds = traj["observation"]["ego_hist"][:, :, 0]
                state_local_courses = traj["observation"]["ego_hist"][:, :, 1]

                state_local_courses = (state_local_courses % 360.0 + 360.0) % 360.0
                state_local_courses = tf.where(
                    state_local_courses > 180.0, state_local_courses - 360.0, state_local_courses
                )

                # TODO: this seems hardcoded, should be a parameter
                if proprio_norm:
                    state_speeds = state_speeds / 20.0
                    state_local_courses = state_local_courses / 180.0

                stacked_state = tf.stack([state_speeds, state_local_courses], axis=-1)
                num_features = tf.shape(stacked_state)[-1] * tf.shape(stacked_state)[-2]
                flat_state = tf.reshape(stacked_state, [traj_len, num_features])

                if include_ego_history:
                    ego_state = flat_state
                else:
                    ego_state = flat_state[:, -2:]

                # Build actions based on output_action_format
                delta_speed_t = traj["action"]["future_10_speed_course_delta_t"][..., 0]
                delta_course_t = traj["action"]["future_10_speed_course_delta_t"][..., 1]
                delta_course_space = traj["action"]["future_10_course_delta_space"]
                delta_xy_t = traj["action"]["future_10_xy_delta_t"]
                delta_xy_space = traj["action"]["future_10_xy_delta_space"]

                delta_course_t = (delta_course_t % 360.0 + 360.0) % 360.0
                delta_course_t = tf.where(delta_course_t > 180.0, delta_course_t - 360.0, delta_course_t)
                delta_course_space = (delta_course_space % 360.0 + 360.0) % 360.0
                delta_course_space = tf.where(delta_course_space > 180.0, delta_course_space - 360.0, delta_course_space)

                delta_speed_t_norm = 10.0
                delta_xy_t_norm = 7.0

                delta_speed_t = delta_speed_t / delta_speed_t_norm
                delta_course_t = delta_course_t / 180.0
                delta_course_space = delta_course_space / 180.0
                delta_xy_t = delta_xy_t / delta_xy_t_norm

                oaf = output_action_format
                if oaf == OutputActionFormat.DELTA_SPEED_T_DELTA_COURSE_T_DELTA_COURSE_SPACE:
                    actions = tf.concat([delta_speed_t[..., None], delta_course_t[..., None], delta_course_space[..., None]], axis=-1)
                elif oaf == OutputActionFormat.DELTA_XY_T_DELTA_XY_SPACE:
                    actions = tf.concat([delta_xy_t, delta_xy_space], axis=-1)
                elif oaf == OutputActionFormat.DELTA_XY_T_DELTA_COURSE_SPACE:
                    actions = tf.concat([delta_xy_t, delta_course_space[..., None]], axis=-1)
                else:
                    raise ValueError(f"Unknown output_action_format: {oaf}")

                actions = actions[:, :action_chunk_size, :]

                # Select language label
                llt = lang_label_type
                if llt == LangLabelType.COMMENTARY:
                    instruction = traj["commentary"]
                elif llt == LangLabelType.GEMINI_SHORTER:
                    instruction = traj["gemini_refined_label"]
                elif llt == LangLabelType.GEMINI_LONGER:
                    instruction = traj["gemini_refined_label_longer"]
                elif llt == LangLabelType.ROUTING_COMMAND:
                    instruction = tf.strings.regex_replace(traj["routing_command"], "^Command: ", "")
                else:
                    raise ValueError(f"Unknown lang_label_type: {llt}")

                if routing_command_in_prompt and llt != LangLabelType.ROUTING_COMMAND:
                    rc = tf.strings.regex_replace(traj["routing_command"], "^Command: ", "")
                    instruction = tf.strings.join([rc, instruction], separator=" ")

                if speed_in_prompt:
                    speed_str = tf.strings.as_string(current_speed_og, precision=1)
                    speed_prompt = tf.strings.join(["The current speed is ", speed_str, " m/s. "])
                    instruction = tf.strings.join([speed_prompt, instruction], separator="")

                if add_suffix_to_prompt:
                    if oaf == OutputActionFormat.DELTA_SPEED_T_DELTA_COURSE_T_DELTA_COURSE_SPACE:
                        suffix = " Predict future changes in driving speed and heading."
                    elif oaf == OutputActionFormat.DELTA_XY_T_DELTA_XY_SPACE:
                        suffix = " Predict the future driving waypoints."
                    elif oaf == OutputActionFormat.DELTA_XY_T_DELTA_COURSE_SPACE:
                        suffix = " Predict future driving waypoints and heading changes."
                    instruction = tf.strings.join([instruction, suffix], separator="")

                front_image = traj["observation"]["image"]

                result = {
                    "actions": actions,
                    "observation": {
                        "image": front_image,
                        "state": ego_state,
                        "current_speed": current_speed_og,
                    },
                    "prompt": instruction,
                }

                if enable_cot:
                    # routing_cmd = tf.strings.regex_replace(traj["routing_command"], "^Command: ", "")
                    result["subtask"] = traj["gemini_refined_label"]
                    result["reasoning"] = traj["commentary"]

                return result

            return restructure

        def prepare_single_dataset(dataset_cfg: SteerVLARLDSDataset):
            builder_kwargs = {"data_dir": data_dir}
            if dataset_cfg.version is not None:
                builder_kwargs["version"] = dataset_cfg.version
            builder = tfds.builder(dataset_cfg.name, **builder_kwargs)
            dataset = dl.DLataset.from_rlds(
                builder, split="train", shuffle=shuffle, num_parallel_reads=num_parallel_reads
            )
            dataset = dataset.repeat()

            if dataset_format == DatasetFormat.NUSCENES:
                restructure_fn = _build_nuscenes_restructure(tf)
            elif dataset_format == DatasetFormat.SIMLINGO:
                restructure_fn = _build_simlingo_restructure()
            else:
                raise ValueError(f"Unknown dataset_format: {dataset_format}")

            dataset = dataset.traj_map(restructure_fn, num_parallel_calls)

            if dataset_format == DatasetFormat.NUSCENES:
                # nuScenes: actions are already chunked in action_chunk field,
                # but may need trimming/padding to action_chunk_size.
                def chunk_actions(traj):
                    current_chunk_size = tf.shape(traj["actions"])[1]
                    if current_chunk_size >= action_chunk_size:
                        traj["actions"] = traj["actions"][:, :action_chunk_size, :]
                    else:
                        pad_size = action_chunk_size - current_chunk_size
                        last_actions = traj["actions"][:, -1:, :]
                        padding = tf.repeat(last_actions, pad_size, axis=1)
                        traj["actions"] = tf.concat([traj["actions"], padding], axis=1)
                    return traj

                dataset = dataset.traj_map(chunk_actions, num_parallel_calls)

            # SimLingo: actions are already trimmed to action_chunk_size in restructure.

            dataset = dataset.flatten(num_parallel_calls=num_parallel_calls)

            def decode_images(frame):
                frame["observation"]["image"] = tf.io.decode_image(
                    frame["observation"]["image"], expand_animations=False, dtype=tf.uint8
                )
                return frame

            return dataset.frame_map(decode_images, num_parallel_calls)

        logging.info(f"Preparing {len(datasets)} SteerVLA datasets ({dataset_format.name})...")
        logging.info("-" * 50)
        for ds, nw in zip(datasets, normalized_weights):
            ver = ds.version or "default"
            logging.info(f"    {ds.name} (v{ver}) weight={ds.weight:.3f} (normalized={nw:.4f})")
        logging.info("-" * 50)

        all_datasets = [prepare_single_dataset(ds) for ds in datasets]

        final_dataset = dl.DLataset.sample_from_datasets(all_datasets, weights=normalized_weights)
        if shuffle:
            final_dataset = final_dataset.shuffle(shuffle_buffer_size)
        final_dataset = shard_dataset_then_batch(final_dataset, batch_size)
        final_dataset = final_dataset.with_ram_budget(1)

        self.dataset = final_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        yield from self.dataset.as_numpy_iterator()

    def __len__(self):
        return 1_000_000
