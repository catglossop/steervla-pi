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


# Framing correction between SimLingo-derived RLDS builds.
#
# Every one of these datasets stores a 512x512 JPEG, but they were built from the same 1024x512
# CARLA front camera under two *different* framings:
#
#   simlingo_dataset_*_img512_1116  crop (170, 0, 852, 359) of the 1024x512 source, then squash to 512x512
#   simplified_reasoning_dataset    no crop -- the full 1024x512 frame squashed to 512x512
#
# So the high-level reasoning corpus shows a wider FOV and includes the ego hood, while the
# low-level corpus is zoomed in on the road ahead with the bottom 30% cut off. Mixing them trains
# the policy on two different cameras. Recovered by solving for the crop box that reproduces each
# stored image from its source file under ``simlingo_rgb_zips`` (6/6 samples agree exactly;
# mean abs residual 0.88/255, i.e. JPEG round-trip noise).
#
# Expressed normalized so it applies to the stored 512x512 image as well as the 1024x512 source:
# on a 512x512 image it is the box (85, 0, 426, 359).
SIMLINGO_FRAMING_CROP = (170 / 1024, 0.0, 852 / 1024, 359 / 512)

# Datasets whose stored images need cropping to match ``SIMLINGO_FRAMING_CROP``. Consulted only when
# a ``SteerVLARLDSDataset`` does not set ``image_crop`` itself. Datasets absent here are left alone,
# which is correct for the simlingo_dataset_* builds (already in that framing) and for any corpus
# whose framing has not been measured.
DATASET_IMAGE_CROPS: dict[str, tuple[float, float, float, float]] = {
    "simplified_reasoning_dataset": SIMLINGO_FRAMING_CROP,
}

# Datasets measured to already be in the canonical framing, so they need no correction. Kept
# separate from "we have never looked" -- see ``resolve_image_crop``.
CANONICAL_FRAMING_PREFIXES = ("simlingo_dataset_",)


def resolve_image_crop(
    dataset: "SteerVLARLDSDataset",
) -> tuple[tuple[float, float, float, float] | None, bool]:
    """Crop for a dataset, plus whether its framing is actually known.

    Every source in a mixture must reach the model through the same camera framing, so a corpus
    whose framing nobody has measured is a silent hazard rather than a safe default. Returning the
    ``known`` flag lets the caller say so out loud instead of quietly assuming canonical.
    """
    if dataset.image_crop is not None:
        return (dataset.image_crop or None), True
    if dataset.name in DATASET_IMAGE_CROPS:
        return DATASET_IMAGE_CROPS[dataset.name], True
    if dataset.name.startswith(CANONICAL_FRAMING_PREFIXES):
        return None, True
    return None, False


@dataclasses.dataclass
class SteerVLARLDSDataset:
    name: str
    weight: float = 1.0
    version: str | None = None
    # Normalized (x0, y0, x1, y1) crop applied to the stored image, then resized back to the stored
    # resolution so every source in the mixture batches at the same shape. ``None`` falls back to
    # ``DATASET_IMAGE_CROPS``; pass ``()`` to force no crop.
    image_crop: tuple[float, float, float, float] | None = None


class SteerVLARldsDataset:
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        datasets: Sequence[SteerVLARLDSDataset],
        *,
        dataset_format: DatasetFormat = DatasetFormat.NUSCENES,
        hl_datasets: Sequence[SteerVLARLDSDataset] = (),
        hl_dataset_format: DatasetFormat | None = None,
        # Action-only sources: the flow expert is supervised, the CoT cross-entropy is not. The CoT
        # segments are still built and fed as prefix context; only their loss is switched off.
        ll_datasets: Sequence[SteerVLARLDSDataset] = (),
        cot_reasoning_key: str = "commentary",
        cot_subtask_key: str = "gemini_refined_label",
        hl_cot_reasoning_key: str = "gemini_refined_label",
        hl_cot_subtask_key: str = "prompt",
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
        split: str = "train",
    ):
        import dlimp as dl
        import tensorflow as tf
        import tensorflow_datasets as tfds

        tf.config.set_visible_devices([], "GPU")

        if hl_datasets and not enable_cot:
            raise ValueError("High-level datasets require enable_cot=True.")

        # (dataset, format, action_supervision, cot_supervision, cot_reasoning_key, cot_subtask_key)
        # The three buckets are the three supervision modes in use:
        #   datasets    -> flow + CoT   (a full update)
        #   ll_datasets -> flow only    (action expert; CoT text is context, not a target)
        #   hl_datasets -> CoT only     (dummy actions, action_loss_mask all-False)
        # The same dataset name may appear in more than one bucket; each occurrence becomes its own
        # weighted source with its own dataset_id.
        source_specs: list[tuple[SteerVLARLDSDataset, DatasetFormat, bool, bool, str, str]] = [
            (d, dataset_format, True, True, cot_reasoning_key, cot_subtask_key) for d in datasets
        ]
        source_specs += [
            (d, dataset_format, True, False, cot_reasoning_key, cot_subtask_key) for d in ll_datasets
        ]
        source_specs += [
            (d, hl_dataset_format or dataset_format, False, True, hl_cot_reasoning_key, hl_cot_subtask_key)
            for d in hl_datasets
        ]

        # Weights are normalized internally by sample_from_datasets; no need to sum to 1.0.
        total_weight = sum(d.weight for d, *_ in source_specs)
        assert total_weight > 0, "Total dataset weight must be positive"
        normalized_weights = [d.weight / total_weight for d, *_ in source_specs]

        def _build_nuscenes_restructure(
            traj_map_tf=tf,
            *,
            action_supervision: bool = True,
        ):
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

                action_loss_mask = traj_map_tf.fill(
                    [traj_len, traj_map_tf.shape(actions)[1]],
                    traj_map_tf.cast(action_supervision, traj_map_tf.bool),
                )

                instruction = traj["language_instruction"]
                if speed_in_prompt:
                    speed_str = traj_map_tf.strings.as_string(current_speed)
                    speed_prompt = traj_map_tf.strings.join(["The current speed is ", speed_str, " m/s. "])
                    instruction = traj_map_tf.strings.join([speed_prompt, instruction], separator="")

                front_image = traj["observation"]["front_image"]

                return {
                    "actions": actions,
                    "action_loss_mask": action_loss_mask,
                    "observation": {
                        "image": front_image,
                        "state": ego_state,
                        "current_speed": current_speed,
                    },
                    "prompt": instruction,
                }

            return restructure

        def _build_simlingo_restructure(
            *,
            action_supervision: bool = True,
            cot_supervision: bool = True,
            cot_reasoning_source_key: str = "commentary",
            cot_subtask_source_key: str = "gemini_refined_label",
        ):
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
                action_loss_mask = tf.fill(
                    [traj_len, action_chunk_size], tf.cast(action_supervision, tf.bool)
                )

                # Select language label 
                llt = lang_label_type
                if llt == LangLabelType.COMMENTARY:
                    instruction = traj["commentary"]
                elif llt == LangLabelType.GEMINI_SHORTER:
                    instruction = traj["gemini_refined_label"]
                elif llt == LangLabelType.GEMINI_LONGER:
                    instruction = traj["gemini_refined_label_longer"]
                elif llt == LangLabelType.ROUTING_COMMAND:
                    instruction = tf.strings.regex_replace(traj["routing_command"], "(?i)^command:\\s*", "")
                else:
                    raise ValueError(f"Unknown lang_label_type: {llt}")

                if routing_command_in_prompt and llt != LangLabelType.ROUTING_COMMAND:
                    rc = tf.strings.regex_replace(traj["routing_command"], "(?i)^command:\\s*", "")
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
                    "action_loss_mask": action_loss_mask,
                    "observation": {
                        "image": front_image,
                        "state": ego_state,
                        "current_speed": current_speed_og,
                    },
                    "prompt": instruction,
                }

                if enable_cot:
                    result["subtask"] = traj[cot_subtask_source_key]
                    result["reasoning"] = traj[cot_reasoning_source_key]
                    # Per-frame gate on the CoT cross-entropy. The text above is still emitted and
                    # still becomes prefix context; this only decides whether it is a target.
                    result["cot_loss_mask"] = tf.fill([traj_len], tf.cast(cot_supervision, tf.bool))

                return result

            return restructure

        def _tag_dataset_id(restructure_fn, dataset_id: int):
            """Attach a constant ``dataset_id`` per frame for downstream eval grouping."""

            def restructure(traj):
                out = restructure_fn(traj)
                traj_len = tf.shape(out["actions"])[0]
                out["dataset_id"] = tf.fill([traj_len], tf.cast(dataset_id, tf.int32))
                return out

            return restructure

        def prepare_single_dataset(
            dataset_cfg: SteerVLARLDSDataset,
            source_dataset_format: DatasetFormat,
            *,
            action_supervision: bool,
            cot_supervision: bool,
            cot_reasoning_source_key: str,
            cot_subtask_source_key: str,
            dataset_id: int,
            split: str = "train",
        ):
            builder_kwargs = {"data_dir": data_dir}
            if dataset_cfg.version is not None:
                builder_kwargs["version"] = dataset_cfg.version
            builder = tfds.builder(dataset_cfg.name, **builder_kwargs)
            dataset = dl.DLataset.from_rlds(
                builder, split=split, shuffle=shuffle, num_parallel_reads=num_parallel_reads
            )
            dataset = dataset.repeat()

            if source_dataset_format == DatasetFormat.NUSCENES:
                restructure_fn = _build_nuscenes_restructure(tf, action_supervision=action_supervision)
            elif source_dataset_format == DatasetFormat.SIMLINGO:
                restructure_fn = _build_simlingo_restructure(
                    action_supervision=action_supervision,
                    cot_supervision=cot_supervision,
                    cot_reasoning_source_key=cot_reasoning_source_key,
                    cot_subtask_source_key=cot_subtask_source_key,
                )
            else:
                raise ValueError(f"Unknown dataset_format: {source_dataset_format}")

            restructure_fn = _tag_dataset_id(restructure_fn, dataset_id)
            dataset = dataset.traj_map(restructure_fn, num_parallel_calls)

            if source_dataset_format == DatasetFormat.NUSCENES:
                # nuScenes: actions are already chunked in action_chunk field,
                # but may need trimming/padding to action_chunk_size.
                def chunk_actions(traj):
                    current_chunk_size = tf.shape(traj["actions"])[1]
                    if current_chunk_size >= action_chunk_size:
                        traj["actions"] = traj["actions"][:, :action_chunk_size, :]
                        traj["action_loss_mask"] = traj["action_loss_mask"][:, :action_chunk_size]
                    else:
                        pad_size = action_chunk_size - current_chunk_size
                        last_actions = traj["actions"][:, -1:, :]
                        last_mask = traj["action_loss_mask"][:, -1:]
                        padding = tf.repeat(last_actions, pad_size, axis=1)
                        mask_padding = tf.repeat(last_mask, pad_size, axis=1)
                        traj["actions"] = tf.concat([traj["actions"], padding], axis=1)
                        traj["action_loss_mask"] = tf.concat([traj["action_loss_mask"], mask_padding], axis=1)
                    return traj

                dataset = dataset.traj_map(chunk_actions, num_parallel_calls)

            # SimLingo: actions are already trimmed to action_chunk_size in restructure.

            dataset = dataset.flatten(num_parallel_calls=num_parallel_calls)

            crop, _ = resolve_image_crop(dataset_cfg)

            def decode_images(frame, crop=crop):
                image = tf.io.decode_image(
                    frame["observation"]["image"], expand_animations=False, dtype=tf.uint8
                )
                if crop:
                    x0, y0, x1, y1 = crop
                    shape = tf.shape(image)
                    h = tf.cast(shape[0], tf.float32)
                    w = tf.cast(shape[1], tf.float32)
                    top = tf.cast(tf.round(y0 * h), tf.int32)
                    left = tf.cast(tf.round(x0 * w), tf.int32)
                    image = tf.image.crop_to_bounding_box(
                        image,
                        top,
                        left,
                        tf.cast(tf.round(y1 * h), tf.int32) - top,
                        tf.cast(tf.round(x1 * w), tf.int32) - left,
                    )
                    # Back to the stored resolution: sources in one mixture must batch at one shape.
                    image = tf.cast(
                        tf.round(tf.image.resize(image, [shape[0], shape[1]], antialias=True)), tf.uint8
                    )
                frame["observation"]["image"] = image
                return frame

            return dataset.frame_map(decode_images, num_parallel_calls)

        logging.info(f"Preparing {len(source_specs)} SteerVLA datasets...")
        logging.info("-" * 50)
        def _mode(*, action_sup: bool, cot_sup: bool) -> str:
            """Human-readable name for a (flow, CoT) supervision pair."""
            return {(True, True): "flow+cot", (True, False): "flow_only", (False, True): "cot_only"}.get(
                (action_sup, cot_sup), "unsupervised"
            )

        for (ds, ds_format, action_supervision, cot_supervision, cot_reason_key, cot_subtask_key), nw in zip(
            source_specs, normalized_weights
        ):
            ver = ds.version or "default"
            ds_crop, crop_known = resolve_image_crop(ds)
            if not crop_known:
                logging.warning(
                    f"    {ds.name}: image framing has not been measured, so it is being fed to the model "
                    f"uncropped. If it was not built with the same crop as the simlingo_dataset_* corpora "
                    f"({tuple(round(v, 4) for v in SIMLINGO_FRAMING_CROP)} of the source frame), this mixes "
                    f"two cameras in one batch. Add an entry to DATASET_IMAGE_CROPS or set image_crop=() "
                    f"to confirm it needs no crop."
                )
            logging.info(
                f"    {ds.name} (v{ver}) format={ds_format.name} "
                f"cot_reasoning_key={cot_reason_key} cot_subtask_key={cot_subtask_key} "
                f"mode={_mode(action_sup=action_supervision, cot_sup=cot_supervision)} "
                f"weight={ds.weight:.3f} (normalized={nw:.4f}) "
                f"image_crop={'none' if not ds_crop else tuple(round(v, 4) for v in ds_crop)}"
            )
        logging.info("-" * 50)

        # A dataset can appear in several buckets, so tag the mode to keep per-dataset eval metric
        # keys distinct (see _short_dataset_name / _compute_per_dataset_loss_metrics).
        self.dataset_names = [
            ds.name if _mode(action_sup=a, cot_sup=c) == "flow+cot" else f"{ds.name}#{_mode(action_sup=a, cot_sup=c)}"
            for ds, _, a, c, _, _ in source_specs
        ]

        all_datasets = [
            prepare_single_dataset(
                ds,
                ds_format,
                action_supervision=action_supervision,
                cot_supervision=cot_supervision,
                cot_reasoning_source_key=cot_reason_key,
                cot_subtask_source_key=cot_subtask_key,
                dataset_id=dataset_id,
                split=split,
            )
            for dataset_id, (
                ds,
                ds_format,
                action_supervision,
                cot_supervision,
                cot_reason_key,
                cot_subtask_key,
            ) in enumerate(source_specs)
        ]

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
