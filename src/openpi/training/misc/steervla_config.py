"""SteerVLA / SimLingo driving configs (fork-specific).

Registered into ``openpi.training.config._CONFIGS`` the same way as the RoboArena and PolaRiS
configs, so upstream's config list stays untouched. The data-config factories these use
(``RLDSSteerVLADataConfig``, ``RLDSSteerVLACoTDataConfig``, ``LeRobotSteerVLADataConfig``) live in
``config.py`` next to upstream's factories.
"""

from typing import TypeAlias

import flax.nnx as nnx

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.policies.steervla_policy as steervla_policy
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.optimizer as _optimizer
import openpi.training.steervla_rlds_dataset as steervla_rlds_dataset
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType

# Relative sampling weights across the SimLingo scenario datasets. Shared by every SteerVLA
# mixture so the scenario balance stays in one place; ``_mixture_share`` rescales the whole set to
# a target share of the batch without disturbing the ratios between them.
_SIMLINGO_SCENARIO_WEIGHTS = {
    "simlingo_dataset_all_img512_1116": 1.0,
    "simlingo_dataset_acceleration_negative5_img512_1116": 0.4,
    "simlingo_dataset_acceleration_negative1_img512_1116": 0.2,
    "simlingo_dataset_acceleration_positive1_img512_1116": 0.1,
    "simlingo_dataset_acceleration_positive5_img512_1116": 0.2,
    "simlingo_dataset_lateral_control12_img512_1116": 0.1,
    "simlingo_dataset_lateral_control_higher5_img512_1116": 0.3,
    "simlingo_dataset_start_from_stop_img512_1116": 0.2,
    "simlingo_dataset_vehicle_front_img512_1116": 0.3,
    "simlingo_dataset_vehicle_side_img512_1116": 0.1,
    "simlingo_dataset_leading_object_vehicle_img512_1116": 0.05,
    "simlingo_dataset_leading_object_traffic_stop_img512_1116": 0.2,
    "simlingo_dataset_leading_object_traffic_light_img512_1116": 0.2,
    "simlingo_dataset_leading_object_walker_img512_1116": 0.2,
    "simlingo_dataset_changed_route_img512_1116": 0.2,
    "simlingo_dataset_parking_lane_img512_1116": 0.3,
}


def _mixture_share(weights: dict[str, float], share: float) -> dict[str, float]:
    """Rescale ``weights`` so they sum to ``share`` of the batch, preserving their ratios.

    ``SteerVLARldsDataset`` normalises across every bucket it is given, so expressing each bucket
    as its intended fraction of the batch makes the mixture readable at the call site: the shares
    across buckets simply sum to 1.
    """
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Mixture weights must sum to a positive value")
    return {name: weight / total * share for name, weight in weights.items()}


def get_steervla_configs():
    # Import here to avoid circular imports.
    from openpi.training.config import AssetsConfig
    from openpi.training.config import DataConfig
    from openpi.training.config import LeRobotSteerVLADataConfig
    from openpi.training.config import RLDSSteerVLACoTDataConfig
    from openpi.training.config import RLDSSteerVLADataConfig
    from openpi.training.config import SimpleDataConfig
    from openpi.training.config import TrainConfig

    configs = [
        #
        # SteerVLA (nuScenes driving) configs.
        #
        TrainConfig(
            name="pi05_steervla",
            model=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=6,
            ),
            data=RLDSSteerVLADataConfig(
                repo_id="steervla",
                rlds_data_dir="<path_to_nuscenes_rlds_dataset>",
                include_ego_history=True,
                include_xy_action=False,
                speed_in_prompt=True,
                proprio_norm=True,
                action_dim=2,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_000,
                peak_lr=1e-4,
                decay_steps=100_000,
                decay_lr=1e-5,
            ),
            num_train_steps=5_000,
            batch_size=96,
            log_interval=100,
            save_interval=500,
            num_workers=0,
        ),
        TrainConfig(
            name="pi0_steervla",
            model=pi0_config.Pi0Config(
                action_dim=32,
                action_horizon=6,
            ),
            data=RLDSSteerVLADataConfig(
                repo_id="steervla",
                rlds_data_dir="<path_to_nuscenes_rlds_dataset>",
                include_ego_history=True,
                include_xy_action=False,
                speed_in_prompt=True,
                proprio_norm=True,
                action_dim=2,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_000,
                peak_lr=1e-4,
                decay_steps=100_000,
                decay_lr=1e-5,
            ),
            num_train_steps=5_000,
            batch_size=96,
            log_interval=100,
            save_interval=500,
            num_workers=0,
        ),
        TrainConfig(
            name="pi0_fast_steervla",
            model=pi0_fast.Pi0FASTConfig(
                action_dim=2,
                action_horizon=6,
                max_token_len=180,
            ),
            data=RLDSSteerVLADataConfig(
                repo_id="steervla",
                rlds_data_dir="<path_to_nuscenes_rlds_dataset>",
                include_ego_history=True,
                include_xy_action=False,
                speed_in_prompt=True,
                proprio_norm=True,
                action_dim=2,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_000,
                peak_lr=1e-4,
                decay_steps=100_000,
                decay_lr=1e-5,
            ),
            num_train_steps=5_000,
            batch_size=96,
            log_interval=100,
            save_interval=500,
            num_workers=0,
        ),
        #
        # SteerVLA SimLingo configs (multi-dataset with sampling weights).
        #
        TrainConfig(
            name="pi05_steervla_simlingo",
            model=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=10,
            ),
            data=RLDSSteerVLADataConfig(
                repo_id="steervla_simlingo",
                rlds_data_dir="gs://tian-us-central2/tensorflow_datasets",
                dataset_format=steervla_rlds_dataset.DatasetFormat.SIMLINGO,
                include_ego_history=False,
                include_xy_action=False,
                speed_in_prompt=True,
                proprio_norm=True,
                action_dim=4,
                output_action_format=steervla_rlds_dataset.OutputActionFormat.DELTA_XY_T_DELTA_XY_SPACE,
                lang_label_type=steervla_rlds_dataset.LangLabelType.ROUTING_COMMAND,
                routing_command_in_prompt=False,
                add_suffix_to_prompt=False,
                dataset_name_weight_mappings={
                    "simlingo_dataset_all_img512_0916": 1.0,
                    "simlingo_dataset_acceleration_negative5_img512_0916": 0.2,
                    "simlingo_dataset_acceleration_negative1_img512_0916": 0.1,
                    "simlingo_dataset_acceleration_positive1_img512_0916": 0.1,
                    "simlingo_dataset_acceleration_positive5_img512_0916": 0.2,
                    "simlingo_dataset_lateral_control12_img512_0916": 0.1,
                    "simlingo_dataset_lateral_control_higher5_img512_0916": 0.3,
                    "simlingo_dataset_start_from_stop_img512_0916": 0.2,
                    "simlingo_dataset_vehicle_front_img512_0916": 0.3,
                    "simlingo_dataset_vehicle_side_img512_0916": 0.1,
                    "simlingo_dataset_leading_object_vehicle_img512_0916": 0.05,
                    "simlingo_dataset_leading_object_traffic_stop_img512_0916": 0.2,
                    "simlingo_dataset_leading_object_traffic_light_img512_0916": 0.2,
                    "simlingo_dataset_leading_object_walker_img512_0916": 0.2,
                    "simlingo_dataset_changed_route_img512_0916": 0.2,
                    "simlingo_dataset_parkinglane_img512_0916": 0.3,
                },
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_000,
                peak_lr=1e-4,
                decay_steps=100_000,
                decay_lr=1e-5,
            ),
            num_train_steps=100_000,
            batch_size=24,
            fsdp_devices=4,
            log_interval=1,
            save_interval=5000,
            num_workers=0,
            checkpoint_base_dir="gs://cat-logs",
        ),
        #
        # SteerVLA Chain-of-Thought: Pi0.5 with reasoning + subtask generation.
        # Uses routing_command as high-level prompt, gemini_refined_label as subtask,
        # and commentary as reasoning (chain-of-thought).
        #
        TrainConfig(
            name="pi05_steervla_cot_fast",
            model=pi0_config.Pi0CoTConfig(
                action_dim=32,
                action_horizon=10,
                max_token_len=112,
                max_subtask_len=48,
                max_reasoning_len=96,
                max_fast_len=64,
                cot_loss_weight=1.0,
                use_fast_tokens=True,
                knowledge_insulation=False,
            ),
            data=RLDSSteerVLACoTDataConfig(
                repo_id="steervla_simlingo_cot",
                rlds_data_dir="gs://tian-us-central2/tensorflow_datasets",
                dataset_format=steervla_rlds_dataset.DatasetFormat.SIMLINGO,
                include_ego_history=False,
                include_xy_action=False,
                speed_in_prompt=True,
                proprio_norm=True,
                action_dim=4,
                output_action_format=steervla_rlds_dataset.OutputActionFormat.DELTA_XY_T_DELTA_XY_SPACE,
                lang_label_type=steervla_rlds_dataset.LangLabelType.ROUTING_COMMAND,
                dataset_name_weight_mappings={
                    "simlingo_dataset_all_img512_1116": 1.0,
                },
                max_subtask_len=48,
                max_reasoning_len=96,
                max_fast_len=64,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_000,
                peak_lr=1e-4,
                decay_steps=100_000,
                decay_lr=1e-5,
            ),
            num_train_steps=100_000,
            batch_size=24,
            fsdp_devices=4,
            log_interval=1,
            save_interval=5000,
            num_workers=0,
            checkpoint_base_dir="gs://cat-logs",
        ),
        TrainConfig(
            name="pi05_steervla_cot",
            model=pi0_config.Pi0CoTConfig(
                action_dim=32,
                action_horizon=10,
                max_token_len=112,
                max_subtask_len=48,
                max_reasoning_len=96,
                cot_loss_weight=1.0,
                knowledge_insulation=False,
            ),
            data=RLDSSteerVLACoTDataConfig(
                repo_id="steervla_simlingo_cot",
                rlds_data_dir="gs://tian-us-central2/tensorflow_datasets",
                dataset_format=steervla_rlds_dataset.DatasetFormat.SIMLINGO,
                include_ego_history=False,
                include_xy_action=False,
                speed_in_prompt=True,
                proprio_norm=True,
                action_dim=4,
                output_action_format=steervla_rlds_dataset.OutputActionFormat.DELTA_XY_T_DELTA_XY_SPACE,
                lang_label_type=steervla_rlds_dataset.LangLabelType.ROUTING_COMMAND,
                dataset_name_weight_mappings={
                    "simlingo_dataset_all_img512_1116": 1.0,
                    "simlingo_dataset_acceleration_negative5_img512_1116": 0.2,
                    "simlingo_dataset_acceleration_negative1_img512_1116": 0.1,
                    "simlingo_dataset_acceleration_positive1_img512_1116": 0.1,
                    "simlingo_dataset_acceleration_positive5_img512_1116": 0.2,
                    "simlingo_dataset_lateral_control12_img512_1116": 0.1,
                    "simlingo_dataset_lateral_control_higher5_img512_1116": 0.3,
                    "simlingo_dataset_start_from_stop_img512_1116": 0.2,
                    "simlingo_dataset_vehicle_front_img512_1116": 0.3,
                    "simlingo_dataset_vehicle_side_img512_1116": 0.1,
                    "simlingo_dataset_leading_object_vehicle_img512_1116": 0.05,
                    "simlingo_dataset_leading_object_traffic_stop_img512_1116": 0.2,
                    "simlingo_dataset_leading_object_traffic_light_img512_1116": 0.2,
                    "simlingo_dataset_leading_object_walker_img512_1116": 0.2,
                    "simlingo_dataset_changed_route_img512_1116": 0.2,
                    "simlingo_dataset_parking_lane_img512_1116": 0.3,
                },
                max_subtask_len=48,
                max_reasoning_len=96,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_000,
                peak_lr=1e-4,
                decay_steps=100_000,
                decay_lr=1e-5,
            ),
            num_train_steps=100_000,
            batch_size=24,
            fsdp_devices=4,
            log_interval=1,
            save_interval=5000,
            num_workers=0,
            checkpoint_base_dir="gs://cat-logs",
        ),
        TrainConfig(
            name="pi05_steervla_cot_ki",
            exp_name="pi05_steervla_cot_ki",
            model=pi0_config.Pi0CoTConfig(
                action_dim=32,
                action_horizon=10,
                max_token_len=112,
                max_subtask_len=48,
                max_reasoning_len=96,
                cot_loss_weight=1.0,
                knowledge_insulation=True,
            ),
            data=RLDSSteerVLACoTDataConfig(
                repo_id="steervla_simlingo_cot",
                # rlds_data_dir="gs://tian-us-central2/tensorflow_datasets",
                rlds_data_dir="/raid/datasets/steervla",
                dataset_format=steervla_rlds_dataset.DatasetFormat.SIMLINGO,
                include_ego_history=False,
                include_xy_action=False,
                speed_in_prompt=True,
                proprio_norm=True,
                action_dim=4,
                output_action_format=steervla_rlds_dataset.OutputActionFormat.DELTA_XY_T_DELTA_XY_SPACE,
                lang_label_type=steervla_rlds_dataset.LangLabelType.ROUTING_COMMAND,
                dataset_name_weight_mappings={
                    "simlingo_dataset_all_img512_1116": 1.0,
                    "simlingo_dataset_acceleration_negative5_img512_1116": 0.2,
                    "simlingo_dataset_acceleration_negative1_img512_1116": 0.1,
                    "simlingo_dataset_acceleration_positive1_img512_1116": 0.1,
                    "simlingo_dataset_acceleration_positive5_img512_1116": 0.2,
                    "simlingo_dataset_lateral_control12_img512_1116": 0.1,
                    "simlingo_dataset_lateral_control_higher5_img512_1116": 0.3,
                    "simlingo_dataset_start_from_stop_img512_1116": 0.2,
                    "simlingo_dataset_vehicle_front_img512_1116": 0.3,
                    "simlingo_dataset_vehicle_side_img512_1116": 0.1,
                    "simlingo_dataset_leading_object_vehicle_img512_1116": 0.05,
                    "simlingo_dataset_leading_object_traffic_stop_img512_1116": 0.2,
                    "simlingo_dataset_leading_object_traffic_light_img512_1116": 0.2,
                    "simlingo_dataset_leading_object_walker_img512_1116": 0.2,
                    "simlingo_dataset_changed_route_img512_1116": 0.2,
                    "simlingo_dataset_parking_lane_img512_1116": 0.3,
                },
                max_subtask_len=48,
                max_reasoning_len=96,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_000,
                peak_lr=1e-4,
                decay_steps=100_000,
                decay_lr=1e-5,
            ),
            num_train_steps=200_000,
            batch_size=512,
            fsdp_devices=8,
            log_interval=1,
            save_interval=5000,
            num_workers=0,
            checkpoint_base_dir="gs://cat-logs",
            # Resume from the step-90000 checkpoint of the previous run. The run dir
            # is the parent of the step subdir (i.e., gs://.../pi05_steervla_cot_ki).
            # The bucket also has later steps (95000, 99999) so resume_step pins to
            # 90000 explicitly instead of letting orbax pick the latest.
            resume=True,
            resume_dir="gs://cat-logs/pi05_steervla_cot_ki/pi05_steervla_cot_ki",
            resume_step=99999,
        ),
        TrainConfig(
            name="pi05_steervla_cot_ki_inference",
            model=pi0_config.Pi0CoTConfig(
                action_dim=32,
                action_horizon=10,
                max_token_len=112,
                max_subtask_len=48,
                max_reasoning_len=96,
                cot_loss_weight=1.0,
                knowledge_insulation=True,
                inference_image_keys=("base_0_rgb",),
            ),
            data=RLDSSteerVLACoTDataConfig(
                repo_id="steervla_simlingo_cot",
                rlds_data_dir="gs://tian-us-central2/tensorflow_datasets",
                dataset_format=steervla_rlds_dataset.DatasetFormat.SIMLINGO,
                include_ego_history=False,
                include_xy_action=False,
                speed_in_prompt=True,
                proprio_norm=True,
                action_dim=4,
                output_action_format=steervla_rlds_dataset.OutputActionFormat.DELTA_XY_T_DELTA_XY_SPACE,
                lang_label_type=steervla_rlds_dataset.LangLabelType.ROUTING_COMMAND,
                dataset_name_weight_mappings={
                    "simlingo_dataset_all_img512_1116": 1.0,
                    "simlingo_dataset_acceleration_negative5_img512_1116": 0.2,
                    "simlingo_dataset_acceleration_negative1_img512_1116": 0.1,
                    "simlingo_dataset_acceleration_positive1_img512_1116": 0.1,
                    "simlingo_dataset_acceleration_positive5_img512_1116": 0.2,
                    "simlingo_dataset_lateral_control12_img512_1116": 0.1,
                    "simlingo_dataset_lateral_control_higher5_img512_1116": 0.3,
                    "simlingo_dataset_start_from_stop_img512_1116": 0.2,
                    "simlingo_dataset_vehicle_front_img512_1116": 0.3,
                    "simlingo_dataset_vehicle_side_img512_1116": 0.1,
                    "simlingo_dataset_leading_object_vehicle_img512_1116": 0.05,
                    "simlingo_dataset_leading_object_traffic_stop_img512_1116": 0.2,
                    "simlingo_dataset_leading_object_traffic_light_img512_1116": 0.2,
                    "simlingo_dataset_leading_object_walker_img512_1116": 0.2,
                    "simlingo_dataset_changed_route_img512_1116": 0.2,
                    "simlingo_dataset_parking_lane_img512_1116": 0.3,
                },
                max_subtask_len=48,
                max_reasoning_len=96,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            # Train only action_out_proj + time_mlp*; freeze everything else.
            # Since TrainConfig.trainable_filter = All(Param, Not(freeze_filter)),
            # using Not(PathRegex("...")) as freeze_filter keeps only regex-matching params trainable.
            freeze_filter=nnx.Not(nnx_utils.PathRegex(".*(action_out_proj|time_mlp).*")),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_000,
                peak_lr=1e-4,
                decay_steps=100_000,
                decay_lr=1e-5,
            ),
            num_train_steps=100_000,
            batch_size=24,
            fsdp_devices=1,
            log_interval=1,
            save_interval=5000,
            num_workers=0,
            checkpoint_base_dir="gs://cat-logs",
            ema_decay=None,
        ),
        TrainConfig(
            name="pi05_steervla_cot_simplified_reasoning",
            model=pi0_config.Pi0CoTConfig(
                action_dim=32,
                action_horizon=10,
                max_token_len=200,
                max_subtask_len=64,
                max_reasoning_len=64,
                cot_loss_weight=0.1,
                knowledge_insulation=False,
                use_fast_tokens=True,
            ),
            data=RLDSSteerVLACoTDataConfig(
                repo_id="steervla_simlingo_cot",
                rlds_data_dir="/raid/datasets/steervla",
                dataset_format=steervla_rlds_dataset.DatasetFormat.SIMLINGO,
                include_ego_history=False,
                include_xy_action=False,
                speed_in_prompt=True,
                proprio_norm=False,
                action_dim=4,
                output_action_format=steervla_rlds_dataset.OutputActionFormat.DELTA_XY_T_DELTA_XY_SPACE,
                lang_label_type=steervla_rlds_dataset.LangLabelType.ROUTING_COMMAND,
                dataset_name_weight_mappings={
                    "simlingo_dataset_all_img512_1116": 1.0,
                    "simlingo_dataset_acceleration_negative5_img512_1116": 0.4,
                    "simlingo_dataset_acceleration_negative1_img512_1116": 0.2,
                    "simlingo_dataset_acceleration_positive1_img512_1116": 0.1,
                    "simlingo_dataset_acceleration_positive5_img512_1116": 0.2,
                    "simlingo_dataset_lateral_control12_img512_1116": 0.1,
                    "simlingo_dataset_lateral_control_higher5_img512_1116": 0.3,
                    "simlingo_dataset_start_from_stop_img512_1116": 0.2,
                    "simlingo_dataset_vehicle_front_img512_1116": 0.3,
                    "simlingo_dataset_vehicle_side_img512_1116": 0.1,
                    "simlingo_dataset_leading_object_vehicle_img512_1116": 0.05,
                    "simlingo_dataset_leading_object_traffic_stop_img512_1116": 0.2,
                    "simlingo_dataset_leading_object_traffic_light_img512_1116": 0.2,
                    "simlingo_dataset_leading_object_walker_img512_1116": 0.2,
                    "simlingo_dataset_changed_route_img512_1116": 0.2,
                    "simlingo_dataset_parking_lane_img512_1116": 0.3,
                },
                hl_dataset_name_weight_mappings={
                    "simplified_reasoning_dataset": 1.85,
                },
                hl_dataset_format=steervla_rlds_dataset.DatasetFormat.SIMLINGO,
                hl_cot_reasoning_key="gemini_refined_label",
                hl_cot_subtask_key="prompt",
                max_subtask_len=64,
                max_reasoning_len=64,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_000,
                peak_lr=2e-5,
                decay_steps=200_000,
                decay_lr=1e-5,
            ),
            num_train_steps=200_000,
            batch_size=512,
            fsdp_devices=3,
            log_interval=1,
            save_interval=2000,
            num_workers=0,
            checkpoint_base_dir="gs://cat-logs",
            resume=False,
            skip_norm_stats=True,
        ),
        # Same as ``pi05_steervla_cot_simplified_reasoning`` but conditions on ego history: the RLDS
        # loader emits the full 4-step [speed, course] history (8 state dims) and those dims are
        # embedded in the prompt (``prompt_state_dim=8``, via ``include_ego_history=True``).
        TrainConfig(
            name="pi05_steervla_cot_simplified_reasoning_ego_history",
            model=pi0_config.Pi0CoTConfig(
                action_dim=32,
                action_horizon=10,
                max_token_len=200,
                max_subtask_len=64,
                max_reasoning_len=64,
                cot_loss_weight=0.1,
                knowledge_insulation=False,
                use_fast_tokens=True,
            ),
            data=RLDSSteerVLACoTDataConfig(
                repo_id="steervla_simlingo_cot",
                rlds_data_dir="/raid/datasets/steervla",
                dataset_format=steervla_rlds_dataset.DatasetFormat.SIMLINGO,
                include_ego_history=True,
                include_xy_action=False,
                speed_in_prompt=True,
                proprio_norm=False,
                action_dim=4,
                output_action_format=steervla_rlds_dataset.OutputActionFormat.DELTA_XY_T_DELTA_XY_SPACE,
                lang_label_type=steervla_rlds_dataset.LangLabelType.ROUTING_COMMAND,
                dataset_name_weight_mappings={
                    "simlingo_dataset_all_img512_1116": 1.0,
                    "simlingo_dataset_acceleration_negative5_img512_1116": 0.4,
                    "simlingo_dataset_acceleration_negative1_img512_1116": 0.2,
                    "simlingo_dataset_acceleration_positive1_img512_1116": 0.1,
                    "simlingo_dataset_acceleration_positive5_img512_1116": 0.2,
                    "simlingo_dataset_lateral_control12_img512_1116": 0.1,
                    "simlingo_dataset_lateral_control_higher5_img512_1116": 0.3,
                    "simlingo_dataset_start_from_stop_img512_1116": 0.2,
                    "simlingo_dataset_vehicle_front_img512_1116": 0.3,
                    "simlingo_dataset_vehicle_side_img512_1116": 0.1,
                    "simlingo_dataset_leading_object_vehicle_img512_1116": 0.05,
                    "simlingo_dataset_leading_object_traffic_stop_img512_1116": 0.2,
                    "simlingo_dataset_leading_object_traffic_light_img512_1116": 0.2,
                    "simlingo_dataset_leading_object_walker_img512_1116": 0.2,
                    "simlingo_dataset_changed_route_img512_1116": 0.2,
                    "simlingo_dataset_parking_lane_img512_1116": 0.3,
                },
                hl_dataset_name_weight_mappings={
                    "simplified_reasoning_dataset": 1.85,
                },
                hl_dataset_format=steervla_rlds_dataset.DatasetFormat.SIMLINGO,
                hl_cot_reasoning_key="gemini_refined_label",
                hl_cot_subtask_key="prompt",
                max_subtask_len=64,
                max_reasoning_len=64,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_000,
                peak_lr=2e-5,
                decay_steps=200_000,
                decay_lr=1e-5,
            ),
            num_train_steps=200_000,
            batch_size=192,
            fsdp_devices=3,
            log_interval=1,
            save_interval=2000,
            num_workers=0,
            checkpoint_base_dir="gs://cat-logs",
            resume=False,
            skip_norm_stats=True,
        ),
        TrainConfig(
            name="pi05_steervla_cot_simplified_reasoning_no_ego_history",
            model=pi0_config.Pi0CoTConfig(
                action_dim=32,
                action_horizon=10,
                max_token_len=200,
                max_subtask_len=64,
                max_reasoning_len=64,
                cot_loss_weight=0.1,
                knowledge_insulation=False,
                use_fast_tokens=True,
            ),
            data=RLDSSteerVLACoTDataConfig(
                repo_id="steervla_simlingo_cot",
                rlds_data_dir="/raid/datasets/steervla",
                dataset_format=steervla_rlds_dataset.DatasetFormat.SIMLINGO,
                include_ego_history=False,
                include_xy_action=False,
                speed_in_prompt=True,
                proprio_norm=False,
                action_dim=4,
                output_action_format=steervla_rlds_dataset.OutputActionFormat.DELTA_XY_T_DELTA_XY_SPACE,
                lang_label_type=steervla_rlds_dataset.LangLabelType.ROUTING_COMMAND,
                dataset_name_weight_mappings={
                    "simlingo_dataset_all_img512_1116": 1.0,
                    "simlingo_dataset_acceleration_negative5_img512_1116": 0.4,
                    "simlingo_dataset_acceleration_negative1_img512_1116": 0.2,
                    "simlingo_dataset_acceleration_positive1_img512_1116": 0.1,
                    "simlingo_dataset_acceleration_positive5_img512_1116": 0.2,
                    "simlingo_dataset_lateral_control12_img512_1116": 0.1,
                    "simlingo_dataset_lateral_control_higher5_img512_1116": 0.3,
                    "simlingo_dataset_start_from_stop_img512_1116": 0.2,
                    "simlingo_dataset_vehicle_front_img512_1116": 0.3,
                    "simlingo_dataset_vehicle_side_img512_1116": 0.1,
                    "simlingo_dataset_leading_object_vehicle_img512_1116": 0.05,
                    "simlingo_dataset_leading_object_traffic_stop_img512_1116": 0.2,
                    "simlingo_dataset_leading_object_traffic_light_img512_1116": 0.2,
                    "simlingo_dataset_leading_object_walker_img512_1116": 0.2,
                    "simlingo_dataset_changed_route_img512_1116": 0.2,
                    "simlingo_dataset_parking_lane_img512_1116": 0.3,
                },
                hl_dataset_name_weight_mappings={
                    "simplified_reasoning_dataset": 1.85,
                },
                hl_dataset_format=steervla_rlds_dataset.DatasetFormat.SIMLINGO,
                hl_cot_reasoning_key="gemini_refined_label",
                hl_cot_subtask_key="prompt",
                max_subtask_len=64,
                max_reasoning_len=64,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_000,
                peak_lr=2e-5,
                decay_steps=200_000,
                decay_lr=1e-5,
            ),
            num_train_steps=200_000,
            batch_size=192,
            fsdp_devices=3,
            log_interval=1,
            save_interval=2000,
            max_to_keep=10,
            num_workers=0,
            checkpoint_base_dir="gs://cat-logs",
            resume=False,
            skip_norm_stats=True,
        ),
        TrainConfig(
            name="pi05_steervla_cot_simplified_reasoning_commentary",
            model=pi0_config.Pi0CoTConfig(
                action_dim=32,
                action_horizon=10,
                max_token_len=200,
                max_subtask_len=64,
                max_reasoning_len=64,
                cot_loss_weight=0.1,
                knowledge_insulation=False,
                use_fast_tokens=True,
            ),
            data=RLDSSteerVLACoTDataConfig(
                repo_id="steervla_simlingo_cot",
                rlds_data_dir="/raid/datasets/steervla",
                dataset_format=steervla_rlds_dataset.DatasetFormat.SIMLINGO,
                include_ego_history=False,
                include_xy_action=False,
                speed_in_prompt=True,
                proprio_norm=False,
                action_dim=4,
                output_action_format=steervla_rlds_dataset.OutputActionFormat.DELTA_XY_T_DELTA_XY_SPACE,
                lang_label_type=steervla_rlds_dataset.LangLabelType.ROUTING_COMMAND,
                dataset_name_weight_mappings={
                    "simlingo_dataset_all_img512_1116": 1.0,
                    "simlingo_dataset_acceleration_negative5_img512_1116": 0.4,
                    "simlingo_dataset_acceleration_negative1_img512_1116": 0.2,
                    "simlingo_dataset_acceleration_positive1_img512_1116": 0.1,
                    "simlingo_dataset_acceleration_positive5_img512_1116": 0.2,
                    "simlingo_dataset_lateral_control12_img512_1116": 0.1,
                    "simlingo_dataset_lateral_control_higher5_img512_1116": 0.3,
                    "simlingo_dataset_start_from_stop_img512_1116": 0.2,
                    "simlingo_dataset_vehicle_front_img512_1116": 0.3,
                    "simlingo_dataset_vehicle_side_img512_1116": 0.1,
                    "simlingo_dataset_leading_object_vehicle_img512_1116": 0.05,
                    "simlingo_dataset_leading_object_traffic_stop_img512_1116": 0.2,
                    "simlingo_dataset_leading_object_traffic_light_img512_1116": 0.2,
                    "simlingo_dataset_leading_object_walker_img512_1116": 0.2,
                    "simlingo_dataset_changed_route_img512_1116": 0.2,
                    "simlingo_dataset_parking_lane_img512_1116": 0.3,
                },
                hl_dataset_name_weight_mappings={
                    "simplified_reasoning_dataset": 1.85,
                },
                hl_dataset_format=steervla_rlds_dataset.DatasetFormat.SIMLINGO,
                hl_cot_reasoning_key="commentary",
                hl_cot_subtask_key="prompt",
                max_subtask_len=64,
                max_reasoning_len=64,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_000,
                peak_lr=2e-5,
                decay_steps=200_000,
                decay_lr=1e-5,
            ),
            num_train_steps=200_000,
            batch_size=192,
            fsdp_devices=3,
            log_interval=1,
            save_interval=2000,
            max_to_keep=10,
            num_workers=0,
            checkpoint_base_dir="gs://cat-logs",
            resume=False,
            skip_norm_stats=True,
        ),
        #
        # Baseline: ``pi05_steervla_cot_simplified_reasoning_no_ego_history``. Data mixture, action
        # format, LR schedule and batch size are copied verbatim so this is a clean A/B against it.
        # Four things change:
        #
        # 1. HL reasoning target reverted to ``gemini_refined_label``. In simplified_reasoning_dataset
        #    that key holds traffic_light_status (4 distinct values), not the meta-action narration it
        #    names in the simlingo_dataset_* family. ``commentary`` (the
        #    ..._commentary variant) put a second, much richer text distribution on the reasoning head
        #    with nothing in the input to disambiguate it from the LL family's ``commentary``, which is
        #    38% the literal string "Follow the route.".
        #
        # 2. ``skip_norm_stats=False`` -- actions and state are actually normalized. Without it the
        #    model trains on the fixed divisors in steervla_rlds_dataset.py, which leave the four
        #    action dims on unrelated scales: dim0 std 0.196, dim1 std 0.074, dim2 mean 0.954 /
        #    std 0.096 (delta_xy_space is never divided at all), dim3 std 0.272. Flow loss is a
        #    uniform MSE over dims, so per-dim gradient tracks per-dim variance -- see the table
        #    below; unnormalized, dim3 takes 59% of the budget and time-domain lateral control
        #    (dim1) gets 7%. Normalizing brings the spread from 2.9x down to 1.8x.
        #    It also fixes the state channel: ``CoTPaligemmaTokenizer.tokenize_prompt`` discretizes
        #    state against fixed [-1, 1] bins, so with proprio_norm=False every speed >= 1 m/s
        #    saturated to token 255 and every course <= -1 deg produced token -1.
        #
        #    REQUIRES, before the first training step:
        #        uv run --group rlds scripts/compute_norm_stats.py \
        #            --config-name pi05_steervla_cot_simplified_reasoning_norm
        #    Stats land in assets/<config name>/<repo_id>/, so they are already isolated from the rest
        #    of the family; the distinct repo_id just makes the normalized variant obvious on disk.
        #    Renaming this config orphans its stats -- point AssetsConfig at the old name instead.
        #
        # 3. Single camera + right-sized token budgets. The old prefix was 1160 tokens/sample
        #    (768 image = 3x256 with two all-zero dummy streams, 200 prompt, 64 reasoning, 64 subtask,
        #    64 FAST) against a measured ~93 actually used. Now 256 + 72 + 56 + 40 + 48 = 472, a 2.5x
        #    cut in prefix length. The dummy streams were already masked out of attention, so
        #    dropping them is a no-op for the math -- identical loss/actions up to bfloat16
        #    accumulation order and identical CoT tokens, see
        #    pi0_cot_test.test_dropping_masked_dummy_cameras_is_a_no_op -- and it saves two full
        #    SigLIP forwards per sample.
        #
        #    Caps are set from untruncated segment lengths measured over the actual weighted mixture
        #    (mean / p99 / max): prompt 48/61/64, reasoning 14/36/48, subtask 19/28/34. Each cap is
        #    max + headroom rounded to a multiple of 8. Do not trim further without re-measuring --
        #    ``_pad_or_truncate`` truncates from the front, so an overflowing segment silently loses
        #    its ``<end_of_*>`` delimiter.
        #
        #    max_fast_len is measured on NORMALIZED actions (16.3/28/36), not raw ones (12.3/21/26):
        #    normalizing widens the action range, which lengthens the FAST encoding. At 32 this
        #    truncated 0.3% of samples. Re-measure if the norm stats or action format change.
        #
        #    NOTE: max_subtask_len / max_reasoning_len / max_fast_len exist on BOTH the model config
        #    (drives inputs_spec) and the data config (drives the tokenizer). They must agree.
        #
        # 4. Eval now also reports the deployed path (``eval/gen_ade_*``, ``eval/gen_fde_*``,
        #    ``eval/gen_action_mse``), sampling actions from the model's own generated CoT rather than
        #    the ground-truth subtask. That is a code change in steervla_visualization.py and applies
        #    to every CoT config; compare runs on the gen_* metrics, not the oracle ``eval/ade_*``.
        #
        # On measuring per-dim action balance: filter to action-supervised rows first. The HL
        # dataset carries dummy (zero) actions and an all-False action_loss_mask, so its rows
        # contribute nothing to the flow loss -- but they are ~31% of the mixture, and on dim 2 a
        # dummy zero normalizes to -3.06, which makes an unfiltered measurement look like dim 2 has
        # a huge tail. Variance share over real-action (simlingo_dataset_*) rows only:
        #
        #        dim              0      1      2      3     spread
        #        no norm       26.3%   7.2%   7.6%  58.9%      2.9x
        #        normalized    41.9%  13.0%  25.3%  19.9%      1.8x   <- this config
        #
        # Plain quantile normalization is the best balance available here, so no dimension is
        # exempted. ``RLDSSteerVLADataConfig.unnormalized_action_dims`` exists if a future action
        # layout needs it, but leaving it empty is correct for this one.
        #
        # ``compute_norm_stats.py`` already applies the same filter when building the stats
        # (it reported excluding 156491/499968 rows = 31.3%, matching the 31.4% HL mixture weight).
        #
        TrainConfig(
            name="pi05_steervla_cot_simplified_reasoning_norm",
            model=pi0_config.Pi0CoTConfig(
                action_dim=32,
                action_horizon=10,
                max_token_len=72,
                max_subtask_len=40,
                max_reasoning_len=56,
                max_fast_len=48,
                cot_loss_weight=0.1,
                knowledge_insulation=False,
                use_fast_tokens=True,
                image_keys=("base_0_rgb",),
            ),
            data=RLDSSteerVLACoTDataConfig(
                repo_id="steervla_simlingo_cot_normed",
                rlds_data_dir="/raid/datasets/steervla",
                dataset_format=steervla_rlds_dataset.DatasetFormat.SIMLINGO,
                include_ego_history=False,
                include_xy_action=False,
                speed_in_prompt=True,
                proprio_norm=False,
                action_dim=4,
                output_action_format=steervla_rlds_dataset.OutputActionFormat.DELTA_XY_T_DELTA_XY_SPACE,
                lang_label_type=steervla_rlds_dataset.LangLabelType.ROUTING_COMMAND,
                dataset_name_weight_mappings={
                    "simlingo_dataset_all_img512_1116": 1.0,
                    "simlingo_dataset_acceleration_negative5_img512_1116": 0.4,
                    "simlingo_dataset_acceleration_negative1_img512_1116": 0.2,
                    "simlingo_dataset_acceleration_positive1_img512_1116": 0.1,
                    "simlingo_dataset_acceleration_positive5_img512_1116": 0.2,
                    "simlingo_dataset_lateral_control12_img512_1116": 0.1,
                    "simlingo_dataset_lateral_control_higher5_img512_1116": 0.3,
                    "simlingo_dataset_start_from_stop_img512_1116": 0.2,
                    "simlingo_dataset_vehicle_front_img512_1116": 0.3,
                    "simlingo_dataset_vehicle_side_img512_1116": 0.1,
                    "simlingo_dataset_leading_object_vehicle_img512_1116": 0.05,
                    "simlingo_dataset_leading_object_traffic_stop_img512_1116": 0.2,
                    "simlingo_dataset_leading_object_traffic_light_img512_1116": 0.2,
                    "simlingo_dataset_leading_object_walker_img512_1116": 0.2,
                    "simlingo_dataset_changed_route_img512_1116": 0.2,
                    "simlingo_dataset_parking_lane_img512_1116": 0.3,
                },
                hl_dataset_name_weight_mappings={
                    "simplified_reasoning_dataset": 1.85,
                },
                hl_dataset_format=steervla_rlds_dataset.DatasetFormat.SIMLINGO,
                hl_cot_reasoning_key="gemini_refined_label",
                hl_cot_subtask_key="prompt",
                max_subtask_len=40,
                max_reasoning_len=56,
                max_fast_len=48,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_000,
                peak_lr=2e-5,
                decay_steps=200_000,
                decay_lr=1e-5,
            ),
            num_train_steps=200_000,
            # Held at the baseline's 192 for a clean A/B. The shorter prefix leaves headroom to raise
            # this -- keep it divisible by jax.device_count() (train.py asserts).
            batch_size=192,
            fsdp_devices=3,
            log_interval=1,
            save_interval=2000,
            max_to_keep=10,
            num_workers=0,
            checkpoint_base_dir="gs://cat-logs",
            resume=False,
            skip_norm_stats=False,
        ),
        #
        # Three-way supervision mixture, derived from pi05_steervla_cot_simplified_reasoning_norm.
        # Identical model, action format, normalization and schedule; only the mixture differs.
        #
        #   10%  simplified_reasoning_dataset   CoT only    (dummy actions -> flow loss zeroed)
        #   20%  simlingo scenarios             flow + CoT  (a full update)
        #   70%  simlingo scenarios             flow only   (CoT text is context, not a target)
        #
        # Motivation: the CoT heads fit their targets within ~2k steps (cot_subtask_ce plateaus around
        # 0.5-0.6, cot_reasoning_ce below 0.1), after which most of the CoT gradient is spent
        # re-learning what the model already knows. Shifting the balance toward action-only updates
        # spends that capacity on the flow expert instead, while keeping enough CoT supervision to
        # stop the reasoning/subtask heads drifting.
        #
        # The 20% and 70% buckets draw from the SAME corpus at the same scenario ratios -- a dataset
        # name may appear in both ``dataset_name_weight_mappings`` and
        # ``ll_dataset_name_weight_mappings``, and each occurrence becomes its own weighted source.
        #
        # "Flow only" gates the CoT cross-entropy per sample (``Observation.cot_loss_mask``); it does
        # NOT remove the CoT segments from the prefix. The action expert still attends to a
        # ground-truth subtask on every sample, so this does not by itself narrow the oracle-vs-
        # generated gap that ``eval/gen_ade_*`` measures -- that needs subtask dropout or scheduled
        # sampling, which is a separate change.
        #
        # Check the realised mixture in wandb: ``train/cot_supervised_frac`` should sit near 0.30
        # (10% CoT-only + 20% full) and ``train/action_supervised_frac`` near 0.90 (20% + 70%).
        #
        TrainConfig(
            name="pi05_steervla_cot_simplified_reasoning_norm_ll_heavy",
            model=pi0_config.Pi0CoTConfig(
                action_dim=32,
                action_horizon=10,
                max_token_len=72,
                max_subtask_len=40,
                max_reasoning_len=56,
                max_fast_len=48,
                cot_loss_weight=0.1,
                knowledge_insulation=False,
                use_fast_tokens=True,
                image_keys=("base_0_rgb",),
            ),
            data=RLDSSteerVLACoTDataConfig(
                # Same corpus, action format and state layout as the _norm config, so its norm stats
                # apply unchanged -- the mixture shifts which rows are drawn, not what a row contains.
                repo_id="steervla_simlingo_cot_normed",
                assets=AssetsConfig(
                    assets_dir="./assets/pi05_steervla_cot_simplified_reasoning_norm",
                    asset_id="steervla_simlingo_cot_normed",
                ),
                rlds_data_dir="/raid/datasets/steervla",
                dataset_format=steervla_rlds_dataset.DatasetFormat.SIMLINGO,
                include_ego_history=False,
                include_xy_action=False,
                speed_in_prompt=True,
                proprio_norm=False,
                action_dim=4,
                output_action_format=steervla_rlds_dataset.OutputActionFormat.DELTA_XY_T_DELTA_XY_SPACE,
                lang_label_type=steervla_rlds_dataset.LangLabelType.ROUTING_COMMAND,
                dataset_name_weight_mappings=_mixture_share(_SIMLINGO_SCENARIO_WEIGHTS, 0.20),
                ll_dataset_name_weight_mappings=_mixture_share(_SIMLINGO_SCENARIO_WEIGHTS, 0.70),
                hl_dataset_name_weight_mappings={"simplified_reasoning_dataset": 0.10},
                hl_dataset_format=steervla_rlds_dataset.DatasetFormat.SIMLINGO,
                hl_cot_reasoning_key="gemini_refined_label",
                hl_cot_subtask_key="prompt",
                max_subtask_len=40,
                max_reasoning_len=56,
                max_fast_len=48,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_000,
                peak_lr=2e-5,
                decay_steps=200_000,
                decay_lr=1e-5,
            ),
            num_train_steps=200_000,
            batch_size=384,
            fsdp_devices=4,
            log_interval=1,
            save_interval=2000,
            max_to_keep=10,
            num_workers=0,
            checkpoint_base_dir="gs://cat-logs",
            resume=False,
            skip_norm_stats=False,
        ),
        #
        # The 10/20/70 supervision mixture WITHOUT normalization -- i.e. ..._norm_ll_heavy with
        # skip_norm_stats=True. Everything else (model, mixture, action format, token budgets,
        # schedule) is identical to that config, so the pair isolates normalization as the single
        # variable, and this one is directly comparable to the pre-normalization runs.
        #
        # CAVEAT, deliberate and worth knowing before reading results: skip_norm_stats disables the
        # Normalize transform outright, so it unnormalizes STATE as well as actions. State then reaches
        # ``CoTPaligemmaTokenizer.tokenize_prompt``, which discretizes against fixed [-1, 1] bins, in
        # raw units -- every speed >= 1 m/s saturates to token 255 and every course <= -1 deg gives
        # token -1, leaving the state channel with roughly two bits. Speed still reaches the model
        # through the prompt text ("The current speed is X m/s."), but heading does not.
        #
        # To vary the actions alone and keep the state channel intact, set skip_norm_stats=False and
        # ``unnormalized_action_dims=(0, 1, 2, 3)`` instead: that exempts every action dim from
        # Normalize via identity stats while state stays quantile-normalized. That is the cleaner
        # ablation of "do normalized actions help"; this config is the cleaner comparison against the
        # older runs.
        #
        # No norm stats needed (nothing is normalized), so no compute_norm_stats run before training.
        #
        TrainConfig(
            name="pi05_steervla_cot_simplified_reasoning_ll_heavy",
            model=pi0_config.Pi0CoTConfig(
                action_dim=32,
                action_horizon=10,
                max_token_len=72,
                max_subtask_len=40,
                max_reasoning_len=56,
                # Held at 48 to match ..._norm_ll_heavy exactly, so the two configs have identical
                # prefix lengths and per-step compute. Raw actions only need ~32 (their FAST encoding
                # is shorter than the normalized one: max 26 vs 36), but matching matters more here.
                max_fast_len=48,
                cot_loss_weight=0.1,
                knowledge_insulation=False,
                use_fast_tokens=True,
                image_keys=("base_0_rgb",),
            ),
            data=RLDSSteerVLACoTDataConfig(
                repo_id="steervla_simlingo_cot",
                rlds_data_dir="/raid/datasets/steervla",
                dataset_format=steervla_rlds_dataset.DatasetFormat.SIMLINGO,
                include_ego_history=False,
                include_xy_action=False,
                speed_in_prompt=True,
                proprio_norm=False,
                action_dim=4,
                output_action_format=steervla_rlds_dataset.OutputActionFormat.DELTA_XY_T_DELTA_XY_SPACE,
                lang_label_type=steervla_rlds_dataset.LangLabelType.ROUTING_COMMAND,
                dataset_name_weight_mappings=_mixture_share(_SIMLINGO_SCENARIO_WEIGHTS, 0.20),
                ll_dataset_name_weight_mappings=_mixture_share(_SIMLINGO_SCENARIO_WEIGHTS, 0.70),
                hl_dataset_name_weight_mappings={"simplified_reasoning_dataset": 0.10},
                hl_dataset_format=steervla_rlds_dataset.DatasetFormat.SIMLINGO,
                hl_cot_reasoning_key="gemini_refined_label",
                hl_cot_subtask_key="prompt",
                max_subtask_len=40,
                max_reasoning_len=56,
                max_fast_len=48,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_000,
                peak_lr=2e-5,
                decay_steps=200_000,
                decay_lr=1e-5,
            ),
            num_train_steps=200_000,
            batch_size=1152,
            fsdp_devices=6,
            log_interval=1,
            save_interval=2000,
            max_to_keep=10,
            num_workers=0,
            checkpoint_base_dir="gs://cat-logs",
            resume=False,
            skip_norm_stats=True,
        ),
        TrainConfig(
            name="pi05_steervla_cot_simplified_reasoning_no_attention",
            model=pi0_config.Pi0CoTConfig(
                action_dim=32,
                action_horizon=10,
                max_token_len=200,
                max_subtask_len=64,
                max_reasoning_len=64,
                cot_loss_weight=0.1,
                knowledge_insulation=False,
                use_fast_tokens=True,
                action_attend_subtask=False,
            ),
            data=RLDSSteerVLACoTDataConfig(
                repo_id="steervla_simlingo_cot",
                rlds_data_dir="/raid/datasets/steervla",
                dataset_format=steervla_rlds_dataset.DatasetFormat.SIMLINGO,
                include_ego_history=False,
                include_xy_action=False,
                speed_in_prompt=True,
                proprio_norm=False,
                action_dim=4,
                output_action_format=steervla_rlds_dataset.OutputActionFormat.DELTA_XY_T_DELTA_XY_SPACE,
                lang_label_type=steervla_rlds_dataset.LangLabelType.ROUTING_COMMAND,
                dataset_name_weight_mappings={
                    "simlingo_dataset_all_img512_1116": 1.0,
                    "simlingo_dataset_acceleration_negative5_img512_1116": 0.4,
                    "simlingo_dataset_acceleration_negative1_img512_1116": 0.2,
                    "simlingo_dataset_acceleration_positive1_img512_1116": 0.1,
                    "simlingo_dataset_acceleration_positive5_img512_1116": 0.2,
                    "simlingo_dataset_lateral_control12_img512_1116": 0.1,
                    "simlingo_dataset_lateral_control_higher5_img512_1116": 0.3,
                    "simlingo_dataset_start_from_stop_img512_1116": 0.2,
                    "simlingo_dataset_vehicle_front_img512_1116": 0.3,
                    "simlingo_dataset_vehicle_side_img512_1116": 0.1,
                    "simlingo_dataset_leading_object_vehicle_img512_1116": 0.05,
                    "simlingo_dataset_leading_object_traffic_stop_img512_1116": 0.2,
                    "simlingo_dataset_leading_object_traffic_light_img512_1116": 0.2,
                    "simlingo_dataset_leading_object_walker_img512_1116": 0.2,
                    "simlingo_dataset_changed_route_img512_1116": 0.2,
                    "simlingo_dataset_parking_lane_img512_1116": 0.3,
                },
                hl_dataset_name_weight_mappings={
                    "simplified_reasoning_dataset": 1.85,
                },
                hl_dataset_format=steervla_rlds_dataset.DatasetFormat.SIMLINGO,
                hl_cot_reasoning_key="gemini_refined_label",
                hl_cot_subtask_key="prompt",
                max_subtask_len=64,
                max_reasoning_len=64,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_000,
                peak_lr=2e-5,
                decay_steps=200_000,
                decay_lr=1e-5,
            ),
            num_train_steps=200_000,
            batch_size=512,
            fsdp_devices=8,
            log_interval=1,
            save_interval=1000,
            keep_period=1000,
            max_to_keep=10,
            num_workers=0,
            checkpoint_base_dir="gs://cat-logs",
            resume=True,
            skip_norm_stats=True,
        ),
        TrainConfig(
            name="pi05_steervla_finetune",
            model=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=6,
            ),
            data=LeRobotSteerVLADataConfig(
                repo_id="your_hf_username/my_steervla_dataset",
                base_config=DataConfig(prompt_from_task=True),
                speed_in_prompt=True,
                include_ego_history=True,
                proprio_norm=True,
                action_dim=2,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            num_train_steps=20_000,
            batch_size=32,
        ),
        #
        # Inference SteerVLA configs.
        #
        TrainConfig(
            name="pi05_steervla_inference",
            model=pi0_config.Pi0Config(pi05=True, action_horizon=6),
            data=SimpleDataConfig(
                assets=AssetsConfig(asset_id="steervla"),
                data_transforms=lambda model: _transforms.Group(
                    inputs=[steervla_policy.SteerVLAInputs(model_type=ModelType.PI05)],
                    outputs=[steervla_policy.SteerVLAOutputs(action_dim=2)],
                ),
                base_config=DataConfig(
                    prompt_from_task=True,
                ),
            ),
        ),
    ]
    configs.append(_gemma3_variant(configs, "pi05_steervla_cot_simplified_reasoning_norm", AssetsConfig))
    return configs


def _gemma3_variant(configs, parent_name: str, assets_config_cls):
    """Same data, token budgets and schedule as ``parent_name``, on a Gemma 3 4B backbone.

    The action expert matches Gemma 3's attention geometry and starts from scratch (no pi05_base
    pretraining), so expect a slower start than the PaliGemma parent. Norm stats come from the parent:
    they describe the data (state/actions), not the backbone.
    """
    import dataclasses

    parent = next(c for c in configs if c.name == parent_name)
    name = parent_name.replace("pi05_", "gemma3_", 1)
    return dataclasses.replace(
        parent,
        name=name,
        exp_name=name,
        model=dataclasses.replace(
            parent.model,
            backbone="gemma3",
            paligemma_variant="gemma3_4b",
            action_expert_variant="gemma3_expert_1024",
            # Native Gemma 3 resolution. The SimLingo frames are 512px, so 448 is a cheaper option that
            # avoids upsampling (position embeddings are interpolated at load time).
            image_resolution=896,
        ),
        data=dataclasses.replace(
            parent.data,
            assets=assets_config_cls(assets_dir=f"./assets/{parent_name}", asset_id=parent.data.repo_id),
        ),
        weight_loader=weight_loaders.Gemma3WeightLoader("google/gemma-3-4b-pt"),
    )
