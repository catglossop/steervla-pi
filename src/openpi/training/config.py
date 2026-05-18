"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.policies.steervla_policy as steervla_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.steervla_rlds_dataset as steervla_rlds_dataset
import openpi.training.misc.polaris_config as polaris_config
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID and SteerVLA).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()

    # SteerVLA RLDS-specific fields.
    steervla_rlds: bool = False
    steervla_datasets: Sequence[steervla_rlds_dataset.SteerVLARLDSDataset] = ()
    steervla_dataset_format: steervla_rlds_dataset.DatasetFormat = steervla_rlds_dataset.DatasetFormat.NUSCENES
    # Optional high-level (reasoning/subtask-only) datasets for CoT training.
    steervla_hl_datasets: Sequence[steervla_rlds_dataset.SteerVLARLDSDataset] = ()
    steervla_hl_dataset_format: steervla_rlds_dataset.DatasetFormat = steervla_rlds_dataset.DatasetFormat.NUSCENES
    steervla_cot_reasoning_key: str = "commentary"
    steervla_cot_subtask_key: str = "gemini_refined_label"
    steervla_hl_cot_reasoning_key: str = "gemini_refined_label"
    steervla_hl_cot_subtask_key: str = "prompt"
    steervla_include_ego_history: bool = True
    steervla_include_xy_action: bool = False
    steervla_speed_in_prompt: bool = True
    steervla_proprio_norm: bool = True
    steervla_output_action_format: steervla_rlds_dataset.OutputActionFormat = steervla_rlds_dataset.OutputActionFormat.DELTA_XY_T_DELTA_XY_SPACE
    steervla_lang_label_type: steervla_rlds_dataset.LangLabelType = steervla_rlds_dataset.LangLabelType.ROUTING_COMMAND
    steervla_routing_command_in_prompt: bool = False
    steervla_add_suffix_to_prompt: bool = False
    steervla_action_dim: int = 4
    steervla_enable_cot: bool = False


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.

    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = (
        droid_rlds_dataset.RLDSDataset(
            name="droid",
            version="1.0.1",
            weight=1.0,
            filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
        ),
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            datasets=self.datasets,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class RLDSSteerVLADataConfig(DataConfigFactory):
    """
    Config for training on driving data (nuScenes or SimLingo), using RLDS data format.
    Adapted from the bigvision-palivla-drive pipeline.

    Datasets can be specified either as:
    - A sequence of SteerVLARLDSDataset objects via `datasets`
    - A dict mapping dataset names to weights via `dataset_name_weight_mappings`
      (more ergonomic for multi-dataset configs like SimLingo)

    Weights do NOT need to sum to 1.0 -- they are normalized internally.
    """

    rlds_data_dir: str | None = None
    dataset_format: tyro.conf.Suppress[steervla_rlds_dataset.DatasetFormat] = steervla_rlds_dataset.DatasetFormat.NUSCENES
    include_ego_history: bool = True
    include_xy_action: bool = False
    speed_in_prompt: bool = True
    proprio_norm: bool = True
    action_dim: int = 2

    # Option 1: explicit dataset list.
    datasets: Sequence[steervla_rlds_dataset.SteerVLARLDSDataset] = ()

    # Option 2: dict mapping dataset_name -> weight (version will use dataset_version).
    dataset_name_weight_mappings: tyro.conf.Suppress[dict[str, float] | None] = None
    dataset_version: tyro.conf.Suppress[str | None] = None

    # SimLingo-specific options.
    output_action_format: tyro.conf.Suppress[steervla_rlds_dataset.OutputActionFormat] = steervla_rlds_dataset.OutputActionFormat.DELTA_XY_T_DELTA_XY_SPACE
    lang_label_type: tyro.conf.Suppress[steervla_rlds_dataset.LangLabelType] = steervla_rlds_dataset.LangLabelType.ROUTING_COMMAND
    routing_command_in_prompt: bool = False
    add_suffix_to_prompt: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Resolve datasets from either datasets list or name->weight mapping.
        resolved_datasets = list(self.datasets)
        if self.dataset_name_weight_mappings is not None:
            for name, weight in self.dataset_name_weight_mappings.items():
                resolved_datasets.append(
                    steervla_rlds_dataset.SteerVLARLDSDataset(
                        name=name, weight=weight, version=self.dataset_version,
                    )
                )

        assert len(resolved_datasets) > 0, "Must specify at least one dataset via `datasets` or `dataset_name_weight_mappings`."

        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation/image",
                        "observation/state": "observation/state",
                        "observation/current_speed": "observation/current_speed",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # For RLDS data, the speed_in_prompt injection already happens in the RLDS loader,
        # so we disable it in the policy transform to avoid double-injection.
        data_transforms = _transforms.Group(
            inputs=[steervla_policy.SteerVLAInputs(
                model_type=model_config.model_type,
                speed_in_prompt=False,
                include_ego_history=False,
                proprio_norm=False,
            )],
            outputs=[steervla_policy.SteerVLAOutputs(action_dim=self.action_dim)],
        )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds_data_dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            steervla_rlds=True,
            rlds_data_dir=self.rlds_data_dir,
            steervla_datasets=tuple(resolved_datasets),
            steervla_dataset_format=self.dataset_format,
            steervla_include_ego_history=self.include_ego_history,
            steervla_include_xy_action=self.include_xy_action,
            steervla_speed_in_prompt=self.speed_in_prompt,
            steervla_proprio_norm=self.proprio_norm,
            steervla_output_action_format=self.output_action_format,
            steervla_lang_label_type=self.lang_label_type,
            steervla_routing_command_in_prompt=self.routing_command_in_prompt,
            steervla_add_suffix_to_prompt=self.add_suffix_to_prompt,
            steervla_action_dim=self.action_dim,
        )


@dataclasses.dataclass(frozen=True)
class RLDSSteerVLACoTDataConfig(RLDSSteerVLADataConfig):
    """Extension of RLDSSteerVLADataConfig that enables chain-of-thought generation.

    Uses the routing_command as the high-level prompt, gemini_refined_label as
    the subtask, and commentary as the reasoning.
    """

    # CoT-specific token lengths.
    max_subtask_len: int = 48
    max_reasoning_len: int = 96
    # Optional high-level (reasoning/subtask-only) datasets.
    hl_dataset_name_weight_mappings: tyro.conf.Suppress[dict[str, float] | None] = None
    hl_dataset_version: tyro.conf.Suppress[str | None] = None
    hl_dataset_format: tyro.conf.Suppress[steervla_rlds_dataset.DatasetFormat | None] = None
    # Source keys for CoT targets.
    cot_reasoning_key: str = "commentary"
    cot_subtask_key: str = "gemini_refined_label"
    hl_cot_reasoning_key: str | None = "gemini_refined_label"
    hl_cot_subtask_key: str | None = "prompt"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        resolved_datasets = list(self.datasets)
        if self.dataset_name_weight_mappings is not None:
            for name, weight in self.dataset_name_weight_mappings.items():
                resolved_datasets.append(
                    steervla_rlds_dataset.SteerVLARLDSDataset(
                        name=name, weight=weight, version=self.dataset_version,
                    )
                )

        assert len(resolved_datasets) > 0

        resolved_hl_datasets: list[steervla_rlds_dataset.SteerVLARLDSDataset] = []
        if self.hl_dataset_name_weight_mappings is not None:
            for name, weight in self.hl_dataset_name_weight_mappings.items():
                resolved_hl_datasets.append(
                    steervla_rlds_dataset.SteerVLARLDSDataset(
                        name=name, 
                        weight=weight, 
                        version=self.hl_dataset_version,
                    )
                )

        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation/image",
                        "observation/state": "observation/state",
                        "observation/current_speed": "observation/current_speed",
                        "actions": "actions",
                        "action_loss_mask": "action_loss_mask",
                        "prompt": "prompt",
                        "subtask": "subtask",
                        "reasoning": "reasoning",
                    }
                )
            ]
        )


        data_transforms = _transforms.Group(
            inputs=[steervla_policy.SteerVLAInputs(
                model_type=model_config.model_type,
                speed_in_prompt=True,
                include_ego_history=False,
                proprio_norm=False,
            )],
            outputs=[steervla_policy.SteerVLAOutputs(action_dim=self.action_dim)],
        )

        from openpi.models.tokenizer import CoTPaligemmaTokenizer
        cot_tokenizer = CoTPaligemmaTokenizer(
            max_prompt_len=model_config.max_token_len,
            max_subtask_len=self.max_subtask_len,
            max_reasoning_len=self.max_reasoning_len,
        )
        model_transforms = _transforms.Group(
            inputs=[
                _transforms.ResizeImages(224, 224),
                _transforms.TokenizeCoTPrompt(cot_tokenizer),
                _transforms.PadStatesAndActions(model_config.action_dim),
            ],
        )

        assert self.rlds_data_dir is not None

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            steervla_rlds=True,
            rlds_data_dir=self.rlds_data_dir,
            steervla_datasets=tuple(resolved_datasets),
            steervla_dataset_format=self.dataset_format,
            steervla_hl_datasets=tuple(resolved_hl_datasets),
            steervla_hl_dataset_format=self.hl_dataset_format or self.dataset_format,
            steervla_cot_reasoning_key=self.cot_reasoning_key,
            steervla_cot_subtask_key=self.cot_subtask_key,
            steervla_hl_cot_reasoning_key=self.hl_cot_reasoning_key or "gemini_refined_label",
            steervla_hl_cot_subtask_key=self.hl_cot_subtask_key or "prompt",
            steervla_include_ego_history=self.include_ego_history,
            steervla_include_xy_action=self.include_xy_action,
            steervla_speed_in_prompt=False,
            steervla_proprio_norm=self.proprio_norm,
            steervla_output_action_format=self.output_action_format,
            steervla_lang_label_type=self.lang_label_type,
            steervla_routing_command_in_prompt=False,
            steervla_add_suffix_to_prompt=False,
            steervla_action_dim=self.action_dim,
            steervla_enable_cot=True,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotSteerVLADataConfig(DataConfigFactory):
    """
    Config for training on a smaller nuScenes-style driving dataset in LeRobot format.
    """

    speed_in_prompt: bool = True
    include_ego_history: bool = True
    proprio_norm: bool = True
    action_dim: int = 2

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[steervla_policy.SteerVLAInputs(
                model_type=model_config.model_type,
                speed_in_prompt=self.speed_in_prompt,
                include_ego_history=self.include_ego_history,
                proprio_norm=self.proprio_norm,
            )],
            outputs=[steervla_policy.SteerVLAOutputs(action_dim=self.action_dim)],
        )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to run eval visualization (0 = disabled).
    eval_interval: int = 0
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False
    # If set, resume training from this exact run directory (the timestamped
    # directory that contains numbered step subdirs). When provided, this
    # overrides the fresh-timestamp run directory that train.py would otherwise
    # create, and implies resume=True. May be a local path or a `gs://...` URI.
    resume_dir: str | None = None

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> epath.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return epath.Path(self.checkpoint_base_dir) / self.name / self.exp_name

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")
        if self.resume_dir and self.overwrite:
            raise ValueError("Cannot resume_dir and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    #
    # Inference Aloha configs.
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # Inference DROID configs.
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # Fine-tuning Libero configs.
    #
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.
    # They are used to define key elements like the dataset you are training on, the base checkpoint you
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on
    # the comments below.
    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_libero",
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=pi0_config.Pi0Config(),
        # Here you define the dataset you are training on. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # Here is an example of loading a pi0-FAST model for full finetuning.
        # Modify action_dim and action_horizon to match your dataset (action horizon is equal to
        # the desired action chunk length).
        # The max_token_len is the maximum number of (non-image) tokens the model can handle.
        # This includes the tokenized prompt, proprioceptive state, and (FAST-tokenized) action tokens.
        # Choosing this value too small may chop off tokens at the end of your sequence (the code will throw
        # a warning), while choosing it too large will waste memory (since we pad each batch element to the
        # max_token_len). A good rule of thumb is to use approx 180 for single-arm robots, and approx 250 for
        # two-arm robots. Generally, err on the lower side here first, and potentially increase the value if
        # you see many warnings being thrown during training.
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # Note that we load the pi0-FAST base model checkpoint here.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),
    #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instructions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi05_aloha_pen_uncap",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
    ),
    #
    # Fine-tuning DROID configs.
    #
    TrainConfig(
        # This config is for fine-tuning pi0-FAST-base on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi0_fast_full_droid_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="<path_to_droid_rlds_dataset>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 100k steps should be sufficient, takes ~2 days on 8x H100s
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05 on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi05_full_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="/mnt/pi-data/kevin",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets/",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=10_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05-DROID on a custom (smaller) DROID dataset.
        # Here, we use LeRobot data format (like for all other fine-tuning examples)
        # To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
        name="pi05_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotDROIDDataConfig(
            # Replace with your custom DROID LeRobot dataset repo id.
            repo_id="your_hf_username/my_droid_dataset",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # Important: reuse the original DROID norm stats during fine-tuning!
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=32,
    ),
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
        eval_interval=100,
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
        eval_interval=500,
        save_interval=5000,
        num_workers=0,
        checkpoint_base_dir="gs://cat-logs",
    ),
    TrainConfig(
        name="pi05_steervla_cot_ki",
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
        num_train_steps=200_000,
        batch_size=24,
        fsdp_devices=4,
        log_interval=1,
        eval_interval=100,
        save_interval=5000,
        num_workers=0,
        checkpoint_base_dir="gs://cat-logs",
    ),
    TrainConfig(
        name="pi05_steervla_cot_ki_simplified_reasoning",
        model=pi0_config.Pi0CoTConfig(
            action_dim=32,
            action_horizon=10,
            max_token_len=112,
            max_subtask_len=64,
            max_reasoning_len=64,
            cot_loss_weight=1.0,
            knowledge_insulation=True,
        ),
        data=RLDSSteerVLACoTDataConfig(
            repo_id="steervla_simlingo_cot",
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
            peak_lr=1e-4,
            decay_steps=200_000,
            decay_lr=1e-5,
        ),
        num_train_steps=200_000,
        batch_size=256,
        fsdp_devices=4,
        log_interval=1,
        eval_interval=100,
        save_interval=2000,
        num_workers=0,
        checkpoint_base_dir="gs://cat-logs",
        resume=True,
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
    
    # ALOHA Sim configs. This config is used to demonstrate how to train on a simple simulated environment.
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # Debugging configs.
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
    # RoboArena & PolaRiS configs.
    *roboarena_config.get_roboarena_configs(),
    *polaris_config.get_polaris_configs(),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
