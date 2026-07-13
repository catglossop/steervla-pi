import dataclasses
import functools
import logging
import platform
from typing import Any
import datetime

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.jax_distributed as jax_distributed
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders

from openpi.visualizing.steervla_visualization import run_cot_visualization, run_visualization_evaluation


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)

    # Suppress noisy orbax CheckpointManager logs about missing per-step
    # `metrics/metrics` files. Those errors are caught inside orbax (see
    # checkpoint_manager.py `metrics()`), so they're harmless but spam the logs
    # when resuming from a checkpoint that was saved without metrics tracking.
    class _OrbaxMetricsFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            if record.filename == "checkpoint_manager.py" and record.lineno in (1654, 1655):
                return False
            msg = record.getMessage()
            if "Missing metrics for step" in msg:
                return False
            if "/metrics/metrics not found" in msg:
                return False
            return True

    logger.addFilter(_OrbaxMetricsFilter())


def init_wandb(
    config: _config.TrainConfig,
    *,
    resuming: bool,
    run_dir: epath.Path,
    start_step: int = 0,
    log_code: bool = False,
    enabled: bool = True,
):
    if not enabled:
        wandb.init(mode="disabled")
        return

    if jax.process_index() != 0:
        wandb.init(mode="disabled")
        return

    # Always start a fresh wandb run (even when resuming training). The training
    # loop calls wandb.log(..., step=step) with the absolute step from the
    # restored TrainState, so the new run picks up at the right step without
    # needing wandb's own resume machinery (which was brittle / errored out).
    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    run_name = run_dir.name
    if resuming:
        run_name = f"{run_name}_resumed_from_step_{start_step}"

    wandb.init(
        name=run_name,
        config=dataclasses.asdict(config),
        project=config.project_name,
    )
    (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        if hasattr(model, "compute_loss_with_aux"):
            chunked_loss, aux_metrics = model.compute_loss_with_aux(rng, observation, actions, train=True)
        else:
            chunked_loss = model.compute_loss(rng, observation, actions, train=True)
            aux_metrics = {}

        loss = jnp.mean(chunked_loss)
        reduced_aux_metrics = {f"train/{k}": jnp.mean(v) for k, v in aux_metrics.items()}
        return loss, reduced_aux_metrics

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, aux_metrics), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, observation, actions
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    info.update(aux_metrics)
    return new_state, info


def _assert_shape_dtype(name: str, value: Any, expected_shape: tuple[int, ...], expected_dtype: Any) -> None:
    arr = np.asarray(value)
    if tuple(arr.shape) != tuple(expected_shape):
        raise ValueError(f"{name} shape mismatch: got {tuple(arr.shape)}, expected {tuple(expected_shape)}")
    if arr.dtype != np.dtype(expected_dtype):
        raise ValueError(f"{name} dtype mismatch: got {arr.dtype}, expected {np.dtype(expected_dtype)}")


def _validate_batch_against_model_spec(
    config: _config.TrainConfig,
    batch: tuple[_model.Observation, _model.Actions],
    *,
    batch_idx: int,
) -> None:
    """Validate incoming batch leaves against model input spec.

    This catches data-loader shape/dtype issues before the first jitted train step,
    which otherwise may fail with opaque XLA buffer mismatch errors.
    """
    observation, actions = batch
    expected_obs, expected_actions = config.model.inputs_spec(batch_size=config.batch_size)

    # Actions.
    _assert_shape_dtype("actions", actions, expected_actions.shape, expected_actions.dtype)

    # Core observation leaves.
    _assert_shape_dtype("observation.state", observation.state, expected_obs.state.shape, expected_obs.state.dtype)

    # Images + masks.
    for key, exp in expected_obs.images.items():
        if key not in observation.images:
            raise ValueError(f"observation.images missing expected key: {key}")
        _assert_shape_dtype(f"observation.images[{key!r}]", observation.images[key], exp.shape, exp.dtype)
    for key, exp in expected_obs.image_masks.items():
        if key not in observation.image_masks:
            raise ValueError(f"observation.image_masks missing expected key: {key}")
        _assert_shape_dtype(f"observation.image_masks[{key!r}]", observation.image_masks[key], exp.shape, exp.dtype)

    # Optional fields (present in CoT/FAST variants).
    optional_fields = (
        "tokenized_prompt",
        "tokenized_prompt_mask",
        "token_ar_mask",
        "token_loss_mask",
        "tokenized_subtask",
        "tokenized_subtask_mask",
        "tokenized_reasoning",
        "tokenized_reasoning_mask",
        "action_loss_mask",
    )
    for field_name in optional_fields:
        exp = getattr(expected_obs, field_name)
        got = getattr(observation, field_name)
        if exp is None:
            continue
        if got is None:
            raise ValueError(f"observation.{field_name} is missing but required by model input spec")
        _assert_shape_dtype(f"observation.{field_name}", got, exp.shape, exp.dtype)

    logging.info("Batch validation passed for batch index %d.", batch_idx)


def main(config: _config.TrainConfig):
    jax_distributed.initialize_if_needed()
    init_logging()
    logging.info(
        "Running on: %s | jax.process_index=%s jax.process_count=%s jax.device_count=%s",
        platform.node(),
        jax.process_index(),
        jax.process_count(),
        jax.device_count(),
    )

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )
    if jax.process_count() > 1 and config.batch_size % jax.process_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by jax.process_count() ({jax.process_count()}) "
            "for multi-host data sharding."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    if config.resume_dir is not None:
        run_dir = epath.Path(config.resume_dir)
        resume_flag = True
    else:
        run_dir = config.checkpoint_dir / (
            config.exp_name + "_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        resume_flag = config.resume

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        run_dir,
        keep_period=config.keep_period,
        max_to_keep=config.max_to_keep,
        overwrite=config.overwrite,
        resume=resume_flag,
    )
    if config.resume_dir is not None and not resuming:
        raise FileNotFoundError(
            f"resume_dir={config.resume_dir} was provided but the directory does not exist or is empty."
        )

    resume_start_step = int(checkpoint_manager.latest_step()) if resuming else 0
    init_wandb(
        config,
        resuming=resuming,
        run_dir=run_dir,
        start_step=resume_start_step,
        enabled=config.wandb_enabled,
    )

    # Train data loader
    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
        skip_norm_stats=config.skip_norm_stats,
    )
    data_iter = iter(data_loader)
    
    # Eval data loader
    eval_batch_size = config.batch_size // jax.device_count()
    eval_config = dataclasses.replace(config, batch_size=eval_batch_size)
    eval_data_loader = _data_loader.create_data_loader(
        eval_config,
        sharding=data_sharding,
        shuffle=False,
        split="val",
        skip_norm_stats=config.skip_norm_stats,
    )
    eval_data_iter = iter(eval_data_loader)
    
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")
    _validate_batch_against_model_spec(config, batch, batch_idx=0)

    # Log images from first batch to sanity check.
    if jax.process_index() == 0 and not resuming:
        images_to_log = [
            wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
            for i in range(min(5, len(next(iter(batch[0].images.values())))))
        ]
        wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(
            checkpoint_manager, train_state, data_loader, step=config.resume_step
        )

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    # Resolve eval visualization settings from the data config.
    data_config = data_loader.data_config()
    eval_action_dim = data_config.steervla_action_dim
    eval_output_format = data_config.steervla_output_action_format.value if data_config.steervla_rlds else None
    eval_dataset_names: list[str] | None = None
    if data_config.steervla_rlds:
        eval_dataset_names = [
            *(d.name for d in data_config.steervla_datasets),
            *(d.name for d in data_config.steervla_hl_datasets),
        ]

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
        disable=jax.process_index() != 0,
    )

    eval_rng = jax.random.key(config.seed + 1)
    infos = []
    validate_num_batches = 4
    for step in pbar:
        if step < start_step + validate_num_batches:
            _validate_batch_against_model_spec(config, batch, batch_idx=step - start_step)
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            if jax.process_index() == 0:
                info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
                pbar.write(f"Step {step}: {info_str}")
                wandb.log(reduced_info, step=step)
            infos = []
            
        batch = next(data_iter)

        if config.eval_interval > 0 and step > 0 and step % config.eval_interval == 0:
            # change eval batch size 
            eval_batch = next(eval_data_iter)
            if jax.process_index() == 0:
                pbar.write(f"Running eval visualization at step {step}...")
            eval_rng, vis_rng = jax.random.split(eval_rng)
            vis_rng, cot_rng = jax.random.split(vis_rng)
            with sharding.set_mesh(mesh):
                run_visualization_evaluation(
                    state=train_state,
                    rng=vis_rng,
                    batch=eval_batch,
                    step=step,
                    action_dim=eval_action_dim,
                    output_action_format=eval_output_format,
                    dataset_names=eval_dataset_names,
                )
                run_cot_visualization(
                    state=train_state,
                    rng=cot_rng,
                    batch=eval_batch,
                    step=step,
                    use_fast_tokens=getattr(config.model, "use_fast_tokens", False),
                    action_horizon=config.model.action_horizon,
                    action_dim=eval_action_dim,
                    model_action_dim=config.model.action_dim,
                    output_action_format=eval_output_format,
                )

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
