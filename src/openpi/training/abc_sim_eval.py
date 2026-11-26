"""Periodic evaluations for ABC sim policies (see ``periodic_eval``).

* :class:`AbcActionMseEval` -- in-process, on held-out episodes: action-chunk MSE in raw joint space at several
  horizons (default 1 s = 30 steps and 0.5 s = 15 steps); per task, mean over tasks, and the mean of the k worst
  tasks. Computed with the task prompt and with the prompt dropped (unconditional).
* :class:`AbcSimEval` -- asynchronous ABC simulator rollouts. At each eval step it launches
  ``scripts/abc_sim_eval_worker.py`` on a dedicated GPU, which serves that step's checkpoint over websocket and
  drives the ABC env (``examples/abc_sim/sim_client.py``, run in ABC's own Python env) with the OpenGL/EGL
  renderer. Results are picked up on later eval calls and logged with the step of the checkpoint they belong to.
"""

import dataclasses
import json
import logging
import os
import pathlib
import pickle
import subprocess
import sys
from typing import Any

import flax.nnx as nnx
import jax
import numpy as np

from openpi.training import abc_sim_dataset as abc
from openpi.training import sharding
from openpi.training.periodic_eval import EvalContext

logger = logging.getLogger("openpi")

DEFAULT_SIM_EVAL_TASKS = (
    "spell_cat",
    "conveyor_pick",
    "pouring_beads",
    "turn_the_mug_right_side_up",
    "put_relative",
    "hang_the_mug_on_the_mug_rack",
)

# Module-level state: hooks live on a frozen TrainConfig, so caches cannot be attributes.
_MSE_CACHE: dict[int, dict[str, Any]] = {}
_SIM_WORKERS: dict[int, subprocess.Popen] = {}
_SIM_LOGGED: set[int] = set()


def _params_for_eval(state):
    return state.ema_params if state.ema_params is not None else state.params


def _unnormalize(x: np.ndarray, stats, *, quantile: bool) -> np.ndarray:
    if stats is None:
        return x
    d = x.shape[-1]
    if quantile:
        q01, q99 = np.asarray(stats.q01)[..., :d], np.asarray(stats.q99)[..., :d]
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
    mean, std = np.asarray(stats.mean)[..., :d], np.asarray(stats.std)[..., :d]
    return x * (std + 1e-6) + mean


def chunk_mse_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    task_ids: np.ndarray,
    *,
    horizons: dict[str, int],
    top_k: int,
    prefix: str,
) -> dict[str, float]:
    """MSE over each horizon's first ``n`` steps and all action dims; per task, task-mean, worst-k task-mean."""
    metrics = {}
    for name, n in horizons.items():
        per_sample = np.mean((pred[:, :n] - target[:, :n]) ** 2, axis=(1, 2))
        per_task = {int(t): float(per_sample[task_ids == t].mean()) for t in np.unique(task_ids)}
        values = np.array(sorted(per_task.values(), reverse=True))
        metrics[f"{prefix}/{name}/mean"] = float(values.mean())
        metrics[f"{prefix}/{name}/worst{top_k}"] = float(values[:top_k].mean())
        for t, v in per_task.items():
            metrics[f"{prefix}/{name}/task/{abc.TASKS[t]}"] = v
    return metrics


@dataclasses.dataclass(frozen=True)
class AbcActionMseEval:
    interval: int = 2_000
    # Fixed held-out sample: num_batches x batch_size frames from the val split (same samples every eval).
    num_batches: int = 8
    batch_size: int = 64
    horizons: tuple[tuple[str, int], ...] = (("1.0s", 30), ("0.5s", 15))
    top_k: int = 5
    num_flow_steps: int = 10
    # Also evaluate with the task prompt dropped (requires training with prompt dropout to be meaningful).
    unconditional: bool = True

    def _cache(self, ctx: EvalContext) -> dict[str, Any]:
        key = id(ctx.config)
        if key in _MSE_CACHE:
            return _MSE_CACHE[key]
        import openpi.training.data_loader as _data_loader

        config = ctx.config
        if config.batch_size and self.batch_size % jax.device_count():
            raise ValueError(f"AbcActionMseEval.batch_size must be divisible by {jax.device_count()} devices")
        variants = {"task_prompt": 0.0}
        if self.unconditional:
            variants["no_prompt"] = 1.0
        batches = {}
        for name, prompt_dropout in variants.items():
            data = dataclasses.replace(
                config.data,
                prompt_dropout=prompt_dropout,
                top_camera_dropout=0.0,
                left_wrist_camera_dropout=0.0,
                right_wrist_camera_dropout=0.0,
            )
            eval_config = dataclasses.replace(
                config, data=data, batch_size=self.batch_size, num_workers=min(8, config.num_workers)
            )
            loader = _data_loader.create_data_loader(
                eval_config,
                sharding=ctx.data_sharding,
                shuffle=True,  # same seed -> same samples for both variants
                num_batches=self.num_batches,
                skip_norm_stats=config.skip_norm_stats,
                split="val",
            )
            batches[name] = [jax.device_get(b) for b in loader]
        data_config = loader.data_config()
        stats = (
            None if (config.skip_norm_stats or data_config.norm_stats is None) else data_config.norm_stats["actions"]
        )
        sample = jax.jit(
            self._sample_fn,
            in_shardings=(ctx.replicated_sharding, ctx.train_state_sharding, ctx.data_sharding),
            out_shardings=ctx.replicated_sharding,
        )
        _MSE_CACHE[key] = {
            "batches": batches,
            "stats": stats,
            "quantile": data_config.use_quantile_norm,
            "sample": sample,
        }
        return _MSE_CACHE[key]

    def _sample_fn(self, rng, state, observation):
        model = nnx.merge(state.model_def, _params_for_eval(state))
        return model.sample_actions(rng, observation, num_steps=self.num_flow_steps)

    def run(self, ctx: EvalContext) -> dict[str, float]:
        cache = self._cache(ctx)
        metrics = {}
        rng = jax.random.key(0)  # fixed noise: comparable across evals
        for name, batches in cache["batches"].items():
            preds, targets, task_ids = [], [], []
            for host_observation, actions in batches:
                observation = jax.device_put(host_observation, ctx.data_sharding)
                with sharding.set_mesh(ctx.mesh):
                    pred = cache["sample"](rng, ctx.train_state, observation)
                preds.append(np.asarray(jax.device_get(pred))[..., : abc.ACTION_DIM])
                targets.append(np.asarray(actions)[..., : abc.ACTION_DIM])
                task_ids.append(np.asarray(observation.dataset_id))
            pred = _unnormalize(np.concatenate(preds), cache["stats"], quantile=cache["quantile"])
            target = _unnormalize(np.concatenate(targets), cache["stats"], quantile=cache["quantile"])
            metrics.update(
                chunk_mse_metrics(
                    pred,
                    target,
                    np.concatenate(task_ids),
                    horizons=dict(self.horizons),
                    top_k=self.top_k,
                    prefix=f"val_mse/{name}",
                )
            )
        return metrics


@dataclasses.dataclass(frozen=True)
class AbcSimEval:
    """Asynchronous ABC simulator evaluation of the checkpoint saved at each eval step.

    ``interval`` must be a multiple of ``TrainConfig.save_interval`` so a checkpoint exists for the step.
    The worker runs on ``gpu`` (JAX without preallocation, plus EGL rendering), so give training memory
    headroom on that GPU (e.g. ``XLA_PYTHON_CLIENT_MEM_FRACTION=0.75``).
    """

    interval: int = 10_000
    tasks: tuple[str, ...] = DEFAULT_SIM_EVAL_TASKS
    worlds_per_task: int = 4
    seed: int = 20260511  # ABC's eval seed
    # Actions executed per policy query (ABC: 15 of the 30-step chunk).
    execute_steps: int = 15
    # Real-time chunking (ABC's default eval). Needs a model trained with action-prefix conditioning
    # (Pi0Config.action_prefix_max > 0). Off: ABC's --no-rtc synchronous loop.
    rtc: bool = False
    rtc_prefix_length: int = 4
    # ABC: 236 chunks x 15 actions ~ 120 s per episode.
    max_chunks: int = 236
    # "mujoco" = OpenGL (EGL headless); ABC's other options are "mjwarp" and "blender".
    camera_backend: str = "mujoco"
    num_flow_steps: int = 10
    # Also roll out with the task prompt dropped.
    unconditional: bool = False
    gpu: int = 7
    abc_python: str = "/raid/users/cglossop/venvs/abc/bin/python"

    def run(self, ctx: EvalContext) -> dict[str, float]:
        if jax.process_index() != 0:
            return {}
        if self.interval % ctx.config.save_interval:
            raise ValueError("AbcSimEval.interval must be a multiple of save_interval (it evaluates checkpoints).")
        metrics = self._collect(ctx)
        running = {s: p for s, p in _SIM_WORKERS.items() if p.poll() is None}
        if running:
            logger.warning("Sim eval for step(s) %s still running; skipping step %d.", sorted(running), ctx.step)
        else:
            self._launch(ctx)
        return metrics

    def _out_dir(self, ctx: EvalContext, step: int) -> pathlib.Path:
        return pathlib.Path(str(ctx.run_dir)) / "sim_eval" / str(step)

    def _launch(self, ctx: EvalContext) -> None:
        out_dir = self._out_dir(ctx, ctx.step)
        out_dir.mkdir(parents=True, exist_ok=True)
        config_path = out_dir / "train_config.pkl"
        try:
            config_path.write_bytes(pickle.dumps(ctx.config))
            config_args = ["--config-pickle", str(config_path)]
        except Exception:  # fall back to the registered config (CLI overrides are lost)
            logger.warning("TrainConfig is not picklable; sim eval loads %r by name.", ctx.config.name)
            config_args = ["--config-name", ctx.config.name]
        sim_args = dataclasses.asdict(self)
        cmd = [
            sys.executable,
            str(pathlib.Path(__file__).resolve().parents[3] / "scripts" / "abc_sim_eval_worker.py"),
            *config_args,
            "--checkpoint-dir",
            str(pathlib.Path(str(ctx.run_dir)) / str(ctx.step)),
            "--out-dir",
            str(out_dir),
            "--sim-args",
            json.dumps(sim_args),
        ]
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": str(self.gpu),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
        log = (out_dir / "worker.log").open("w")
        _SIM_WORKERS[ctx.step] = subprocess.Popen(
            cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        logger.info(
            "Launched ABC sim eval for step %d on GPU %d (logs: %s)", ctx.step, self.gpu, out_dir / "worker.log"
        )

    def _collect(self, ctx: EvalContext) -> dict[str, float]:
        metrics = {}
        root = pathlib.Path(str(ctx.run_dir)) / "sim_eval"
        if not root.exists():
            return metrics
        for step_dir in sorted(root.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else -1):
            if not step_dir.name.isdigit() or int(step_dir.name) in _SIM_LOGGED:
                continue
            result = step_dir / "results.json"
            if not result.exists():
                continue
            summary = json.loads(result.read_text())["summary"]
            # Latest finished checkpoint wins this logging step; older ones are logged once each in order.
            metrics = {f"sim_eval/{k}": v for k, v in summary.items()}
            metrics["sim_eval/checkpoint_step"] = int(step_dir.name)
            _SIM_LOGGED.add(int(step_dir.name))
        return metrics


def summarize_rollouts(rollouts: list[dict]) -> dict[str, float]:
    """Success rate / max progress per task (and per prompt condition), plus means over tasks."""
    summary: dict[str, float] = {}
    by_key: dict[tuple[str, str], list[dict]] = {}
    for r in rollouts:
        by_key.setdefault((r["condition"], r["task"]), []).append(r)
    for (condition, task), rs in sorted(by_key.items()):
        summary[f"{condition}/success/{task}"] = float(np.mean([r["success"] for r in rs]))
        summary[f"{condition}/max_reward/{task}"] = float(np.mean([r["max_reward"] for r in rs]))
    for condition in sorted({c for c, _ in by_key}):
        rates = [v for k, v in summary.items() if k.startswith(f"{condition}/success/")]
        summary[f"{condition}/success_mean"] = float(np.mean(rates))
    return summary
