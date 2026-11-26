"""Generic hook for evaluations that ``scripts/train.py`` runs every ``interval`` steps.

A config lists hooks in ``TrainConfig.periodic_evals``; train.py calls ``hook.run(ctx)`` right after any
checkpoint save at that step and logs the returned metrics to wandb. Hooks must be cheap to construct and
hashable/frozen (they live on the frozen TrainConfig); keep any state in module-level caches.
"""

import dataclasses
from typing import Any, Protocol, runtime_checkable

import etils.epath as epath
import jax


@dataclasses.dataclass
class EvalContext:
    step: int
    train_state: Any  # training_utils.TrainState
    train_state_sharding: Any
    data_sharding: jax.sharding.Sharding
    replicated_sharding: jax.sharding.Sharding
    mesh: jax.sharding.Mesh
    config: Any  # TrainConfig
    run_dir: epath.Path  # Directory holding the numbered checkpoint subdirectories.


@runtime_checkable
class PeriodicEval(Protocol):
    interval: int

    def run(self, ctx: EvalContext) -> dict[str, float]:
        """Return metrics to log at ``ctx.step`` (may be empty)."""
