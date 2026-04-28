"""tf.data / DLimp sharding for JAX multi-host training."""

from __future__ import annotations

import jax


def shard_dataset_then_batch(dataset, batch_size: int):
    """Shard examples across JAX processes, then batch with per-process batch size.

    Global batch size per step is ``batch_size`` (sum over processes). Each process
    yields batches of shape ``(batch_size // process_count, ...)``.
    """
    n = jax.process_count()
    if n <= 1:
        return dataset.batch(batch_size)
    if batch_size % n != 0:
        raise ValueError(
            f"batch_size ({batch_size}) must be divisible by jax.process_count() ({n}) "
            "for multi-host RLDS training."
        )
    per_host = batch_size // n
    return dataset.shard(n, jax.process_index()).batch(per_host)
