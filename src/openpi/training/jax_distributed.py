"""JAX multi-host bootstrap (Cloud TPU pod, Slurm, or explicit env).

Call :func:`initialize_if_needed` once at the start of ``main()`` before any JAX
computations that depend on global device topology (mesh, ``device_count``, etc.).

Environment variables (optional, explicit bootstrap):

* ``JAX_COORDINATOR_ADDRESS`` — ``host:port`` for process 0's coordinator service
* ``JAX_NUM_PROCESSES`` — total number of processes (hosts)
* ``JAX_PROCESS_INDEX`` — this process's index in ``0 .. num_processes-1``

On supported Cloud TPU VM pods, a no-argument ``jax.distributed.initialize()`` may
succeed when the above are unset. On single-host machines, initialization is skipped.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("openpi")


def initialize_if_needed() -> None:
    """Initialize JAX distributed runtime if this is a multi-host job."""
    import jax

    if jax.distributed.is_initialized():
        logger.info(
            "JAX distributed already initialized: process_count=%s process_index=%s",
            jax.process_count(),
            jax.process_index(),
        )
        return

    coord = os.environ.get("JAX_COORDINATOR_ADDRESS")
    nproc_raw = os.environ.get("JAX_NUM_PROCESSES")
    pid_raw = os.environ.get("JAX_PROCESS_INDEX")

    if coord and nproc_raw is not None and pid_raw is not None:
        nproc = int(nproc_raw)
        pid = int(pid_raw)
        jax.distributed.initialize(
            coordinator_address=coord,
            num_processes=nproc,
            process_id=pid,
        )
        logger.info(
            "JAX distributed initialized from env: process_index=%s/%s coordinator=%s",
            pid,
            nproc,
            coord,
        )
        return

    try:
        jax.distributed.initialize()
        logger.info(
            "JAX distributed initialized (auto): process_count=%s process_index=%s",
            jax.process_count(),
            jax.process_index(),
        )
    except Exception as e:
        logger.info("JAX distributed not initialized (single-host): %s", e)
