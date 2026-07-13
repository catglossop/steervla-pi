"""Context-Smoothed Pre-training (CSP): a forward noise schedule over the policy's context.

CSP trains a policy across all noise levels of a diffusion-style forward schedule applied to
the *context* (here, the SigLIP image tokens in the VLM prefix) rather than to the actions.
``t_context = 0`` leaves the context clean and recovers precise imitation; ``t_context = 1``
leaves it uninformative, so the policy falls back to the broad marginal over actions. A
high-level policy (TMRL) later selects ``t_context`` per action chunk.

The noise is applied to *embeddings*, not pixels: ``x_t = sqrt(abar_t) * x + sqrt(1 - abar_t) * eps``.
The context timestep reaches the action expert through adaRMS alongside the flow-matching
timestep -- the analog of ``DualTimestepEncoder`` in the reference TMRL implementation.

This module is intentionally dependency-free (jax only) so that it can be imported from
``pi0_config`` without creating an import cycle. The learned timestep-embedding parameters
live on the model (see ``Pi0CoT.ctx_time_mlp_in`` / ``ctx_time_mlp_out``).
"""

import dataclasses
from typing import Literal

import jax
import jax.numpy as jnp

from openpi.shared import array_typing as at


@dataclasses.dataclass(frozen=True)
class ContextSmoothingConfig:
    """Forward-schedule settings for Context-Smoothed Pre-training."""

    # Noise schedule over abar_t.
    #
    # "cosine" (Nichol & Dhariwal) spans the full spectrum: abar goes from ~1 at t=0 to ~0 at
    # t=1, so the maximum noise level really does leave the context uninformative.
    #
    # "linear" reproduces the reference TMRL schedule (betas = linspace(1e-4, 0.02, num_train_steps),
    # abar = cumprod(1 - betas)). Note that those betas are DDPM's, which assume T=1000; at the
    # TMRL default of num_train_steps=100 they only reach sqrt(abar) = 0.60 at t=1, i.e. 60% of the
    # clean context survives even at maximum noise. Use num_train_steps=1000 for full corruption.
    schedule: Literal["cosine", "linear"] = "cosine"

    # Number of discrete levels the continuous t_context in [0, 1) is bucketed into. Matches the
    # reference implementation, which indexes a precomputed abar table.
    num_train_steps: int = 100

    # How the noise is scaled relative to the context embeddings.
    #
    # SigLIP token embeddings are not unit-variance, so an unscaled N(0, 1) corruption produces
    # tokens whose norm is wrong for the LLM's input distribution -- "maximum noise" would read as
    # a broken input rather than an uninformative one. "rms" matches the noise scale to each
    # token's own RMS (stop-gradient), so t=1 yields a scale-correct uninformative token.
    # "unit" uses plain N(0, 1), matching the reference implementation.
    noise_scale: Literal["rms", "unit"] = "rms"

    # Cosine schedule offset (Nichol & Dhariwal use 0.008). Unused when schedule="linear".
    cosine_offset: float = 0.008


@at.typecheck
def alpha_bar(
    t: at.Float[at.Array, " b"], config: ContextSmoothingConfig
) -> at.Float[at.Array, " b"]:
    """abar_t for continuous t in [0, 1], bucketed into ``config.num_train_steps`` levels."""
    n = config.num_train_steps

    if config.schedule == "linear":
        betas = jnp.linspace(1e-4, 0.02, n, dtype=jnp.float32)
        table = jnp.cumprod(1.0 - betas)
    else:
        s = config.cosine_offset
        steps = jnp.arange(n, dtype=jnp.float32) / n
        f = jnp.cos((steps + s) / (1.0 + s) * jnp.pi / 2.0) ** 2
        table = f / f[0]

    idx = jnp.clip((t * n).astype(jnp.int32), 0, n - 1)
    return table[idx]


@at.typecheck
def sample_t_context(
    rng: at.KeyArrayLike, batch_shape: tuple[int, ...], config: ContextSmoothingConfig
) -> at.Float[at.Array, " b"]:
    """Uniform over the full spectrum of noise levels, as in the reference implementation."""
    del config
    return jax.random.uniform(rng, batch_shape, dtype=jnp.float32)


@at.typecheck
def noise_context(
    x: at.Float[at.Array, "b s emb"],
    t: at.Float[at.Array, " b"],
    rng: at.KeyArrayLike,
    config: ContextSmoothingConfig,
) -> at.Float[at.Array, "b s emb"]:
    """Apply the forward schedule to a block of context tokens."""
    ab = alpha_bar(t, config)[:, None, None]
    eps = jax.random.normal(rng, x.shape, dtype=x.dtype)

    if config.noise_scale == "rms":
        # Per-token RMS, detached: the scale is a property of the schedule, not something the
        # vision encoder should be able to shrink to dodge the noise.
        rms = jnp.sqrt(jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True) + 1e-6)
        eps = eps * jax.lax.stop_gradient(rms).astype(eps.dtype)

    return (jnp.sqrt(ab) * x + jnp.sqrt(1.0 - ab) * eps).astype(x.dtype)
