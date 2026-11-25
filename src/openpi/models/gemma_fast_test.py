import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import gemma
from openpi.models import gemma_fast


def _fast_module(variant):
    return gemma_fast.Module(**gemma_fast.get_config(variant), embed_dtype="float32")


def fast_vs_mixture_logits(variant, params=None, tokens=None):
    """Logits from gemma_fast.Module and single-expert gemma.Module sharing one parameter tree."""
    fast = _fast_module(variant)
    if tokens is None:
        tokens = jax.random.randint(jax.random.key(1), (2, 9), 0, 1000)
    if params is None:
        params = nn.Module.init(fast, jax.random.key(0), tokens)
    fast_logits, _, _ = fast.apply(params, tokens)

    mixture = gemma.Module(configs=[gemma.get_config(variant)], embed_dtype="float32")
    t = tokens.shape[1]
    emb = mixture.apply(params, tokens, method="embed")
    b = tokens.shape[0]
    causal = jnp.broadcast_to(jnp.tril(jnp.ones((t, t), bool)), (b, t, t))
    (out,), _ = mixture.apply(params, [emb], jnp.broadcast_to(jnp.arange(t), tokens.shape), causal)
    return fast_logits, mixture.apply(params, out, method="decode_logits")


@pytest.mark.parametrize("variant", ["dummy_gemma3"])
def test_gemma_fast_matches_gemma_module(variant):
    fast_logits, mixture_logits = fast_vs_mixture_logits(variant)
    np.testing.assert_allclose(fast_logits, mixture_logits, rtol=1e-5, atol=1e-5)


def test_gemma_fast_gemma3_decode_matches_prefill():
    """Token-by-token decoding with the KV cache must reproduce the full-sequence logits."""
    fast = _fast_module("dummy_gemma3")
    tokens = jax.random.randint(jax.random.key(1), (1, 6), 0, 1000)
    params = nn.Module.init(fast, jax.random.key(0), tokens)
    full_logits, _, _ = fast.apply(params, tokens)

    cache_size, t0 = 8, 3
    mask = jnp.tril(jnp.ones((1, t0, cache_size), bool))
    logits, kv_cache, _ = fast.apply(params, tokens[:, :t0], positions=jnp.arange(t0)[None], mask=mask, decode=True)
    step_logits = [logits[:, -1]]
    for t in range(t0, tokens.shape[1]):
        step_mask = (jnp.arange(cache_size) <= t)[None, None, :]
        logits, kv_cache, _ = fast.apply(
            params, tokens[:, t : t + 1], positions=jnp.array([[t]]), mask=step_mask, decode=True, kv_cache=kv_cache
        )
        step_logits.append(logits[:, -1])
    np.testing.assert_allclose(jnp.stack(step_logits, 1), full_logits[:, t0 - 1 :], rtol=1e-4, atol=1e-4)
