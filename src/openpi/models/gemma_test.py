import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import gemma


def _run(knowledge_insulation: bool, *, module=gemma):
    configs = [module.get_config("dummy"), module.get_config("dummy")]
    llm = module.Module(configs=configs, embed_dtype="float32", knowledge_insulation=knowledge_insulation)
    b, n_backbone, n_action, width = 2, 5, 3, configs[0].width
    x_backbone = jax.random.normal(jax.random.key(1), (b, n_backbone, width))
    x_action = jax.random.normal(jax.random.key(2), (b, n_action, configs[1].width))
    total = n_backbone + n_action
    # Backbone tokens attend bidirectionally among themselves; action tokens see everything.
    mask = jnp.ones((b, total, total), bool).at[:, :n_backbone, n_backbone:].set(False)
    positions = jnp.broadcast_to(jnp.arange(total), (b, total))
    # gemma.Module defines its own `init` method, so call linen's explicitly.
    params = nn.Module.init(llm, jax.random.key(0), [x_backbone, x_action], positions, mask)

    def action_out(params, x_backbone, x_action):
        (_, out_action), _ = llm.apply(params, [x_backbone, x_action], positions, mask)
        return out_action

    def loss(params, x_backbone, x_action):
        return jnp.sum(action_out(params, x_backbone, x_action) ** 2)

    grads = jax.grad(loss, argnums=(0, 1, 2))(params, x_backbone, x_action)
    return action_out(params, x_backbone, x_action), grads


def test_knowledge_insulation_blocks_action_loss_into_backbone():
    out_plain, (_, g_backbone_plain, g_action_plain) = _run(knowledge_insulation=False)
    out_ki, (g_params_ki, g_backbone_ki, g_action_ki) = _run(knowledge_insulation=True)

    # Forward pass is unchanged: stop_gradient only affects the backward pass.
    np.testing.assert_allclose(out_ki, out_plain, rtol=1e-6, atol=1e-6)
    # Without KI the action loss reaches the backbone tokens; with KI it must not.
    assert float(jnp.abs(g_backbone_plain).sum()) > 0
    np.testing.assert_array_equal(g_backbone_ki, 0.0)
    # The action expert itself still trains.
    assert float(jnp.abs(g_action_ki).sum()) > 0
    # Backbone-expert weights get no gradient from the action loss either.
    backbone_qkv = g_params_ki["params"]["layers"]["attn"]["q_einsum"]["w"]
    np.testing.assert_array_equal(backbone_qkv, 0.0)


@pytest.mark.parametrize("knowledge_insulation", [False, True])
def test_kv_cache_paths_match_full_forward(knowledge_insulation):
    """KI only applies without a KV cache; prefix-then-suffix decoding must match one full pass."""
    configs = [gemma.get_config("dummy"), gemma.get_config("dummy")]
    llm = gemma.Module(configs=configs, embed_dtype="float32", knowledge_insulation=knowledge_insulation)
    b, n_prefix, n_suffix, width = 2, 4, 3, configs[0].width
    x_prefix = jax.random.normal(jax.random.key(1), (b, n_prefix, width))
    x_suffix = jax.random.normal(jax.random.key(2), (b, n_suffix, width))
    total = n_prefix + n_suffix
    mask = jnp.ones((b, total, total), bool).at[:, :n_prefix, n_prefix:].set(False)
    positions = jnp.broadcast_to(jnp.arange(total), (b, total))
    params = nn.Module.init(llm, jax.random.key(0), [x_prefix, x_suffix], positions, mask)

    (_, full_suffix), _ = llm.apply(params, [x_prefix, x_suffix], positions, mask)
    _, kv_cache = llm.apply(params, [x_prefix, None], positions[:, :n_prefix], mask[:, :n_prefix, :n_prefix])
    (_, cached_suffix), _ = llm.apply(
        params, [None, x_suffix], positions[:, n_prefix:], mask[:, n_prefix:, :], kv_cache=kv_cache
    )
    np.testing.assert_allclose(cached_suffix, full_suffix, rtol=1e-5, atol=1e-5)
