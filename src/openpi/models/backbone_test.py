"""The Gemma 3 backbone option, end to end on tiny models (dummy_gemma3 sizes, 224px to stay cheap)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import backbone as _backbone
from openpi.models import pi0_config
from openpi.models import pi0_fast
from openpi.models import pi0_cot_test
from openpi.shared import array_typing as at


def _pi05_gemma3(**kw):
    kw = {
        "image_resolution": 224,
        "paligemma_variant": "dummy_gemma3",
        "action_expert_variant": "dummy_gemma3",
        **kw,
    }
    return pi0_config.Pi0Config(pi05=True, backbone="gemma3", **kw)


@pytest.mark.parametrize("knowledge_insulation", [False, True])
def test_pi05_gemma3_loss_and_sampling(knowledge_insulation):
    config = _pi05_gemma3(knowledge_insulation=knowledge_insulation)
    model = config.create(jax.random.key(0))
    with at.disable_typechecking():
        obs, act = config.fake_obs(2), config.fake_act(2)
    loss = model.compute_loss(jax.random.key(1), obs, act, train=True)
    assert loss.shape == (2, config.action_horizon) and bool(jnp.all(jnp.isfinite(loss)))
    actions = model.sample_actions(jax.random.key(2), obs, num_steps=2)
    assert actions.shape == (2, config.action_horizon, config.action_dim)
    # 256 Gemma 3 image tokens per camera, projected into the LLM width.
    tokens, _ = model.PaliGemma.img(obs.images["base_0_rgb"], train=False)
    assert tokens.shape == (2, 256, 64)


def test_pi0_cot_gemma3_loss_sampling_and_cot():
    import dataclasses

    config = dataclasses.replace(
        pi0_cot_test._tiny_config(("base_0_rgb",)),  # noqa: SLF001
        backbone="gemma3",
        image_resolution=224,
        paligemma_variant="dummy_gemma3",
        action_expert_variant="dummy_gemma3",
    )
    model = config.create(jax.random.key(0))
    image = jax.random.uniform(jax.random.key(1), (2, 224, 224, 3), minval=-1.0, maxval=1.0)
    obs = pi0_cot_test._observation(config, image, ())  # noqa: SLF001
    actions = jax.random.uniform(jax.random.key(8), (2, config.action_horizon, config.action_dim))
    loss, aux = model.compute_loss_with_aux(jax.random.key(7), obs, actions, train=True)
    assert bool(jnp.all(jnp.isfinite(loss))) and "cot_reasoning_ce" in aux
    cot = model.sample_cot(jax.random.key(3), obs, temperature=0.0)
    assert cot["tokenized_reasoning"].shape == (2, config.max_reasoning_len)
    # Delimiters come from the Gemma 3 vocab (<unused> ids above 256k), not PaliGemma's.
    assert int(cot["tokenized_reasoning"][0, 0]) == model._start_of_reasoning_id > 256_000  # noqa: SLF001
    sampled = model.sample_actions(jax.random.key(4), obs, num_steps=2)
    assert sampled.shape == actions.shape


def test_pi0_fast_gemma3_loss_and_decoding():
    config = pi0_fast.Pi0FASTConfig(backbone="gemma3", image_resolution=224, paligemma_variant="dummy_gemma3")
    model = config.create(jax.random.key(0))
    with at.disable_typechecking():
        obs, act = config.fake_obs(2), config.fake_act(2)
    loss = model.compute_loss(jax.random.key(1), obs, act, train=True)
    assert bool(jnp.all(jnp.isfinite(loss)))
    tokens = model.sample_actions(jax.random.key(2), obs, max_decoding_steps=3)
    assert tokens.shape[0] == 2


def test_backbone_validation_and_resolution():
    assert _backbone.resolution(pi0_config.Pi0Config()) == (224, 224)
    assert _backbone.resolution(_pi05_gemma3(image_resolution=None)) == (896, 896)
    with pytest.raises(ValueError, match="paligemma_variant"):
        pi0_config.Pi0Config(backbone="gemma3").create(jax.random.key(0))
    with pytest.raises(ValueError, match="action_expert_variant"):
        _pi05_gemma3(action_expert_variant="gemma_300m").create(jax.random.key(0))
    with pytest.raises(ValueError, match="multiple of 224"):
        _pi05_gemma3(image_resolution=500).create(jax.random.key(0))
    np.testing.assert_equal(_backbone.DEFAULT_RESOLUTION, {"paligemma": 224, "gemma3": 896})
