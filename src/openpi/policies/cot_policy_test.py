import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.policies import cot_policy
from openpi.shared import array_typing as at


def _tiny_model():
    config = pi0_config.Pi0CoTConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=4,
        max_token_len=16,
        max_subtask_len=8,
        max_reasoning_len=8,
        image_keys=("base_0_rgb",),
    )
    return config, config.create(jax.random.key(0))


def _unbatched_inputs(config):
    tok = np.asarray(jax.random.randint(jax.random.key(3), (config.max_token_len,), 0, 1000), np.int32)
    return {
        "image": {"base_0_rgb": np.asarray(jax.random.uniform(jax.random.key(1), (224, 224, 3), minval=-1, maxval=1))},
        "image_mask": {"base_0_rgb": np.True_},
        "state": np.asarray(jax.random.uniform(jax.random.key(2), (config.action_dim,))),
        "tokenized_prompt": tok,
        "tokenized_prompt_mask": np.ones(config.max_token_len, bool),
        "tokenized_reasoning": np.zeros(config.max_reasoning_len, np.int32),
        "tokenized_reasoning_mask": np.zeros(config.max_reasoning_len, bool),
        "tokenized_subtask": np.zeros(config.max_subtask_len, np.int32),
        "tokenized_subtask_mask": np.zeros(config.max_subtask_len, bool),
    }


def test_infer_with_cot_matches_direct_model_calls():
    config, model = _tiny_model()
    # Policy defaults to rng=jax.random.key(0) (upstream's `rng or ...` cannot take a typed key).
    rng = jax.random.key(0)
    policy = cot_policy.CoTPolicy(
        model,
        sample_kwargs={"num_steps": 2, "image_keys": ["base_0_rgb"]},
        cot_sample_kwargs={"temperature": 0.0},
    )
    inputs = _unbatched_inputs(config)
    out = policy.infer_with_cot(inputs)

    # Reference: the same rng split and the same two model calls, made directly.
    _, rng_cot, rng_act = jax.random.split(rng, 3)
    with at.disable_typechecking():
        obs = _model.Observation.from_dict(jax.tree.map(lambda x: jnp.asarray(x)[None], inputs))
    cot = model.sample_cot(rng_cot, obs, temperature=0.0)
    obs = obs.replace(**{k: cot[k] for k in cot if k.startswith("tokenized_")})
    actions = model.sample_actions(rng_act, obs, num_steps=2, image_keys=("base_0_rgb",))

    np.testing.assert_array_equal(out["tokenized_reasoning"], np.asarray(cot["tokenized_reasoning"][0]))
    np.testing.assert_array_equal(out["tokenized_subtask"], np.asarray(cot["tokenized_subtask"][0]))
    np.testing.assert_allclose(out["actions"], np.asarray(actions[0]), rtol=1e-5, atol=1e-5)
    assert set(out["policy_timing"]) == {"infer_cot_ms", "infer_actions_ms", "infer_ms"}
