"""Tests for Pi0CoT prefix handling.

Both tests here guard the same subtlety: an unused camera contributes a full block of *masked-out*
image tokens in the middle of the prefix, so prefix positions are not left-packed.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import openpi.models.model as _model
import openpi.models.pi0_config as _pi0_config
from openpi.models.pi0_cot import Pi0CoT
import openpi.shared.array_typing as at
import openpi.training.config as _config
import openpi.transforms as _transforms

# Static helper under test; exercised directly because its failure mode (indexing into a
# masked-out camera block) is invisible in end-to-end output.
_gather_last_valid_hidden = Pi0CoT._gather_last_valid_hidden  # noqa: SLF001


def test_gather_last_valid_hidden_skips_masked_image_blocks():
    """The gather must land on the last valid position, not on ``num_valid - 1``.

    Layout below mimics the real prefix: a valid camera block, a masked-out dummy camera block,
    then the prompt. ``num_valid - 1`` would index into the dummy block.
    """
    n_img, n_dummy, n_prompt = 4, 4, 3
    total = n_img + n_dummy + n_prompt
    mask = jnp.array([[True] * n_img + [False] * n_dummy + [True] * n_prompt])
    # Hidden state = its own index, so the returned value names the position it came from.
    prefix_out = jnp.arange(total, dtype=jnp.float32)[None, :, None]

    got = _gather_last_valid_hidden(prefix_out, mask)

    last_valid = n_img + n_dummy + n_prompt - 1
    assert got.shape == (1, 1, 1)
    assert float(got[0, 0, 0]) == float(last_valid)
    # The old implementation returned index num_valid-1, which sits inside the masked dummy block.
    assert float(got[0, 0, 0]) != float(n_img + n_prompt - 1)


def test_gather_last_valid_hidden_handles_trailing_padding_and_empty_rows():
    mask = jnp.array(
        [
            [True, True, False, False],  # trailing padding
            [False, False, False, False],  # degenerate all-masked row
        ]
    )
    prefix_out = jnp.tile(jnp.arange(4, dtype=jnp.float32)[None, :, None], (2, 1, 1))

    got = _gather_last_valid_hidden(prefix_out, mask)

    assert float(got[0, 0, 0]) == 1.0  # last True
    assert float(got[1, 0, 0]) == 0.0  # clamped, not negative-indexed


def _tiny_config(image_keys):
    return _pi0_config.Pi0CoTConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=32,
        action_horizon=4,
        max_token_len=16,
        max_subtask_len=8,
        max_reasoning_len=8,
        max_fast_len=8,
        use_fast_tokens=True,
        image_keys=image_keys,
    )


def _observation(config, image, extra_zero_cameras):
    b = image.shape[0]
    images = {"base_0_rgb": image}
    image_masks = {"base_0_rgb": jnp.ones((b,), bool)}
    for name in extra_zero_cameras:
        images[name] = jnp.zeros_like(image)
        image_masks[name] = jnp.zeros((b,), bool)

    def tok(n, seed):
        return jax.random.randint(jax.random.key(seed), (b, n), 0, 1000)

    with at.disable_typechecking():
        return _model.Observation(
            images=images,
            image_masks=image_masks,
            state=jax.random.uniform(jax.random.key(2), (b, config.action_dim)),
            tokenized_prompt=tok(config.max_token_len, 3),
            tokenized_prompt_mask=jnp.ones((b, config.max_token_len), bool),
            tokenized_reasoning=tok(config.max_reasoning_len, 4),
            tokenized_reasoning_mask=jnp.ones((b, config.max_reasoning_len), bool),
            tokenized_subtask=tok(config.max_subtask_len, 5),
            tokenized_subtask_mask=jnp.ones((b, config.max_subtask_len), bool),
            tokenized_fast=tok(config.max_fast_len, 6),
            tokenized_fast_mask=jnp.ones((b, config.max_fast_len), bool),
            action_loss_mask=jnp.ones((b, config.action_horizon), bool),
        )


def test_dropping_masked_dummy_cameras_is_a_no_op():
    """A single-camera model must be numerically identical to one fed masked-out dummy cameras.

    Masked image tokens are already excluded from attention, so dropping them should change only
    cost, never results (up to bfloat16 accumulation order). This is what makes
    ``image_keys=("base_0_rgb",)`` safe for driving
    configs -- and it only holds because ``_gather_last_valid_hidden`` indexes the last *valid*
    position rather than assuming the prefix is left-packed.
    """
    cfg_one = _tiny_config(("base_0_rgb",))
    cfg_three = _tiny_config(None)
    # Same rng -> identical parameters; SigLIP/Gemma weights do not depend on the camera count.
    model_one = cfg_one.create(jax.random.key(0))
    model_three = cfg_three.create(jax.random.key(0))

    b = 2
    image = jax.random.uniform(jax.random.key(1), (b, 224, 224, 3), minval=-1.0, maxval=1.0)
    obs_one = _observation(cfg_one, image, ())
    obs_three = _observation(cfg_three, image, ("left_wrist_0_rgb", "right_wrist_0_rgb"))

    rng = jax.random.key(7)
    actions = jax.random.uniform(jax.random.key(8), (b, cfg_one.action_horizon, cfg_one.action_dim))
    noise = jax.random.normal(jax.random.key(9), actions.shape)

    # Tolerances, not exact equality: the model runs in bfloat16, and a 256- vs 768-token prefix
    # changes matmul reduction order. The residual is accumulation noise, not a different prefix.
    np.testing.assert_allclose(
        np.asarray(model_one.compute_loss(rng, obs_one, actions, train=False)),
        np.asarray(model_three.compute_loss(rng, obs_three, actions, train=False)),
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        np.asarray(model_one.sample_actions(rng, obs_one, num_steps=2, noise=noise)),
        np.asarray(model_three.sample_actions(rng, obs_three, num_steps=2, noise=noise)),
        rtol=1e-4,
        atol=1e-4,
    )

    cot_one = model_one.sample_cot(rng, obs_one, temperature=0.0)
    cot_three = model_three.sample_cot(rng, obs_three, temperature=0.0)
    for key in ("tokenized_reasoning", "tokenized_subtask"):
        np.testing.assert_array_equal(np.asarray(cot_one[key]), np.asarray(cot_three[key]))


def test_image_keys_drives_inputs_spec():
    """scripts/train.py validates batches against inputs_spec, so it must follow image_keys."""
    obs_spec, _ = _tiny_config(("base_0_rgb",)).inputs_spec(batch_size=1)
    assert list(obs_spec.images) == ["base_0_rgb"]
    assert list(obs_spec.image_masks) == ["base_0_rgb"]

    obs_spec, _ = _tiny_config(None).inputs_spec(batch_size=1)
    assert list(obs_spec.images) == list(_model.IMAGE_KEYS)


def test_inference_image_keys_overrides_image_keys_at_sampling_only():
    config = dataclasses.replace(
        _tiny_config(None), inference_image_keys=("base_0_rgb",)
    )
    model = config.create(jax.random.key(0))
    # Training consumes every declared stream; sampling honours the inference override.
    assert model._image_keys == tuple(_model.IMAGE_KEYS)  # noqa: SLF001
    assert model._preprocess_image_keys == ("base_0_rgb",)  # noqa: SLF001


def test_exempted_dim_is_identity_and_round_trips():
    """An exempted action dim must pass through Normalize/Unnormalize unchanged.

    This is what lets a config rebalance the flow-matching loss across action dims without
    touching the policy's output contract: the deployed consumer keeps seeing the units the
    dataset emitted.
    """
    stats = _transforms.NormStats(
        mean=np.array([0.2, 0.0, 0.95, 0.0]),
        std=np.array([0.2, 0.1, 0.1, 0.3]),
        q01=np.array([-0.1, -0.35, 0.51, -0.8]),
        q99=np.array([0.7, 0.35, 1.0, 0.8]),
    )
    exempt = _config._exempt_dims_from_normalization({"actions": stats}, "actions", (2,))["actions"]  # noqa: SLF001

    # Identity-inducing stats on the exempted dim only; every other dim untouched.
    np.testing.assert_array_equal(exempt.q01, [-0.1, -0.35, -1.0, -0.8])
    np.testing.assert_array_equal(exempt.q99, [0.7, 0.35, 1.0, 0.8])
    np.testing.assert_array_equal(exempt.mean, [0.2, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(exempt.std, [0.2, 0.1, 1.0, 0.3])
    # Source stats must not be mutated in place.
    assert stats.q01[2] == 0.51

    rng = np.random.default_rng(0)
    raw = rng.uniform(-1.5, 1.5, (64, 4))
    for use_quantiles in (True, False):
        normed = _transforms.Normalize({"actions": exempt}, use_quantiles=use_quantiles)({"actions": raw})["actions"]
        np.testing.assert_allclose(normed[:, 2], raw[:, 2], atol=1e-5)
        # Round trip through the 32-dim padded vector the policy actually returns.
        padded = np.concatenate([normed, np.zeros((64, 28))], axis=-1)
        back = _transforms.Unnormalize({"actions": exempt}, use_quantiles=use_quantiles)({"actions": padded})[
            "actions"
        ]
        np.testing.assert_allclose(back[:, :4], raw, atol=1e-5)


def test_exempt_dims_rejects_out_of_range():
    stats = _transforms.NormStats(mean=np.zeros(4), std=np.ones(4), q01=-np.ones(4), q99=np.ones(4))
    with pytest.raises(ValueError, match="out of range"):
        _config._exempt_dims_from_normalization({"actions": stats}, "actions", (4,))  # noqa: SLF001
    with pytest.raises(ValueError, match="no 'actions' norm stats"):
        _config._exempt_dims_from_normalization({"state": stats}, "actions", (0,))  # noqa: SLF001
