import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import context_smoothing as _cs
from openpi.models import pi0_config


def _dummy_config(csp: _cs.ContextSmoothingConfig | None) -> pi0_config.Pi0CoTConfig:
    return pi0_config.Pi0CoTConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=32,
        action_horizon=10,
        max_token_len=32,
        max_subtask_len=8,
        max_reasoning_len=8,
        context_smoothing=csp,
    )


def test_cosine_schedule_spans_the_full_spectrum():
    config = _cs.ContextSmoothingConfig(schedule="cosine")
    t = jnp.array([0.0, 1.0], dtype=jnp.float32)
    signal = np.sqrt(np.asarray(_cs.alpha_bar(t, config)))
    # t=0 is the clean, precise-imitation end; t=1 leaves the context uninformative.
    assert signal[0] == pytest.approx(1.0, abs=1e-3)
    assert signal[1] < 0.05


def test_linear_schedule_matches_tmrl_reference():
    config = _cs.ContextSmoothingConfig(schedule="linear", num_train_steps=100)
    betas = np.linspace(1e-4, 0.02, 100)
    expected = np.cumprod(1.0 - betas)
    t = jnp.array([0.0, 0.5, 0.999], dtype=jnp.float32)
    got = np.asarray(_cs.alpha_bar(t, config))
    np.testing.assert_allclose(got, expected[[0, 50, 99]], rtol=1e-5)


def test_noise_context_interpolates_and_preserves_scale():
    x = jax.random.normal(jax.random.key(0), (2, 16, 64), dtype=jnp.float32) * 7.5
    config = _cs.ContextSmoothingConfig(schedule="cosine", noise_scale="rms")

    clean = _cs.noise_context(x, jnp.zeros((2,), jnp.float32), jax.random.key(1), config)
    noisy = _cs.noise_context(x, jnp.ones((2,), jnp.float32), jax.random.key(1), config)

    # t=0 leaves the context essentially untouched, t=1 decorrelates it.
    assert float(jnp.corrcoef(clean.ravel(), x.ravel())[0, 1]) > 0.99
    assert abs(float(jnp.corrcoef(noisy.ravel(), x.ravel())[0, 1])) < 0.1
    # rms mode keeps the token scale in-distribution for the LLM at every noise level.
    assert float(jnp.sqrt(jnp.mean(noisy**2))) == pytest.approx(float(jnp.sqrt(jnp.mean(x**2))), rel=0.05)


def test_context_timestep_branch_is_zero_at_init():
    """A freshly-built CSP model must be identical to a stock pi0.5 checkpoint, so it warm-starts."""
    config = _dummy_config(_cs.ContextSmoothingConfig())
    model = config.create(jax.random.key(0))
    obs = config.fake_obs(batch_size=2)

    x_t = jnp.zeros((2, config.action_horizon, config.action_dim), jnp.float32)
    time = jnp.full((2,), 0.5, jnp.float32)

    *_, adarms_off = model._embed_action_suffix(obs, x_t, time, None)  # noqa: SLF001
    *_, adarms_on = model._embed_action_suffix(obs, x_t, time, jnp.ones((2,), jnp.float32))  # noqa: SLF001

    # ctx_time_mlp_out is zero-initialized, so the t_context branch contributes exactly nothing.
    np.testing.assert_array_equal(np.asarray(adarms_on), np.asarray(adarms_off))


def test_compute_loss_reports_context_metrics():
    config = _dummy_config(_cs.ContextSmoothingConfig())
    model = config.create(jax.random.key(0))
    obs, actions = config.fake_obs(batch_size=8), config.fake_act(batch_size=8)

    loss, metrics = model.compute_loss_with_aux(jax.random.key(1), obs, actions, train=True)

    assert loss.shape == (8, config.action_horizon)
    assert metrics["t_context"].shape == (8,)
    assert jnp.all((metrics["t_context"] >= 0.0) & (metrics["t_context"] <= 1.0))
    assert "flow_loss_clean_ctx" in metrics
    assert "flow_loss_noisy_ctx" in metrics


def test_disabled_by_default():
    """Without a context_smoothing config there are no new params and no t_context plumbing."""
    config = _dummy_config(None)
    model = config.create(jax.random.key(0))
    obs, actions = config.fake_obs(batch_size=2), config.fake_act(batch_size=2)

    _, metrics = model.compute_loss_with_aux(jax.random.key(1), obs, actions, train=True)
    assert "t_context" not in metrics
    assert not hasattr(model, "ctx_time_mlp_in")

    with pytest.raises(ValueError, match="context_smoothing"):
        model.sample_actions(jax.random.key(2), obs, num_steps=2, t_context=0.5)


def test_sample_actions_accepts_t_context():
    config = _dummy_config(_cs.ContextSmoothingConfig())
    model = config.create(jax.random.key(0))
    obs = config.fake_obs(batch_size=2)

    for t_context in (None, 0.0, 1.0):
        actions = model.sample_actions(jax.random.key(1), obs, num_steps=2, t_context=t_context)
        assert actions.shape == (2, config.action_horizon, config.action_dim)
