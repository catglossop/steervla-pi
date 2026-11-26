import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import pi0_config
from openpi.policies import abc_policy
from openpi.shared import array_typing as at
from openpi.training import abc_sim_eval


def _raw_obs(prompt="spell cat"):
    rng = np.random.default_rng(0)
    return {
        "observation/images/top": rng.integers(0, 255, (168, 224, 3), dtype=np.uint8),
        "observation/images/left_wrist": rng.integers(0, 255, (3, 168, 224), dtype=np.uint8),  # CHW, as the sim sends
        "observation/images/right_wrist": rng.integers(0, 255, (168, 224, 3), dtype=np.uint8),
        "observation/state": rng.normal(size=14).astype(np.float32),
        "actions": rng.normal(size=(30, 14)).astype(np.float32),
        "prompt": prompt,
        "task_group": np.int32(16),
    }


def test_abc_inputs_layout():
    out = abc_policy.AbcInputs(model_type=pi0_config.Pi0Config(pi05=True).model_type)(_raw_obs())
    assert set(out["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert out["image"]["left_wrist_0_rgb"].shape == (168, 224, 3)  # CHW converted
    assert all(bool(m) for m in out["image_mask"].values())
    assert out["actions"].shape == (30, 14)
    assert int(out["dataset_id"]) == 16


def test_dropout_probabilities_are_independent():
    raw = _raw_obs()
    always_prompt = abc_policy.AbcObservationDropout(prompt=1.0)(raw)
    assert always_prompt["prompt"] == ""
    np.testing.assert_array_equal(always_prompt["observation/images/top"], raw["observation/images/top"])
    left_only = abc_policy.AbcObservationDropout(left_wrist=1.0)(raw)
    assert left_only["prompt"] == "spell cat"
    assert not np.any(left_only["observation/images/left_wrist"])
    inputs = abc_policy.AbcInputs(model_type=pi0_config.Pi0Config(pi05=True).model_type)(left_only)
    assert not inputs["image_mask"]["left_wrist_0_rgb"]
    assert inputs["image_mask"]["base_0_rgb"]
    assert inputs["image_mask"]["right_wrist_0_rgb"]
    unchanged = abc_policy.AbcObservationDropout()(raw)
    assert unchanged["prompt"] == raw["prompt"]
    assert "observation/image_mask/top" not in unchanged


def test_chunk_mse_metrics_horizons_and_worst_k():
    target = np.zeros((6, 30, 14))
    pred = np.zeros((6, 30, 14))
    tasks = np.array([0, 0, 1, 1, 2, 2])
    pred[tasks == 1] = 1.0  # task 1: error 1 everywhere
    pred[tasks == 2, 15:] = 2.0  # task 2: error 4 only in the second half-second
    m = abc_sim_eval.chunk_mse_metrics(pred, target, tasks, horizons={"1.0s": 30, "0.5s": 15}, top_k=2, prefix="p")
    assert m["p/1.0s/task/conveyor_pick"] == 1.0
    assert m["p/1.0s/task/count_into_opaque_box"] == 2.0
    assert m["p/0.5s/task/count_into_opaque_box"] == 0.0
    assert m["p/1.0s/mean"] == pytest.approx(1.0)
    assert m["p/1.0s/worst2"] == pytest.approx(1.5)
    assert m["p/0.5s/worst2"] == pytest.approx(0.5)


def test_summarize_rollouts():
    rollouts = [
        {"task": "spell_cat", "condition": "task_prompt", "success": True, "max_reward": 1.0},
        {"task": "spell_cat", "condition": "task_prompt", "success": False, "max_reward": 0.5},
        {"task": "conveyor_pick", "condition": "task_prompt", "success": False, "max_reward": 0.0},
    ]
    s = abc_sim_eval.summarize_rollouts(rollouts)
    assert s["task_prompt/success/spell_cat"] == 0.5
    assert s["task_prompt/success_mean"] == 0.25


def _rtc_config(prefix_max):
    return pi0_config.Pi0Config(
        pi05=True,
        backbone="gemma3",
        image_resolution=224,
        paligemma_variant="dummy_gemma3",
        action_expert_variant="dummy_gemma3",
        action_horizon=30,
        action_prefix_max=prefix_max,
    )


def test_rtc_prefix_conditioning_loss_and_sampling():
    config = _rtc_config(prefix_max=8)
    model = config.create(jax.random.key(0))
    with at.disable_typechecking():
        obs, act = config.fake_obs(4), config.fake_act(4)
    act = jax.random.normal(jax.random.key(5), act.shape)
    loss = model.compute_loss(jax.random.key(1), obs, act, train=True)
    # Prefix lengths are U[0, 8): loss is exactly zero on each sample's prefix and nonzero beyond position 7.
    assert loss.shape == (4, 30)
    assert bool(jnp.all(loss[:, 8:] > 0))
    assert int(jnp.sum(loss[:, :8] == 0)) > 0

    prefix = jax.random.normal(jax.random.key(6), act.shape)
    prefix_len = jnp.array([0, 1, 4, 7])
    out = model.sample_actions(jax.random.key(2), obs, num_steps=3, action_prefix=prefix, action_prefix_len=prefix_len)
    for i, n in enumerate(prefix_len.tolist()):
        np.testing.assert_array_equal(out[i, :n], prefix[i, :n])
    assert not np.allclose(out[0, :1], prefix[0, :1])


def test_rtc_off_is_unchanged():
    on, off = _rtc_config(prefix_max=8), _rtc_config(prefix_max=0)
    model_off = off.create(jax.random.key(0))
    with at.disable_typechecking():
        obs = off.fake_obs(2)
    # Same parameters either way: RTC adds no params.
    model_on = on.create(jax.random.key(0))
    a = model_off.sample_actions(jax.random.key(2), obs, num_steps=2)
    b = model_on.sample_actions(jax.random.key(2), obs, num_steps=2)  # no prefix given -> plain sampling
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)
