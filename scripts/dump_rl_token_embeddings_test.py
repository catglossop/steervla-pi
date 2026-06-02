"""Tests for scripts/dump_rl_token_embeddings.py, ordered cheap -> expensive.

Run subsets:
  uv run pytest scripts/dump_rl_token_embeddings_test.py -k config -s
  uv run pytest scripts/dump_rl_token_embeddings_test.py -k spec   -s
  uv run pytest scripts/dump_rl_token_embeddings_test.py -k data   -s   # needs GCS
  uv run pytest scripts/dump_rl_token_embeddings_test.py -k smoke  -s   # needs GCS + GPU + checkpoint
"""
import dataclasses
import os
import pathlib

import numpy as np
import pytest

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.training.config as _config

CONFIG_NAME = "pi05_steervla_cot_simplified_reasoning_traffic_light_only"
EXPECTED_DATASET = "simlingo_dataset_leading_object_traffic_light_img512_1116"


# ----- 1. Cheap: config registry & schema -----

def test_config_resolves():
    config = _config.get_config(CONFIG_NAME)
    assert config.name == CONFIG_NAME
    assert isinstance(config.model, pi0_config.Pi0CoTConfig)
    assert config.model.max_token_len == 200
    assert config.model.max_subtask_len == 64
    assert config.model.max_reasoning_len == 64
    assert config.model.knowledge_insulation is False
    assert config.model.use_fast_tokens is True

    data = config.data
    assert data.dataset_format.name == "SIMLINGO"
    assert data.include_ego_history is False
    assert data.action_dim == 4
    assert dict(data.dataset_name_weight_mappings) == {EXPECTED_DATASET: 1.0}
    assert data.rlds_data_dir == "gs://tian-us-central2/tensorflow_datasets"


def test_model_inputs_spec_has_cot_fields():
    # eval-shape only; no weight init. Pin GPU off to keep it cheap.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax  # noqa: E402

    config = _config.get_config(CONFIG_NAME)
    obs_spec, action_spec = config.model.inputs_spec(batch_size=2)

    # Standard fields.
    assert set(obs_spec.images) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    for k, v in obs_spec.images.items():
        assert v.shape == (2, *_model.IMAGE_RESOLUTION, 3), f"image {k}: {v.shape}"
    assert obs_spec.state.shape == (2, 32)
    assert obs_spec.tokenized_prompt.shape == (2, 200)

    # CoT-specific fields that the dumper relies on.
    assert obs_spec.tokenized_subtask is not None
    assert obs_spec.tokenized_subtask.shape == (2, 64)
    assert obs_spec.tokenized_reasoning is not None
    assert obs_spec.tokenized_reasoning.shape == (2, 64)
    # use_fast_tokens=True → tokenized_fast present.
    assert getattr(obs_spec, "tokenized_fast", None) is not None
    assert obs_spec.tokenized_fast.shape == (2, 64)

    assert action_spec.shape == (2, 10, 32)
    _ = jax  # keep import used


# ----- 2. Medium: data loader against real GCS dataset -----

@pytest.mark.skipif(
    os.environ.get("RLT_SKIP_GCS") == "1",
    reason="Set RLT_SKIP_GCS=1 to skip tests that hit gs://tian-us-central2/...",
)
def test_data_loader_yields_cot_batch():
    """Pulls a single batch from GCS and checks the dumper's expected fields/shapes."""
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax  # noqa: E402

    import openpi.training.data_loader as _data_loader  # noqa: E402
    import openpi.training.sharding as sharding  # noqa: E402

    config = dataclasses.replace(_config.get_config(CONFIG_NAME), batch_size=2, fsdp_devices=1)
    mesh = sharding.make_mesh(1)
    sh = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    loader = _data_loader.create_data_loader(
        config, sharding=sh, shuffle=False, skip_norm_stats=True, split="train"
    )
    obs, actions = next(iter(loader))

    # Action shape.
    assert tuple(actions.shape) == (2, 10, 32)

    # Vision.
    for k in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"):
        assert k in obs.images, f"missing image {k}"
        assert tuple(obs.images[k].shape) == (2, *_model.IMAGE_RESOLUTION, 3)
        assert tuple(obs.image_masks[k].shape) == (2,)
    assert bool(np.asarray(obs.image_masks["base_0_rgb"][0])) is True
    assert bool(np.asarray(obs.image_masks["left_wrist_0_rgb"][0])) is False
    assert bool(np.asarray(obs.image_masks["right_wrist_0_rgb"][0])) is False

    # Prompt / CoT / FAST.
    assert tuple(obs.tokenized_prompt.shape) == (2, 200)
    assert tuple(obs.tokenized_subtask.shape) == (2, 64)
    assert tuple(obs.tokenized_reasoning.shape) == (2, 64)
    assert tuple(obs.tokenized_fast.shape) == (2, 64)
    for mname in ("tokenized_prompt_mask", "tokenized_subtask_mask", "tokenized_reasoning_mask", "tokenized_fast_mask"):
        m = getattr(obs, mname)
        assert m is not None, mname
        # Each row should have at least one valid token.
        assert int(np.asarray(m).sum(axis=1).min()) > 0, mname


# ----- 3. Heavy: end-to-end smoke test of the dumper -----

@pytest.mark.skipif(
    os.environ.get("RLT_SKIP_SMOKE") == "1",
    reason="Set RLT_SKIP_SMOKE=1 to skip end-to-end dumper smoke test (requires GPU + GCS + checkpoint).",
)
def test_smoke_dump_two_batches(tmp_path: pathlib.Path):
    """Run dump_rl_token_embeddings.main() for 2 batches and check the output files."""
    from . import dump_rl_token_embeddings as dumper  # noqa: E402

    args = dumper.Args(
        config_name=CONFIG_NAME,
        checkpoint_dir=dumper.DEFAULT_CHECKPOINT,
        output_dir=str(tmp_path),
        num_batches=2,
        split="train",
    )
    dumper.main(args)

    shards = sorted(tmp_path.glob("shard_*.npz"))
    assert len(shards) == 2, f"expected 2 shards, got {len(shards)}"

    z = np.load(shards[0])
    bsz = 8  # config.batch_size for traffic_light_only
    per_cam = int(z["tokens_per_cam"])
    n_prompt = int(z["n_prompt"])
    n_reasoning = int(z["n_reasoning"])
    n_subtask = int(z["n_subtask"])
    n_fast = int(z["n_fast"])
    expected_seq = per_cam + n_prompt + n_reasoning + n_subtask + n_fast

    assert z["prefix_out"].shape == (bsz, expected_seq, 2048), z["prefix_out"].shape
    assert z["prefix_out"].dtype == np.float16
    assert z["prefix_mask"].shape == (bsz, expected_seq)
    assert z["prefix_mask"].dtype == np.bool_
    assert z["actions"].shape == (bsz, 10, 32)
    assert z["actions"].dtype == np.float32

    # Sanity: non-trivial activations, no NaNs.
    out = z["prefix_out"].astype(np.float32)
    assert np.isfinite(out).all(), "prefix_out contains NaN/inf"
    assert float(out.std()) > 1e-3, f"prefix_out std={out.std()}; likely all-zero"

    # Mask density plausible: vision is fully unmasked (per_cam tokens, always True)
    # plus variable text tokens. Density should sit roughly in [per_cam/total, 1.0].
    density = float(z["prefix_mask"].mean())
    assert per_cam / expected_seq - 0.01 <= density <= 1.0, density

    # Vision positions must be fully unmasked.
    assert bool(z["prefix_mask"][:, :per_cam].all()), "vision tokens unexpectedly masked"

    print(
        f"OK: per_cam={per_cam} n_prompt={n_prompt} n_reasoning={n_reasoning} "
        f"n_subtask={n_subtask} n_fast={n_fast} total_seq={expected_seq} "
        f"density={density:.3f} std={out.std():.4f}"
    )
