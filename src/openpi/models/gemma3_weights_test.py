"""Numerical check of the Gemma 3 port against Hugging Face transformers (float32, CPU).

Manual: needs the gated google/gemma-3-4b-pt weights (HF_TOKEN) and ~40 GB of host RAM. Point
OPENPI_GEMMA3_CKPT at a local copy to skip the download.
"""

import os

import flax.linen as nn
import flax.traverse_util as tu
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import gemma
from openpi.models import gemma3_vision
from openpi.models import gemma3_weights

CKPT = os.environ.get("OPENPI_GEMMA3_CKPT", gemma3_weights.DEFAULT_REPO_ID)


@pytest.fixture(scope="module")
def converted():
    flat = gemma3_weights.load_gemma3_params(CKPT)
    return {k: np.asarray(v, np.float32) for k, v in flat.items()}


@pytest.fixture(scope="module")
def hf_model():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    model = transformers.Gemma3ForConditionalGeneration.from_pretrained(
        str(gemma3_weights.resolve_checkpoint_dir(CKPT)), torch_dtype=torch.float32
    )
    return model.eval()


def _subtree(flat, prefix):
    return tu.unflatten_dict({k[len(prefix) + 1 :]: v for k, v in flat.items() if k.startswith(prefix + "/")}, sep="/")


@pytest.mark.manual
def test_llm_logits_match_transformers(converted, hf_model):
    import torch

    tokens = np.array([[2, 818, 5279, 529, 7001, 563, 1080, 1458, 236761, 107, 4396, 236787]], np.int32)
    with torch.no_grad():
        ref = hf_model.language_model(input_ids=torch.from_numpy(tokens).long()).last_hidden_state.numpy()
        ref_logits = hf_model.lm_head(torch.from_numpy(ref)).numpy()

    llm = gemma.Module(configs=[gemma.get_config("gemma3_4b")], embed_dtype="float32")
    params = {"params": _subtree(converted, "PaliGemma/llm")}
    t = tokens.shape[1]
    mask = jnp.tril(jnp.ones((1, t, t), bool))
    positions = jnp.arange(t)[None]
    emb = llm.apply(params, jnp.asarray(tokens), method="embed")
    (out,), _ = llm.apply(params, [emb], positions, mask)
    logits = llm.apply(params, out, method="decode_logits")

    np.testing.assert_allclose(np.asarray(out), ref, rtol=2e-3, atol=2e-3)
    np.testing.assert_array_equal(np.argmax(np.asarray(logits), -1), np.argmax(ref_logits, -1))
    np.testing.assert_allclose(np.asarray(logits), ref_logits, rtol=2e-3, atol=2e-2)


@pytest.mark.manual
def test_image_features_match_transformers(converted, hf_model):
    import torch

    image = np.asarray(jax.random.uniform(jax.random.key(0), (1, 896, 896, 3), minval=-1, maxval=1))
    with torch.no_grad():
        ref = hf_model.get_image_features(torch.from_numpy(image.transpose(0, 3, 1, 2).copy())).numpy()

    tower = gemma3_vision.VisionTower(out_width=2560)
    params = {"params": _subtree(converted, "PaliGemma/img")}
    ours, _ = tower.apply(params, jnp.asarray(image))
    np.testing.assert_allclose(np.asarray(ours), ref, rtol=2e-3, atol=2e-3)


@pytest.mark.manual
def test_posemb_resize_is_identity_at_native_resolution(converted):
    posemb = converted["PaliGemma/img/siglip/pos_embedding"]
    np.testing.assert_array_equal(gemma3_weights.resize_posemb(posemb, posemb.shape[1]), posemb)
    assert gemma3_weights.resize_posemb(posemb, 32 * 32).shape == (1, 1024, posemb.shape[-1])


def test_vision_tower_runs_at_448():
    tower = gemma3_vision.VisionTower(out_width=64)
    image = jnp.zeros((1, 448, 448, 3))
    params = jax.eval_shape(lambda: nn.Module.init(tower, jax.random.key(0), image))
    assert params["params"]["siglip"]["pos_embedding"].shape == (1, 32 * 32, 1152)
    with pytest.raises(ValueError, match="multiple of 224"):
        gemma3_vision.check_resolution(500)


@pytest.mark.manual
def test_gemma_fast_logits_match_transformers(converted, hf_model):
    import torch

    from openpi.models import gemma_fast

    tokens = np.array([[2, 818, 5279, 529, 7001, 563, 1080, 1458]], np.int32)
    with torch.no_grad():
        hidden = hf_model.language_model(input_ids=torch.from_numpy(tokens).long()).last_hidden_state
        ref_logits = hf_model.lm_head(hidden).numpy()
    fast = gemma_fast.Module(**gemma_fast.get_config("gemma3_4b"), embed_dtype="float32")
    logits, _, _ = fast.apply({"params": _subtree(converted, "PaliGemma/llm")}, jnp.asarray(tokens))
    np.testing.assert_array_equal(np.argmax(np.asarray(logits), -1), np.argmax(ref_logits, -1))
    np.testing.assert_allclose(np.asarray(logits), ref_logits, rtol=2e-3, atol=2e-2)
