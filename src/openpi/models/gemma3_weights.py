"""Load Hugging Face Gemma 3 (``Gemma3ForConditionalGeneration``) weights into openpi's parameter layout.

Produces flat ``{"PaliGemma/llm/...": array, "PaliGemma/img/...": array}`` dicts matching
``gemma.Module`` (Gemma 3 configs) and ``gemma3_vision.VisionTower``. Only the VLM is converted; the action
expert has no pretrained counterpart and keeps its fresh initialization.
"""

from collections.abc import Callable
import json
import logging
import pathlib

import jax
import ml_dtypes
import numpy as np

logger = logging.getLogger("openpi")

DEFAULT_REPO_ID = "google/gemma-3-4b-pt"


def resolve_checkpoint_dir(repo_or_dir: str) -> pathlib.Path:
    """A local directory as-is; otherwise a Hugging Face repo id, downloaded (gated: needs HF_TOKEN)."""
    path = pathlib.Path(repo_or_dir).expanduser()
    if path.is_dir():
        return path
    import huggingface_hub

    return pathlib.Path(
        huggingface_hub.snapshot_download(repo_or_dir, allow_patterns=["*.safetensors", "*.json", "tokenizer.model"])
    )


def _iter_tensors(ckpt_dir: pathlib.Path):
    """Yield (name, float32 numpy array) for every tensor in the sharded safetensors checkpoint."""
    import safetensors
    import torch

    index = json.loads((ckpt_dir / "model.safetensors.index.json").read_text())["weight_map"]
    for shard in sorted(set(index.values())):
        with safetensors.safe_open(ckpt_dir / shard, framework="pt") as f:
            for name in f.keys():  # noqa: SIM118 -- safe_open is not a dict
                yield name, f.get_tensor(name).to(torch.float32).numpy()


def _bf16(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x).astype(ml_dtypes.bfloat16)


def _llm_layer_params(get: Callable[[str], np.ndarray], i: int, *, num_heads: int, num_kv_heads: int, head_dim: int):
    p = f"language_model.model.layers.{i}."
    q = get(p + "self_attn.q_proj.weight")  # (H*D, W)
    k = get(p + "self_attn.k_proj.weight")  # (K*D, W)
    v = get(p + "self_attn.v_proj.weight")
    o = get(p + "self_attn.o_proj.weight")  # (W, H*D)
    width = q.shape[1]

    def heads(w, n):  # (n*D, W) -> (n, W, D)
        return w.reshape(n, head_dim, width).transpose(0, 2, 1)

    return {
        "attn/q_einsum/w": heads(q, num_heads),
        "attn/kv_einsum/w": np.stack([heads(k, num_kv_heads), heads(v, num_kv_heads)]),
        "attn/attn_vec_einsum/w": o.T.reshape(num_heads, head_dim, width),
        "attn/q_norm/scale": get(p + "self_attn.q_norm.weight"),
        "attn/k_norm/scale": get(p + "self_attn.k_norm.weight"),
        "pre_attention_norm/scale": get(p + "input_layernorm.weight"),
        "post_attention_norm/scale": get(p + "post_attention_layernorm.weight"),
        "pre_ffw_norm/scale": get(p + "pre_feedforward_layernorm.weight"),
        "post_ffw_norm/scale": get(p + "post_feedforward_layernorm.weight"),
        "mlp/gating_einsum": np.stack([get(p + "mlp.gate_proj.weight").T, get(p + "mlp.up_proj.weight").T]),
        "mlp/linear": get(p + "mlp.down_proj.weight").T,
    }


def _vision_layer_params(get: Callable[[str], np.ndarray], i: int, *, num_heads: int):
    p = f"vision_tower.vision_model.encoder.layers.{i}."
    width = get(p + "self_attn.q_proj.weight").shape[0]
    head_dim = width // num_heads

    def qkv(name):
        return {
            "kernel": get(p + f"self_attn.{name}_proj.weight").T.reshape(width, num_heads, head_dim),
            "bias": get(p + f"self_attn.{name}_proj.bias").reshape(num_heads, head_dim),
        }

    out = {}
    for hf, ours in (("q", "query"), ("k", "key"), ("v", "value")):
        for leaf, value in qkv(hf).items():
            out[f"MultiHeadDotProductAttention_0/{ours}/{leaf}"] = value
    out["MultiHeadDotProductAttention_0/out/kernel"] = get(p + "self_attn.out_proj.weight").T.reshape(
        num_heads, head_dim, width
    )
    out["MultiHeadDotProductAttention_0/out/bias"] = get(p + "self_attn.out_proj.bias")
    out["LayerNorm_0/scale"] = get(p + "layer_norm1.weight")
    out["LayerNorm_0/bias"] = get(p + "layer_norm1.bias")
    out["LayerNorm_1/scale"] = get(p + "layer_norm2.weight")
    out["LayerNorm_1/bias"] = get(p + "layer_norm2.bias")
    out["MlpBlock_0/Dense_0/kernel"] = get(p + "mlp.fc1.weight").T
    out["MlpBlock_0/Dense_0/bias"] = get(p + "mlp.fc1.bias")
    out["MlpBlock_0/Dense_1/kernel"] = get(p + "mlp.fc2.weight").T
    out["MlpBlock_0/Dense_1/bias"] = get(p + "mlp.fc2.bias")
    return out


def _stack_layers(per_layer: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """openpi scans over layers, so each per-layer parameter is stacked along a leading depth axis."""
    return {k: np.stack([layer[k] for layer in per_layer]) for k in per_layer[0]}


def resize_posemb(posemb: np.ndarray, num_tokens: int) -> np.ndarray:
    """Bilinearly resize a learned (1, side*side, C) position embedding to (1, num_tokens, C)."""
    _, length, c = posemb.shape
    if length == num_tokens:
        return posemb
    old, new = round(length**0.5), round(num_tokens**0.5)
    grid = posemb.reshape(old, old, c).astype(np.float32)
    resized = jax.image.resize(grid, (new, new, c), method="bilinear")
    return np.asarray(resized).reshape(1, new * new, c)


def load_gemma3_params(
    repo_or_dir: str = DEFAULT_REPO_ID,
    *,
    num_heads: int = 8,
    num_kv_heads: int = 4,
    head_dim: int = 256,
    vision_heads: int = 16,
    llm_prefix: str = "PaliGemma/llm",
    img_prefix: str = "PaliGemma/img",
    include_vision: bool = True,
) -> dict[str, np.ndarray]:
    """Convert a Gemma 3 checkpoint into flat openpi params (bfloat16 numpy arrays)."""
    ckpt_dir = resolve_checkpoint_dir(repo_or_dir)
    logger.info("Loading Gemma 3 weights from %s", ckpt_dir)
    hf = dict(_iter_tensors(ckpt_dir))
    get = hf.__getitem__

    depth = 1 + max(int(k.split(".")[3]) for k in hf if k.startswith("language_model.model.layers."))
    out = {
        f"{llm_prefix}/embedder/input_embedding": get("language_model.model.embed_tokens.weight"),
        f"{llm_prefix}/final_norm/scale": get("language_model.model.norm.weight"),
    }
    layers = _stack_layers(
        [
            _llm_layer_params(get, i, num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim)
            for i in range(depth)
        ]
    )
    out.update({f"{llm_prefix}/layers/{k}": v for k, v in layers.items()})

    if include_vision:
        v = "vision_tower.vision_model."
        vdepth = 1 + max(int(k.split(".")[4]) for k in hf if k.startswith(v + "encoder.layers."))
        vis = {
            "siglip/embedding/kernel": get(v + "embeddings.patch_embedding.weight").transpose(2, 3, 1, 0),
            "siglip/embedding/bias": get(v + "embeddings.patch_embedding.bias"),
            "siglip/pos_embedding": get(v + "embeddings.position_embedding.weight")[None],
            "siglip/Transformer/encoder_norm/scale": get(v + "post_layernorm.weight"),
            "siglip/Transformer/encoder_norm/bias": get(v + "post_layernorm.bias"),
            "mm_soft_emb_norm/scale": get("multi_modal_projector.mm_soft_emb_norm.weight"),
            "mm_input_projection": get("multi_modal_projector.mm_input_projection_weight"),
        }
        vlayers = _stack_layers([_vision_layer_params(get, i, num_heads=vision_heads) for i in range(vdepth)])
        vis.update({f"siglip/Transformer/encoderblock/{k}": val for k, val in vlayers.items()})
        out.update({f"{img_prefix}/{k}": val for k, val in vis.items()})

    hf.clear()
    return {k: _bf16(val) for k, val in out.items()}
