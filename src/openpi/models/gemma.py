# Copyright 2024 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gemma adaptation for Pi, taken from big_vision.

We follow this einsum axis naming convention:
  B: batch
  T: query length
  S: k/v length
  N: num query heads
  K: num k/v heads
  G: num query heads per k/v head
  H: head dim
  D: d_model ("features")
"""

from collections.abc import Sequence
import dataclasses
from typing import Literal, TypeAlias

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp

import openpi.models.lora as lora
import openpi.shared.array_typing as at
import openpi.training.sharding as sharding

PALIGEMMA_VOCAB_SIZE = 257_152

# Chunk size for greedy vocab projection in :meth:`Embedder.decode_argmax_chunked`.
# Avoids materializing a full ``(..., PALIGEMMA_VOCAB_SIZE)`` logits tensor (saves peak VRAM).
# Smaller chunks lower peak (logits + cuBLAS workspace) at the cost of more matmul steps.
_DECODE_ARGMAX_VOCAB_CHUNK = 4096


@dataclasses.dataclass
class Config:
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    lora_configs: dict[str, lora.LoRAConfig] = dataclasses.field(default_factory=dict)
    # Gemma 3 options. The defaults reproduce the Gemma (PaliGemma) architecture exactly. Attention-wide settings
    # (RoPE, sliding window, query scale) are read from the first expert's config, since all experts attend jointly.
    vocab_size: int = PALIGEMMA_VOCAB_SIZE
    qk_norm: bool = False  # RMSNorm on per-head queries and keys before RoPE.
    post_norms: bool = False  # Extra RMSNorm on the attention / MLP outputs before the residual.
    # Every `sliding_window_pattern`-th layer is global; the rest are local. None = all layers global.
    sliding_window_pattern: int | None = None
    sliding_window: int | None = None
    rope_local_base: float = 10_000.0
    rope_global_base: float = 10_000.0
    rope_global_scale: float = 1.0  # Linear RoPE scaling on global layers (positions are divided by this).
    query_pre_attn_scalar: float | None = None  # None = head_dim.


Variant = Literal[
    "dummy",
    "gemma_300m",
    "gemma_300m_lora",
    "gemma_2b",
    "gemma_2b_lora",
    "gemma3_4b",
    "gemma3_expert_1024",
    "dummy_gemma3",
]

GEMMA3_VOCAB_SIZE = 262_208  # Embedding rows in the gemma-3-4b checkpoint (262_144 text tokens + image/padding).


def _gemma3(width: int, depth: int, mlp_dim: int, head_dim: int, *, vocab_size: int = GEMMA3_VOCAB_SIZE) -> Config:
    """Gemma 3 4B attention geometry (8 heads, 4 KV heads) and layer style; width/depth/mlp set per variant."""
    return Config(
        width=width,
        depth=depth,
        mlp_dim=mlp_dim,
        num_heads=8,
        num_kv_heads=4,
        head_dim=head_dim,
        vocab_size=vocab_size,
        qk_norm=True,
        post_norms=True,
        sliding_window_pattern=6,
        sliding_window=1024,
        rope_local_base=10_000.0,
        rope_global_base=1_000_000.0,
        rope_global_scale=8.0,
        query_pre_attn_scalar=float(head_dim),
    )


def get_config(variant: Variant) -> Config:
    """Returns config for specified gemma variant."""
    if variant == "dummy":
        return Config(
            width=64,
            depth=4,
            mlp_dim=128,
            num_heads=8,
            num_kv_heads=1,
            head_dim=16,
        )
    if variant == "gemma3_4b":
        return _gemma3(width=2560, depth=34, mlp_dim=10_240, head_dim=256)
    if variant == "gemma3_expert_1024":
        # Action expert matching gemma3_4b's attention geometry (depth 34, 8 heads, 4 KV heads, head_dim 256).
        return _gemma3(width=1024, depth=34, mlp_dim=4096, head_dim=256)
    if variant == "dummy_gemma3":
        # Tiny Gemma 3-style model for tests; depth 6 so one layer is global.
        return _gemma3(width=64, depth=6, mlp_dim=128, head_dim=16, vocab_size=GEMMA3_VOCAB_SIZE)
    if variant == "gemma_300m":
        # 311M params
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_2b":
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_2b_lora":
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={"attn": lora.LoRAConfig(rank=16, alpha=16.0), "ffn": lora.LoRAConfig(rank=16, alpha=16.0)},
        )
    if variant == "gemma_300m_lora":
        # 311M params
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={"attn": lora.LoRAConfig(rank=32, alpha=32.0), "ffn": lora.LoRAConfig(rank=32, alpha=32.0)},
        )
    raise ValueError(f"Unknown variant: {variant}")


@at.typecheck
class RMSNorm(nn.Module):
    @nn.compact
    def __call__(self, x, cond):
        dtype = x.dtype  # original dtype, could be half-precision
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)  # compute variance in float32
        normed_inputs = jnp.asarray(x * jnp.reciprocal(jnp.sqrt(var + 1e-06)))  # compute normalization in float32
        if cond is None:
            # regular RMSNorm
            scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1]))
            normed_inputs = normed_inputs * (
                1 + scale
            )  # scale by learned parameter in float32 (matches Flax implementation)
            return normed_inputs.astype(dtype), None  # return in original dtype

        # adaptive RMSNorm
        modulation = nn.Dense(x.shape[-1] * 3, kernel_init=nn.initializers.zeros, dtype=dtype)(cond)
        scale, shift, gate = jnp.split(modulation[:, None, :], 3, axis=-1)
        normed_inputs = normed_inputs * (1 + scale) + shift  # scale and shift in float32
        return normed_inputs.astype(dtype), gate


@at.typecheck
class Embedder(nn.Module):
    """Embedder module."""

    vocab_size: int
    embed_dim: int

    def setup(self):
        self.input_embedding_table = self.param(
            "input_embedding",
            nn.initializers.normal(),
            (self.vocab_size, self.embed_dim),
        )

    def encode(self, x):
        x = self.input_embedding_table[(x,)]
        x *= jnp.sqrt(self.embed_dim).astype(x.dtype)
        return x

    def decode(self, x):
        return jnp.dot(x, self.input_embedding_table.T)

    def decode_argmax_chunked(self, x):
        """Argmax token ids over the vocabulary without a full ``(vocab,)`` logits vector.

        ``decode`` uses ``jnp.dot(x, table.T)``, which forces a large output and often a large
        cuBLAS workspace (~hundreds of MiB). For greedy decoding (``temperature == 0``), only
        the winning index matters; this path reduces peak memory at the cost of extra matmuls.
        """
        table = self.input_embedding_table
        v = int(self.vocab_size)
        chunk = _DECODE_ARGMAX_VOCAB_CHUNK
        orig_shape = x.shape
        d = orig_shape[-1]
        x_flat = jnp.reshape(x, (-1, d))
        n = x_flat.shape[0]
        best_score = jnp.full((n,), -jnp.inf, dtype=jnp.float32)
        best_idx = jnp.zeros((n,), dtype=jnp.int32)
        num_chunks = (v + chunk - 1) // chunk
        for i in range(num_chunks):
            start = i * chunk
            end = min(start + chunk, v)
            subt = table[start:end]
            # Keep matmul dtype (usually bf16); avoid a full float32 (n, chunk) logits tensor.
            logits = jnp.dot(x_flat, subt.T)
            local_max = jnp.max(logits, axis=-1).astype(jnp.float32)
            local_idx = jnp.argmax(logits, axis=-1).astype(jnp.int32) + start
            better = local_max > best_score
            best_score = jnp.where(better, local_max, best_score)
            best_idx = jnp.where(better, local_idx, best_idx)
        return jnp.reshape(best_idx, orig_shape[:-1])


def _attend(q, k, v, attn_mask, num_kv_heads, dtype):
    q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=num_kv_heads)
    logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32)

    if attn_mask.shape != (q.shape[0], 1, q.shape[1], k.shape[1]):
        raise ValueError(
            f"Attention mask with shape {attn_mask.shape} but shapes for q and k are: {q.shape} and {k.shape}"
        )

    # big_neg = jnp.finfo(logits.dtype).min
    big_neg = -2.3819763e38  # See gemma/modules.py
    masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, big_neg)

    probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)

    encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
    return einops.rearrange(encoded, "B T K G H -> B T (K G) H")


@at.typecheck
class Attention(nn.Module):
    """Attention module."""

    configs: Sequence[Config]
    knowledge_insulation: bool = False

    @nn.compact
    def __call__(self, xs, positions, attn_mask, kv_cache, layer_idx=None):
        # all experts must share the same head dim, num heads, and num kv heads for self-attention to work
        assert all(config.head_dim == self.configs[0].head_dim for config in self.configs)
        assert all(config.num_heads == self.configs[0].num_heads for config in self.configs)
        assert all(config.num_kv_heads == self.configs[0].num_kv_heads for config in self.configs)

        dtype = next(x.dtype for x in xs if x is not None)  # original dtype, could be half-precision

        qkvs = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is None:
                continue
            if config.num_kv_heads == config.num_heads:
                qkv_einsum = lora.Einsum(
                    shape=(3, config.num_heads, config.width, config.head_dim),
                    name=_name("qkv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=config.lora_configs.get("attn"),
                )
                qkvs.append(qkv_einsum("BSD,3KDH->3BSKH", x))
            else:
                q_einsum = lora.Einsum(
                    shape=(config.num_heads, config.width, config.head_dim),
                    name=_name("q_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
                    lora_config=config.lora_configs.get("attn"),
                )
                q = q_einsum("BTD,NDH->BTNH", x)
                kv_einsum = lora.Einsum(
                    shape=(2, config.num_kv_heads, config.width, config.head_dim),
                    name=_name("kv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=config.lora_configs.get("attn"),
                )
                k, v = kv_einsum("BSD,2KDH->2BSKH", x)
                qkvs.append((q, k, v))
            if config.qk_norm:
                q, k, v = qkvs[-1]
                q = RMSNorm(name=_name("q_norm", i))(q, None)[0]
                k = RMSNorm(name=_name("k_norm", i))(k, None)[0]
                qkvs[-1] = (q, k, v)

        q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs, strict=True))

        cfg = self.configs[0]
        if cfg.sliding_window_pattern is None:
            q = _apply_rope(q, positions=positions)
            k = _apply_rope(k, positions=positions)
        else:
            # Gemma 3: global layers use a larger RoPE base with linear position scaling; local layers do not.
            is_global = (layer_idx + 1) % cfg.sliding_window_pattern == 0
            base = jnp.where(is_global, cfg.rope_global_base, cfg.rope_local_base)
            rope_positions = positions / jnp.where(is_global, cfg.rope_global_scale, 1.0)
            q = _apply_rope(q, positions=rope_positions, max_wavelength=base)
            k = _apply_rope(k, positions=rope_positions, max_wavelength=base)
        q *= (cfg.query_pre_attn_scalar or cfg.head_dim) ** -0.5

        # should still be half-precision here (if input was half-precision)
        assert q.dtype == k.dtype == v.dtype == dtype

        if kv_cache is not None and len(kv_cache) == 3:
            # Preallocated cache (Pi0CoT autoregressive decoding): write the new k/v at `cache_idx`.
            cache_idx, cache_k, cache_v = kv_cache
            update_idx = jnp.asarray(cache_idx, dtype=jnp.int32)
            update_len = k.shape[1]
            k = jax.lax.dynamic_update_slice(cache_k, k.astype(cache_k.dtype), (0, update_idx, 0, 0))
            v = jax.lax.dynamic_update_slice(cache_v, v.astype(cache_v.dtype), (0, update_idx, 0, 0))
            new_kv_cache = (cache_idx + update_len, k, v)
        else:
            if kv_cache is not None:
                cache_k, cache_v = kv_cache
                k = jnp.concatenate([cache_k, k], axis=1)
                v = jnp.concatenate([cache_v, v], axis=1)
            new_kv_cache = (k, v)

        num_kv_heads = self.configs[0].num_kv_heads
        experts_present = [x.shape[1] for x in xs if x is not None]
        if self.knowledge_insulation and kv_cache is None and len(experts_present) > 1:
            # Knowledge insulation (Driess et al., 2025): the action expert reads the backbone's K/V
            # through stop_gradient, so the flow-matching loss cannot update the VLM backbone via
            # attention. Softmax is per query row, so attending backbone and action-expert queries
            # separately is exact. Training only (both experts present, no KV cache).
            n = experts_present[0]
            k_sg = jnp.concatenate([jax.lax.stop_gradient(k[:, :n]), k[:, n:]], axis=1)
            v_sg = jnp.concatenate([jax.lax.stop_gradient(v[:, :n]), v[:, n:]], axis=1)
            encoded = jnp.concatenate(
                [
                    _attend(q[:, :n], k, v, attn_mask[:, :, :n], num_kv_heads, dtype),
                    _attend(q[:, n:], k_sg, v_sg, attn_mask[:, :, n:], num_kv_heads, dtype),
                ],
                axis=1,
            )
        else:
            encoded = _attend(q, k, v, attn_mask, num_kv_heads, dtype)

        out = []
        start = 0
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                end = start + x.shape[1]
                out_einsum = lora.Einsum(
                    shape=(config.num_heads, config.head_dim, config.width),
                    name=_name("attn_vec_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
                    lora_config=config.lora_configs.get("attn"),
                )
                out.append(out_einsum("BTNH,NHD->BTD", encoded[:, start:end]))
                start = end
            else:
                out.append(None)

        return out, new_kv_cache


@at.typecheck
class FeedForward(nn.Module):
    """Feed forward module."""

    features: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        dtype = x.dtype  # original dtype, could be half-precision
        w_gating = self.param(
            "gating_einsum",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
            (2, self.features, self.hidden_dim),
        ).astype(dtype)
        ff_gate = jnp.dot(x, w_gating[0])
        gate_value = nn.gelu(ff_gate)

        ff1 = jnp.dot(x, w_gating[1])
        activations = gate_value * ff1

        w_linear = self.param(
            "linear",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
            (self.hidden_dim, self.features),
        ).astype(dtype)
        outputs = jnp.dot(activations, w_linear)
        assert outputs.dtype == dtype
        return outputs


@at.typecheck
class Block(nn.Module):
    """Transformer block."""

    configs: tuple[Config, ...]

    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()
    knowledge_insulation: bool = False

    @nn.compact
    def __call__(self, xs, kv_cache, positions, attn_mask, adarms_cond, deterministic=True, layer_idx=None):  # noqa: FBT002
        xs = sharding.activation_sharding_constraint(xs)
        drop = nn.Dropout(self.dropout, self.dropout_bdims) if self.dropout else lambda x, _: x

        attn = Attention(configs=self.configs, knowledge_insulation=self.knowledge_insulation, name="attn")

        pre_attn = []
        gates = []
        for i, x in enumerate(xs):
            if x is not None:
                x, gate = RMSNorm(name=_name("pre_attention_norm", i))(x, adarms_cond[i])  # noqa: PLW2901
            pre_attn.append(x)
            gates.append(gate if x is not None else None)

        pre_attn = sharding.activation_sharding_constraint(pre_attn)
        post_attn, kv_cache = attn(pre_attn, positions, attn_mask, kv_cache, layer_idx)
        post_attn = [
            RMSNorm(name=_name("post_attention_norm", i))(y, None)[0] if y is not None and config.post_norms else y
            for i, (y, config) in enumerate(zip(post_attn, self.configs, strict=True))
        ]
        post_attn = jax.tree.map(lambda x: drop(x, deterministic), post_attn)
        post_attn = sharding.activation_sharding_constraint(post_attn)
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, post_attn, gates, strict=True)]
        xs = sharding.activation_sharding_constraint(xs)

        out = []
        gates = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                x, gate = RMSNorm(name=_name("pre_ffw_norm", i))(x, adarms_cond[i])  # noqa: PLW2901
                x = lora.FeedForward(  # noqa: PLW2901
                    features=config.width,
                    hidden_dim=config.mlp_dim,
                    name=_name("mlp", i),
                    lora_config=config.lora_configs.get("ffn"),
                )(x)
                if config.post_norms:
                    x = RMSNorm(name=_name("post_ffw_norm", i))(x, None)[0]  # noqa: PLW2901
            out.append(x)
            gates.append(gate if x is not None else None)

        out = sharding.activation_sharding_constraint(out)
        out = jax.tree.map(lambda x: drop(x, deterministic), out)
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, out, gates, strict=True)]
        xs = sharding.activation_sharding_constraint(xs)

        return xs, kv_cache


KVCache: TypeAlias = (
    tuple[at.Float[at.Array, "l b _t _k _h"], at.Float[at.Array, "l b _t _v _h"]]
    | tuple[
        at.Int[at.Array, " l"],
        at.Float[at.Array, "l b _t _k _h"],
        at.Float[at.Array, "l b _t _v _h"],
    ]
)


@at.typecheck
class Module(nn.Module):
    """Transformer model, supporting a mixture of different weights for different tokens."""

    configs: Sequence[Config]  # list of configs, one for each expert
    embed_dtype: str

    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()  # Every float is dropped independently.
    adarms: bool = False
    knowledge_insulation: bool = False

    def setup(self):
        # all experts must have the same depth
        assert all(config.depth == self.configs[0].depth for config in self.configs)

        self.embedder = Embedder(
            vocab_size=self.configs[0].vocab_size,
            embed_dim=self.configs[0].width,  # embedder for first expert only
            name="embedder",
        )
        block_cls = nn.remat(
            Block,
            prevent_cse=False,
            static_argnums=(5,),  # 0=self, 6=deterministic
            policy=jax.checkpoint_policies.nothing_saveable,
        )
        # Gemma 3 layers differ by position in the stack (local vs global attention), so the layer index is scanned in.
        layer_idx_axes = (0,) if self.configs[0].sliding_window_pattern is not None else ()
        self.layers = nn.scan(
            block_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(
                0,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                *layer_idx_axes,
            ),  # 0=kv_cache, 1=positions, 2=mask, 3=adarms_cond, 4=deterministic, (5=layer_idx)
            length=self.configs[0].depth,
        )(
            configs=self.configs,
            dropout=self.dropout,
            dropout_bdims=self.dropout_bdims,
            knowledge_insulation=self.knowledge_insulation,
        )
        self.final_norms = [RMSNorm(name=_name("final_norm", i)) for i in range(len(self.configs))]

    @at.typecheck
    def embed(self, tokens: at.Int[at.Array, "b t"]) -> at.Float[at.Array, "b t d"]:
        return self.embedder.encode(tokens).astype(self.embed_dtype)

    @at.typecheck
    def decode_logits(self, tokens: at.Float[at.Array, "b t d"]) -> at.Float[at.Array, "b t v"]:
        """Vocabulary logits from hidden states (for CE training). `decode` returns argmax ids only."""
        return self.embedder.decode(tokens).astype(jnp.float32)

    @at.typecheck
    def decode_argmax_chunked(self, tokens: at.Float[at.Array, "b t d"]) -> at.Int[at.Array, "b t"]:
        """Greedy ids via chunked projection; see :meth:`Embedder.decode_argmax_chunked`."""
        return self.embedder.decode_argmax_chunked(tokens).astype(jnp.int32)

    @at.typecheck
    def decode(self, tokens: at.Float[at.Array, "b t d"]) -> at.Int[at.Array, "b t"]:
        logits = self.decode_logits(tokens)
        tokens = jnp.argmax(logits, axis=-1)
        return tokens.astype(jnp.int32)

    @at.typecheck
    def __call__(
        self,
        # list of token arrays, one for each expert, or None if that expert should not be run
        embedded: Sequence[at.Float[at.Array, "b _t _d"] | None],
        positions: at.Int[at.Array, "b t"],
        mask: at.Bool[at.Array, "b t s"],
        adarms_cond: Sequence[at.Float[at.Array, "b _d"] | None] | None = None,
        *,
        kv_cache: KVCache | None = None,
        deterministic: bool = True,
    ) -> tuple[Sequence[at.Float[at.Array, "b _t _d"] | None], KVCache]:
        embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype), embedded)
        mask = jnp.asarray(mask)[:, None, :, :]
        if adarms_cond is None:
            adarms_cond = [None] * len(self.configs)

        cfg = self.configs[0]
        if cfg.sliding_window_pattern is None:
            embedded, kv_cache = self.layers(embedded, kv_cache, positions, mask, adarms_cond, deterministic)
        else:
            if cfg.sliding_window is not None and mask.shape[-1] > cfg.sliding_window:
                # Within the window, Gemma 3's local layers attend exactly like global ones, so no band mask is
                # needed. Longer sequences would need it; fail loudly rather than silently diverge.
                raise NotImplementedError(
                    f"Sequence length {mask.shape[-1]} exceeds the Gemma 3 sliding window ({cfg.sliding_window})."
                )
            layer_idx = jnp.arange(cfg.depth, dtype=jnp.int32)
            embedded, kv_cache = self.layers(embedded, kv_cache, positions, mask, adarms_cond, deterministic, layer_idx)

        assert all(e.dtype == jnp.dtype(self.embed_dtype) for e in embedded if e is not None)

        return [
            f(e, a)[0] if e is not None else e for f, e, a in zip(self.final_norms, embedded, adarms_cond, strict=True)
        ], kv_cache

    def init(self, use_adarms: Sequence[bool]):
        """Convenience method for initializing all parameters, necessary due to the quirks of linen."""
        self.embed(jnp.zeros((1, 1), dtype=jnp.int32))
        self(
            [jnp.zeros((1, 1, c.width)) for c in self.configs],
            jnp.zeros((1, len(self.configs)), dtype=jnp.int32),
            jnp.zeros((1, len(self.configs), len(self.configs)), dtype=bool),
            adarms_cond=[jnp.zeros((1, c.width)) if u else None for u, c in zip(use_adarms, self.configs, strict=True)],
        )


def _apply_rope(x, *, positions, max_wavelength=10_000):
    """Applies RoPE positions [B, L] to x [B, L, H, D]."""
    freq_exponents = (2.0 / x.shape[-1]) * jnp.arange(x.shape[-1] // 2, dtype=jnp.float32)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None] / timescale[None, None, :]
    radians = radians[..., None, :]
    assert radians.dtype == jnp.float32
    # radians.shape = [...,L,1,d=D/2]
    sin, cos = jnp.sin(radians), jnp.cos(radians)
    x1, x2 = jnp.split(x, 2, axis=-1)
    res = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    assert res.dtype == jnp.float32
    # The original bigvision impl allows RoPE to upcast to float32. It is then immediately downcast again to the cache
    # dtype when in inference mode (but not in training mode). I don't think any of this was intentional. Based on the
    # original DeepMind impl, as well as the widely-used transformers impl, it is ok to always downcast back to bfloat16
    # here.
    return res.astype(x.dtype)


def _name(name, i):
    # we name layers like this because we want the first expert's weights to have no suffix (e.g., "attn"), so that they
    # can be loaded seamlessly from the existing PaliGemma checkpoint. subsequent experts will have a suffix (e.g.,
    # "attn_1") and their weights will be initialized from scratch. in practice, we only use two experts -- PaliGemma,
    # and the action expert.
    if i == 0:
        return name
    return f"{name}_{i}"


def _gated_residual(x, y, gate):
    assert (x is None) == (y is None)
    if x is None:
        return None
    if gate is None:
        return x + y
    return x + y * gate
