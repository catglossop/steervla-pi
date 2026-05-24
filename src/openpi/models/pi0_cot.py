"""Pi0.5 with Chain-of-Thought: reasoning + subtask generation before action decoding.

Prefix layout (language, after image tokens; see ``CoTPaligemmaTokenizer``):

    [Prompt:...;State:...;]                                           <-- bidirectional
    [<start_of_reasoning> reasoning_body <end_of_reasoning>]          <-- causal
    [<start_of_subtask>   subtask_body   <end_of_subtask> <eos>]      <-- causal
    [Action: FAST tokens |]  (optional)                               <-- causal

Each segment owns its start/end delimiters: ``tokenize_reasoning`` and
``tokenize_subtask`` in ``CoTPaligemmaTokenizer`` prepend ``<start_of_*>`` and
append ``<end_of_*>`` themselves. The bidirectional prompt segment has no CoT
delimiters of its own. FAST action tokens use ``CoTPaligemmaTokenizer.tokenize_fast_actions``.

Supervision (see ``_compute_loss_and_metrics``):
    - ``first_reasoning_ce`` predicts ``<start_of_reasoning>`` from the last
      valid image+prompt hidden.
    - ``reasoning_ce`` (shifted CE) predicts reasoning body and
      ``<end_of_reasoning>``.
    - ``first_subtask_ce`` predicts ``<start_of_subtask>`` from the hidden at
      ``<end_of_reasoning>``.
    - ``subtask_ce`` (shifted CE) predicts subtask body, ``<end_of_subtask>``
      and the trailing EOS.
    - ``fast_ce`` / ``first_fast_ce`` (optional) supervise FAST action tokens.

Suffix (Action expert, adaRMS):
    [action tokens] — attend to images + prompt + subtask + FAST (not reasoning).

Losses: CE on reasoning + subtask (+ FAST when enabled), flow-matching on actions.
"""

import functools
import logging
from collections.abc import Sequence

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.models.tokenizer import COT_DELIMITER_TOKEN_SLOTS, PALIGEMMA_VOCAB_SKIP_TOKENS
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

logger = logging.getLogger("openpi")

# Must match ``CoTPaligemmaTokenizer`` reserved-token layout.
START_OF_SUBTASK_ID = _gemma.PALIGEMMA_VOCAB_SIZE - 1 - PALIGEMMA_VOCAB_SKIP_TOKENS - 1
END_OF_SUBTASK_ID = _gemma.PALIGEMMA_VOCAB_SIZE - 1 - PALIGEMMA_VOCAB_SKIP_TOKENS - 2
START_OF_REASONING_ID = _gemma.PALIGEMMA_VOCAB_SIZE - 1 - PALIGEMMA_VOCAB_SKIP_TOKENS - 3
END_OF_REASONING_ID = _gemma.PALIGEMMA_VOCAB_SIZE - 1 - PALIGEMMA_VOCAB_SKIP_TOKENS - 4

_COT_MODULE_JIT_CACHE: dict[int, dict[str, object]] = {}


@functools.lru_cache(maxsize=1)
def _fast_segment_layout_token_ids() -> tuple[tuple[int, ...], int, int]:
    """``Action:`` prefix, ``|`` delimiter, and EOS ids for FAST segment generation."""
    import sentencepiece

    from openpi.shared import download

    path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
    with path.open("rb") as f:
        sp = sentencepiece.SentencePieceProcessor(model_proto=f.read())
    return tuple(int(x) for x in sp.encode("Action: ")), int(sp.encode("|")[0]), int(sp.eos_id())


class Pi0CoT(_model.BaseModel):
    """Pi0.5 model extended with chain-of-thought reasoning and subtask generation."""

    def __init__(self, config: "pi0_config.Pi0CoTConfig", rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.cot_loss_weight = config.cot_loss_weight

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=True,
                knowledge_insulation=config.knowledge_insulation,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)

        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        self.max_subtask_len = config.max_subtask_len
        self.max_reasoning_len = config.max_reasoning_len
        self.max_fast_len = config.max_fast_len
        self._use_fast_tokens = config.use_fast_tokens
        self._preprocess_image_keys: tuple[str, ...] = (
            tuple(config.inference_image_keys)
            if config.inference_image_keys is not None
            else tuple(_model.IMAGE_KEYS)
        )
        self._cot_jit_decode = bool(config.cot_jit_decode)
        self._cot_jit_transformer_forward = bool(config.cot_jit_transformer_forward)
        self._cot_replay_reasoning = bool(config.cot_replay_reasoning)
        self.deterministic = True

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _embed_images(self, obs: _model.Observation):
        """Embed images, returning (tokens, mask, ar_mask_list)."""
        tokens, masks, ar = [], [], []
        for name in obs.images:
            img_tok, _ = self.PaliGemma.img(obs.images[name], train=False)
            tokens.append(img_tok)
            masks.append(einops.repeat(obs.image_masks[name], "b -> b s", s=img_tok.shape[1]))
            ar += [False] * img_tok.shape[1]
        return tokens, masks, ar

    def _embed_text_tokens(self, token_ids):
        """Embed integer token IDs through the PaliGemma embedder."""
        return self.PaliGemma.llm(token_ids, method="embed")

    @staticmethod
    def _gather_last_valid_hidden(
        prefix_out: jnp.ndarray, prefix_mask: jnp.ndarray
    ) -> jnp.ndarray:
        """Last valid timestep hidden state (b, 1, d) for next-token prediction."""
        # idx = num_valid - 1, clamped (handles empty-mask edge case).
        num_valid = jnp.sum(prefix_mask, axis=1)
        idx = jnp.clip(num_valid - 1, 0, prefix_out.shape[1] - 1)
        b = prefix_out.shape[0]
        batch_i = jnp.arange(b)
        return prefix_out[batch_i, idx, :][:, None, :]

    def _embed_action_suffix(self, obs, noisy_actions, timestep):
        """Embed action tokens for the action expert (same as Pi0.5 suffix)."""
        action_tokens = self.action_in_proj(noisy_actions)
        time_emb = pi0.posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        time_emb = self.time_mlp_in(time_emb)
        time_emb = nnx.swish(time_emb)
        time_emb = self.time_mlp_out(time_emb)
        time_emb = nnx.swish(time_emb)
        mask = jnp.ones(action_tokens.shape[:2], dtype=jnp.bool_)
        ar = [True] + [False] * (self.action_horizon - 1)
        return action_tokens, mask, ar, time_emb

    # ------------------------------------------------------------------
    # Custom attention mask builder
    # ------------------------------------------------------------------
    @staticmethod
    def _build_attention_mask(
        prefix_mask: jnp.ndarray,
        prefix_ar: jnp.ndarray,
        suffix_mask: jnp.ndarray,
        suffix_ar: jnp.ndarray,
        n_img: int,
        n_prompt: int,
        n_subtask: int,
        n_reasoning: int,
        n_fast: int,
        n_action: int,
    ) -> jnp.ndarray:
        """Build a (b, total, total) attention mask implementing:

        - Images + prompt: bidirectional among themselves
        - Reasoning: causal, attending to images + prompt + earlier reasoning
        - Subtask: causal, attending to images + prompt + reasoning + earlier subtask
        - FAST tokens: causal among themselves, attending to images + prompt + subtask
        - Actions: causal among themselves, attend to images + prompt + subtask + FAST (NOT reasoning)
        """
        total = prefix_mask.shape[1] + suffix_mask.shape[1]

        combined_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        combined_ar = jnp.concatenate([prefix_ar, suffix_ar], axis=0)

        # Start with the standard cumsum-based mask from pi0
        cumsum = jnp.cumsum(combined_ar, axis=0)
        attn = cumsum[None, None, :] <= cumsum[None, :, None]
        valid = combined_mask[:, None, :] * combined_mask[:, :, None]
        base_mask = jnp.logical_and(attn, valid)

        # Prefix order: images, prompt, reasoning, subtask, [fast]. Suffix: action expert.
        reasoning_start = n_img + n_prompt
        reasoning_end = reasoning_start + n_reasoning
        fast_start = reasoning_end + n_subtask
        action_start = prefix_mask.shape[1]  # suffix

        col_is_reasoning = (jnp.arange(total) >= reasoning_start) & (jnp.arange(total) < reasoning_end)
        row_is_action = (jnp.arange(total) >= action_start) & (jnp.arange(total) < action_start + n_action)
        row_is_fast = (jnp.arange(total) >= fast_start) & (jnp.arange(total) < fast_start + n_fast)
        row_blocks_reasoning = row_is_action | row_is_fast
        block_mask = ~(row_blocks_reasoning[:, None] & col_is_reasoning[None, :])

        return base_mask & block_mask[None, :, :]

    # ------------------------------------------------------------------
    # Training: compute_loss
    # ------------------------------------------------------------------

    def _compute_loss_and_metrics(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # --- Build prefix tokens ---
        img_tokens, img_masks, img_ar = self._embed_images(observation)
        n_img = sum(t.shape[1] for t in img_tokens)

        prompt_emb = self._embed_text_tokens(observation.tokenized_prompt)
        prompt_mask = observation.tokenized_prompt_mask
        n_prompt = prompt_emb.shape[1]

        reasoning_emb = self._embed_text_tokens(observation.tokenized_reasoning)
        reasoning_mask = observation.tokenized_reasoning_mask
        n_reasoning = reasoning_emb.shape[1]

        subtask_emb = self._embed_text_tokens(observation.tokenized_subtask)
        subtask_mask = observation.tokenized_subtask_mask
        n_subtask = subtask_emb.shape[1]

        prefix_parts = img_tokens + [prompt_emb, reasoning_emb, subtask_emb]
        prefix_mask_parts = img_masks + [prompt_mask, reasoning_mask, subtask_mask]
        prefix_ar_list = (
            img_ar + [False] * n_prompt + [True] * n_reasoning + [True] * n_subtask
        )
        n_fast = 0
        if self._use_fast_tokens and observation.tokenized_fast is not None:
            fast_emb = self._embed_text_tokens(observation.tokenized_fast)
            fast_mask = observation.tokenized_fast_mask
            n_fast = fast_emb.shape[1]
            prefix_parts.append(fast_emb)
            prefix_mask_parts.append(fast_mask)
            prefix_ar_list += [True] * n_fast

        prefix_tokens = jnp.concatenate(prefix_parts, axis=1)
        prefix_mask = jnp.concatenate(prefix_mask_parts, axis=1)
        prefix_ar = jnp.array(prefix_ar_list)

        # --- Build suffix tokens (action expert) ---
        suffix_tokens, suffix_mask, suffix_ar_list, adarms_cond = self._embed_action_suffix(observation, x_t, time)
        suffix_ar = jnp.array(suffix_ar_list)
        n_action = suffix_tokens.shape[1]

        # --- Custom attention mask ---
        attn_mask = self._build_attention_mask(
            prefix_mask, prefix_ar, suffix_mask, suffix_ar,
            n_img, n_prompt, n_subtask, n_reasoning, n_fast, n_action,
        )

        # --- Forward pass ---
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )

        # --- Action flow-matching loss ---
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
        flow_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)  # (batch, horizon)
        if observation.action_loss_mask is not None:
            # Allow disabling action-head supervision for selected samples/timesteps
            # while still training CoT targets on the same batch.
            flow_loss = flow_loss * observation.action_loss_mask.astype(flow_loss.dtype)

        # --- VLM cross-entropy loss on reasoning and subtask ---
        # Project VLM outputs back to vocab logits via the shared embedder
        reasoning_start = n_img + n_prompt
        reasoning_out = prefix_out[:, reasoning_start:reasoning_start + n_reasoning]
        subtask_start = reasoning_start + n_reasoning
        subtask_out = prefix_out[:, subtask_start:subtask_start + n_subtask]
        fast_start = subtask_start + n_subtask
        fast_out = prefix_out[:, fast_start:fast_start + n_fast] if n_fast > 0 else None
        # Logits via embedder (shared embedding weights)
        reasoning_logits = self.PaliGemma.llm(reasoning_out, method="decode_logits")
        subtask_logits = self.PaliGemma.llm(subtask_out, method="decode_logits")

        # Teacher-forced targets: shifted by 1 (predict next token)
        reasoning_targets = observation.tokenized_reasoning
        subtask_targets = observation.tokenized_subtask

        # Shift: logits at segment index i predict target token i+1.
        reasoning_ce = self._token_cross_entropy(
            reasoning_logits[:, :-1], reasoning_targets[:, 1:], reasoning_mask[:, 1:]
        )
        subtask_ce = self._token_cross_entropy(
            subtask_logits[:, :-1], subtask_targets[:, 1:], subtask_mask[:, 1:]
        )

        # Boundary terms omitted by the shifted slices above (same layout as ``sample_cot``):
        # - ``<start_of_reasoning>`` is predicted from the last valid image+prompt hidden;
        # - ``<start_of_subtask>`` is predicted from the last valid reasoning hidden
        #   (after ``<end_of_reasoning>``). End delimiters are covered by the shifted CE.
        prompt_prefix_out = prefix_out[:, :reasoning_start]
        prompt_prefix_mask = jnp.concatenate(img_masks + [prompt_mask], axis=1)
        h_after_prompt = self._gather_last_valid_hidden(prompt_prefix_out, prompt_prefix_mask)
        first_reasoning_logits = self.PaliGemma.llm(h_after_prompt, method="decode_logits")
        first_reasoning_ce = self._token_cross_entropy(
            first_reasoning_logits, reasoning_targets[:, :1], reasoning_mask[:, :1]
        )

        n_reasoning_cols = reasoning_out.shape[1]
        if n_reasoning_cols == 0:
            first_subtask_ce = jnp.zeros_like(reasoning_ce)
        else:
            last_rea_idx = jnp.sum(reasoning_mask, axis=1, keepdims=True).astype(jnp.int32) - 1
            last_rea_idx = jnp.clip(last_rea_idx, 0, n_reasoning_cols - 1)
            bridge_h = jnp.take_along_axis(reasoning_out, last_rea_idx[:, :, None], axis=1)
            first_subtask_logits = self.PaliGemma.llm(bridge_h, method="decode_logits")
            bridge_ok = reasoning_mask.any(axis=1, keepdims=True) & subtask_mask[:, :1]
            first_subtask_ce = self._token_cross_entropy(
                first_subtask_logits, subtask_targets[:, :1], bridge_ok
            )

        fast_ce = jnp.zeros_like(reasoning_ce)
        first_fast_ce = jnp.zeros_like(reasoning_ce)
        if n_fast > 0 and fast_out is not None:
            fast_mask = observation.tokenized_fast_mask
            fast_targets = observation.tokenized_fast
            fast_logits = self.PaliGemma.llm(fast_out, method="decode_logits")
            fast_ce = self._token_cross_entropy(
                fast_logits[:, :-1], fast_targets[:, 1:], fast_mask[:, 1:]
            )
            n_subtask_cols = subtask_out.shape[1]
            if n_subtask_cols == 0:
                first_fast_ce = jnp.zeros_like(fast_ce)
            else:
                last_sub_idx = jnp.sum(subtask_mask, axis=1, keepdims=True).astype(jnp.int32) - 1
                last_sub_idx = jnp.clip(last_sub_idx, 0, n_subtask_cols - 1)
                bridge_h_fast = jnp.take_along_axis(subtask_out, last_sub_idx[:, :, None], axis=1)
                first_fast_logits = self.PaliGemma.llm(bridge_h_fast, method="decode_logits")
                bridge_fast_ok = subtask_mask.any(axis=1, keepdims=True) & fast_mask[:, :1]
                first_fast_ce = self._token_cross_entropy(
                    first_fast_logits, fast_targets[:, :1], bridge_fast_ok
                )

        cot_loss = reasoning_ce + subtask_ce + first_reasoning_ce + first_subtask_ce + fast_ce + first_fast_ce

        # Combine: flow loss is per-timestep (batch, horizon), cot_loss is scalar per batch
        # Broadcast cot_loss to match flow_loss shape for the return
        combined = flow_loss + self.cot_loss_weight * cot_loss[:, None]
        metrics = {
            "flow_loss": jnp.mean(flow_loss, axis=-1),
            "cot_loss": cot_loss,
            "cot_reasoning_ce": reasoning_ce,
            "cot_subtask_ce": subtask_ce,
            "cot_first_reasoning_ce": first_reasoning_ce,
            "cot_first_subtask_ce": first_subtask_ce,
            "cot_fast_ce": fast_ce,
            "cot_first_fast_ce": first_fast_ce,
        }
        return combined, metrics

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        combined, _ = self._compute_loss_and_metrics(rng, observation, actions, train=train)
        return combined

    def compute_loss_with_aux(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        return self._compute_loss_and_metrics(rng, observation, actions, train=train)

    @staticmethod
    def _token_cross_entropy(logits, targets, mask):
        """Per-example cross-entropy loss with masking."""
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        target_log_probs = jnp.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]
        masked = target_log_probs * mask
        return -jnp.sum(masked, axis=-1) / jnp.clip(jnp.sum(mask, axis=-1), 1.0)

    # ------------------------------------------------------------------
    # Inference helpers: optional post-restore module_jit kernels for sample_cot
    # ------------------------------------------------------------------

    def _cot_module_jit(self, name: str, meth, *jit_args, **jit_kwargs):
        cache = _COT_MODULE_JIT_CACHE.setdefault(id(self), {})
        fn = cache.get(name)
        if fn is None:
            fn = nnx_utils.module_jit(meth, *jit_args, **jit_kwargs)
            cache[name] = fn
        return fn

    def _cot_decode_logits_row_impl(self, h: jnp.ndarray) -> jnp.ndarray:
        return self.PaliGemma.llm(h, method="decode_logits")[:, 0, :]

    def _cot_decode_argmax_row_impl(self, h: jnp.ndarray) -> jnp.ndarray:
        """Greedy ids via chunked projection (same as eager path)."""
        return self.PaliGemma.llm(h, method="decode_argmax_chunked")[:, 0]

    def _cot_decode_logits_row(self, h: jnp.ndarray) -> jnp.ndarray:
        if self._cot_jit_decode:
            return self._cot_module_jit("decode_logits_row", self._cot_decode_logits_row_impl)(h)
        return self._cot_decode_logits_row_impl(h)

    def _cot_decode_argmax_row(self, h: jnp.ndarray) -> jnp.ndarray:
        """Greedy next-token ids without full ``(vocab,)`` logits (lower peak VRAM than ``decode_logits``)."""
        if self._cot_jit_decode:
            return self._cot_module_jit("decode_argmax_row", self._cot_decode_argmax_row_impl)(h)
        return self._cot_decode_argmax_row_impl(h)

    def _cot_forward_one_token_impl(
        self,
        tok: jnp.ndarray,
        kv_cache,
        abs_pos: jnp.ndarray,
        attn_mask: jnp.ndarray,
    ):
        emb = self._embed_text_tokens(tok[:, None])
        (out, _), kv_new = self.PaliGemma.llm(
            [emb, None],
            mask=attn_mask,
            positions=abs_pos,
            kv_cache=kv_cache,
        )
        assert out is not None
        return out, kv_new, abs_pos + 1

    def _cot_forward_one_token(
        self,
        tok: jnp.ndarray,
        kv_cache,
        abs_pos: jnp.ndarray,
        attn_mask: jnp.ndarray,
    ):
        if self._cot_jit_transformer_forward:
            return self._cot_module_jit("forward_one_token", self._cot_forward_one_token_impl)(
                tok, kv_cache, abs_pos, attn_mask
            )
        return self._cot_forward_one_token_impl(tok, kv_cache, abs_pos, attn_mask)

    def _cot_replay_one_token_impl(
        self,
        tok: jnp.ndarray,
        h: jnp.ndarray,
        kv_cache,
        abs_pos: jnp.ndarray,
        attn_mask: jnp.ndarray,
        step_ok: jnp.ndarray,
    ):
        emb = self._embed_text_tokens(tok[:, None])
        (out, _), kv_new = self.PaliGemma.llm(
            [emb, None],
            mask=attn_mask,
            positions=abs_pos,
            kv_cache=kv_cache,
        )
        assert out is not None
        h_new = jnp.where(step_ok[:, None, None], out, h)
        return h_new, kv_new, abs_pos + 1

    def _cot_replay_one_token(
        self,
        tok: jnp.ndarray,
        h: jnp.ndarray,
        kv_cache,
        abs_pos: jnp.ndarray,
        attn_mask: jnp.ndarray,
        step_ok: jnp.ndarray,
    ):
        if self._cot_jit_transformer_forward:
            return self._cot_module_jit("replay_one_token", self._cot_replay_one_token_impl)(
                tok, h, kv_cache, abs_pos, attn_mask, step_ok
            )
        return self._cot_replay_one_token_impl(tok, h, kv_cache, abs_pos, attn_mask, step_ok)

    def _cot_fixed_cache(self, kv_cache, max_total_len: int):
        cache_k, cache_v = kv_cache
        pad = max_total_len - cache_k.shape[2]
        cache_k = jnp.pad(cache_k, ((0, 0), (0, 0), (0, pad), (0, 0), (0, 0)))
        cache_v = jnp.pad(cache_v, ((0, 0), (0, 0), (0, pad), (0, 0), (0, 0)))
        cache_idx = jnp.full((cache_k.shape[0],), cache_k.shape[2] - pad, dtype=jnp.int32)
        return cache_idx, cache_k, cache_v

    def _cot_decode_token(self, h: jnp.ndarray, rng: jax.Array, temperature: float):
        if temperature and temperature > 0:
            logits = self._cot_decode_logits_row_impl(h)
            rng, step_rng = jax.random.split(rng)
            tok = jax.random.categorical(step_rng, logits / jnp.maximum(temperature, 1e-6))
            return rng, tok.astype(jnp.int32)
        return rng, self._cot_decode_argmax_row_impl(h)

    def _cot_forward_fixed(
        self,
        tok: jnp.ndarray,
        h: jnp.ndarray,
        kv_cache,
        abs_pos: jnp.ndarray,
        key_mask: jnp.ndarray,
        cur_pos: jnp.ndarray,
        step_ok: jnp.ndarray,
    ):
        key_mask = key_mask.at[:, cur_pos].set(step_ok)
        emb = self._embed_text_tokens(tok[:, None])
        (out, _), kv_cache = self.PaliGemma.llm(
            [emb, None],
            mask=key_mask[:, None, :],
            positions=abs_pos,
            kv_cache=kv_cache,
        )
        assert out is not None
        h = jnp.where(step_ok[:, None, None], out, h)
        return h, kv_cache, abs_pos + 1, key_mask, cur_pos + 1

    def _sample_cot_core_impl(
        self,
        rng: at.KeyArrayLike,
        prefix_tokens: jnp.ndarray,
        prefix_mask: jnp.ndarray,
        prefix_ar: jnp.ndarray,
        mr: int,
        ms: int,
        mf: int,
        temperature: float,
    ):
        batch_size = prefix_tokens.shape[0]
        prefix_len = prefix_tokens.shape[1]
        max_total_len = prefix_len + mr + ms + mf + 1

        prefix_attn_mask = pi0.make_attn_mask(prefix_mask, prefix_ar)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), kv_prompt = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=positions,
        )
        assert prefix_out is not None

        h = self._gather_last_valid_hidden(prefix_out, prefix_mask)
        kv_cache = self._cot_fixed_cache(kv_prompt, max_total_len)
        key_mask = jnp.zeros((batch_size, max_total_len), dtype=jnp.bool_)
        key_mask = key_mask.at[:, :prefix_len].set(prefix_mask)
        abs_pos = jnp.sum(prefix_mask, axis=1, keepdims=True).astype(jnp.int32)
        cur_pos = jnp.asarray(prefix_len, dtype=jnp.int32)

        rea_buf = jnp.zeros((batch_size, mr), dtype=jnp.int32)
        rea_m = jnp.zeros((batch_size, mr), dtype=jnp.bool_)
        done_r = jnp.zeros((batch_size,), dtype=jnp.bool_)

        # Bootstrap reasoning with <start_of_reasoning>. The model is also
        # trained (via first_reasoning_ce) to predict this token from the last
        # prompt hidden, but force-injecting it keeps the segment anchored even
        # when the start prediction is noisy, mirroring the subtask bootstrap
        # below.
        if mr > 0:
            rng, sor_tok = self._cot_decode_token(h, rng, temperature)
            sor_tok = jnp.where(sor_tok == START_OF_REASONING_ID, sor_tok, START_OF_REASONING_ID)
            rea_buf = rea_buf.at[:, 0].set(sor_tok)
            rea_m = rea_m.at[:, 0].set(True)
            h, kv_cache, abs_pos, key_mask, cur_pos = self._cot_forward_fixed(
                sor_tok,
                h,
                kv_cache,
                abs_pos,
                key_mask,
                cur_pos,
                jnp.ones((batch_size,), dtype=jnp.bool_),
            )

        def reasoning_cond(carry):
            j, *_rest, done = carry
            return (j < mr) & (~jnp.all(done))

        def reasoning_body(carry):
            j, rng, h, kv_cache, abs_pos, key_mask, cur_pos, rea_buf, rea_m, done = carry
            active = ~done
            rng, tok = self._cot_decode_token(h, rng, temperature)
            tok = jnp.where(active, tok, jnp.int32(0))
            rea_buf = rea_buf.at[:, j].set(tok)
            rea_m = rea_m.at[:, j].set(active)
            done = done | (active & (tok == END_OF_REASONING_ID))
            should_forward = (j < (mr - 1)) & jnp.any(~done)

            def do_forward(args):
                h, kv_cache, abs_pos, key_mask, cur_pos = args
                # Forward the token for rows that decoded at this step.
                step_ok = active
                return self._cot_forward_fixed(tok, h, kv_cache, abs_pos, key_mask, cur_pos, step_ok)

            h, kv_cache, abs_pos, key_mask, cur_pos = jax.lax.cond(
                should_forward,
                do_forward,
                lambda args: args,
                (h, kv_cache, abs_pos, key_mask, cur_pos),
            )
            return j + 1, rng, h, kv_cache, abs_pos, key_mask, cur_pos, rea_buf, rea_m, done

        rea_start = jnp.asarray(1 if mr > 0 else 0, dtype=jnp.int32)
        (
            _j,
            rng,
            h,
            kv_cache,
            abs_pos,
            key_mask,
            cur_pos,
            rea_buf,
            rea_m,
            _done_r,
        ) = jax.lax.while_loop(
            reasoning_cond,
            reasoning_body,
            (rea_start, rng, h, kv_cache, abs_pos, key_mask, cur_pos, rea_buf, rea_m, done_r),
        )

        rr = jnp.arange(mr, dtype=jnp.int32)[None, :]
        pos_r = rr
        generated_len_r = jnp.sum(rea_m, axis=-1, dtype=jnp.int32)
        matches_end_r = (rea_buf == END_OF_REASONING_ID) & rea_m
        has_end_r = jnp.any(matches_end_r, axis=-1)
        first_end_r = jnp.min(jnp.where(matches_end_r, pos_r, mr), axis=-1)
        # End marker position in the normalized reasoning segment.
        end_pos_r = jnp.where(has_end_r, first_end_r, generated_len_r)
        can_append_end = end_pos_r < mr
        overflow_end = ~can_append_end

        # Keep existing generated tokens only.
        rea_m = rr < generated_len_r[:, None]
        rea_buf = jnp.where(rea_m, rea_buf, 0)

        if mr > 0:
            safe_end_pos = jnp.clip(end_pos_r, 0, mr - 1)
            write_end = (~overflow_end)[:, None] & (rr == safe_end_pos[:, None])
            rea_buf = jnp.where(write_end, END_OF_REASONING_ID, rea_buf)
            rea_m = rea_m | write_end

        # If we ran out of reasoning budget, force end-of-reasoning at the tail slot.
        if mr == 1:
            rea_buf = jnp.where(overflow_end[:, None], END_OF_REASONING_ID, rea_buf)
            rea_m = jnp.where(overflow_end[:, None], jnp.ones_like(rea_m), rea_m)
        elif mr > 1:
            tail_end = overflow_end[:, None] & (rr == (mr - 1))
            rea_buf = jnp.where(tail_end, END_OF_REASONING_ID, rea_buf)
            rea_m = jnp.where(overflow_end[:, None], rr < mr, rea_m)

        def catchup_cond(carry):
            t, *_ = carry
            in_bounds = t < mr
            any_step = jax.lax.cond(
                in_bounds,
                lambda _: jnp.any(rea_m[:, t]),
                lambda _: jnp.array(False),
                operand=None,
            )
            return in_bounds & any_step

        def catchup_body(carry):
            t, h, kv_cache, abs_pos, key_mask, cur_pos = carry
            tok = rea_buf[:, t]
            step_ok = rea_m[:, t]
            h, kv_cache, abs_pos, key_mask, cur_pos = self._cot_forward_fixed(
                tok, h, kv_cache, abs_pos, key_mask, cur_pos, step_ok
            )
            return t + 1, h, kv_cache, abs_pos, key_mask, cur_pos

        # Replaying from 0 guarantees per-row boundary edits are reflected in KV/hidden.
        catchup_start = jnp.asarray(0, dtype=jnp.int32)
        _, h, kv_cache, abs_pos, key_mask, cur_pos = jax.lax.while_loop(
            catchup_cond,
            catchup_body,
            (catchup_start, h, kv_cache, abs_pos, key_mask, cur_pos),
        )

        sub_buf = jnp.zeros((batch_size, ms), dtype=jnp.int32)
        sub_m = jnp.zeros((batch_size, ms), dtype=jnp.bool_)
        done_s = jnp.zeros((batch_size,), dtype=jnp.bool_)

        # Bootstrap subtask with <start_of_subtask> (sampled from h or appended).
        if ms > 0:
            rng, sos_tok = self._cot_decode_token(h, rng, temperature)
            sos_tok = jnp.where(sos_tok == START_OF_SUBTASK_ID, sos_tok, START_OF_SUBTASK_ID)
            sub_buf = sub_buf.at[:, 0].set(sos_tok)
            sub_m = sub_m.at[:, 0].set(True)
            h, kv_cache, abs_pos, key_mask, cur_pos = self._cot_forward_fixed(
                sos_tok,
                h,
                kv_cache,
                abs_pos,
                key_mask,
                cur_pos,
                jnp.ones((batch_size,), dtype=jnp.bool_),
            )

        def subtask_cond(carry):
            i, *_rest, done = carry
            return (i < ms) & (~jnp.all(done))

        def subtask_body(carry):
            i, rng, h, kv_cache, abs_pos, key_mask, cur_pos, sub_buf, sub_m, done = carry
            active = ~done
            rng, tok = self._cot_decode_token(h, rng, temperature)
            tok = jnp.where(active, tok, jnp.int32(0))
            sub_buf = sub_buf.at[:, i].set(tok)
            sub_m = sub_m.at[:, i].set(active)
            done = done | (active & (tok == END_OF_SUBTASK_ID))
            should_forward = (i < (ms - 1)) & jnp.any(~done)

            def do_forward(args):
                h, kv_cache, abs_pos, key_mask, cur_pos = args
                # Forward the token for rows that decoded at this step.
                step_ok = active
                return self._cot_forward_fixed(tok, h, kv_cache, abs_pos, key_mask, cur_pos, step_ok)

            h, kv_cache, abs_pos, key_mask, cur_pos = jax.lax.cond(
                should_forward,
                do_forward,
                lambda args: args,
                (h, kv_cache, abs_pos, key_mask, cur_pos),
            )
            return i + 1, rng, h, kv_cache, abs_pos, key_mask, cur_pos, sub_buf, sub_m, done

        sub_start = jnp.asarray(1 if ms > 0 else 0, dtype=jnp.int32)
        _, _, _, _, _, _, _, sub_buf, sub_m, _ = jax.lax.while_loop(
            subtask_cond,
            subtask_body,
            (sub_start, rng, h, kv_cache, abs_pos, key_mask, cur_pos, sub_buf, sub_m, done_s),
        )
        pos_s = jnp.arange(ms, dtype=jnp.int32)[None, :]
        matches_end_s = (sub_buf == END_OF_SUBTASK_ID) & sub_m
        has_end_s = jnp.any(matches_end_s, axis=-1)
        first_end_s = jnp.min(jnp.where(matches_end_s, pos_s, ms), axis=-1)
        total_len_s = jnp.where(has_end_s, first_end_s + 1, jnp.sum(sub_m, axis=-1))
        sub_m = pos_s < total_len_s[:, None]
        sub_buf = jnp.where(sub_m, sub_buf, 0)

        fast_buf = jnp.zeros((batch_size, mf), dtype=jnp.int32)
        fast_m = jnp.zeros((batch_size, mf), dtype=jnp.bool_)
        if mf > 0:
            action_prefix_ids, pipe_token_id, eos_token_id = _fast_segment_layout_token_ids()
            idx = 0
            for tok_id in action_prefix_ids:
                if idx >= mf:
                    break
                tok = jnp.full((batch_size,), tok_id, dtype=jnp.int32)
                fast_buf = fast_buf.at[:, idx].set(tok)
                fast_m = fast_m.at[:, idx].set(True)
                h, kv_cache, abs_pos, key_mask, cur_pos = self._cot_forward_fixed(
                    tok,
                    h,
                    kv_cache,
                    abs_pos,
                    key_mask,
                    cur_pos,
                    jnp.ones((batch_size,), dtype=jnp.bool_),
                )
                idx += 1

            done_f = jnp.zeros((batch_size,), dtype=jnp.bool_)

            def fast_cond(carry):
                step_idx, *_rest, done = carry
                return (step_idx < mf) & (~jnp.all(done))

            def fast_body(carry):
                step_idx, rng, h, kv_cache, abs_pos, key_mask, cur_pos, fast_buf, fast_m, done = carry
                active = ~done
                rng, tok = self._cot_decode_token(h, rng, temperature)
                tok = jnp.where(active, tok, jnp.int32(0))
                stop = active & ((tok == pipe_token_id) | (tok == eos_token_id))
                fast_buf = fast_buf.at[:, step_idx].set(tok)
                fast_m = fast_m.at[:, step_idx].set(active)
                done = done | stop
                should_forward = (step_idx < (mf - 1)) & jnp.any(~done)

                def do_forward(args):
                    h, kv_cache, abs_pos, key_mask, cur_pos = args
                    return self._cot_forward_fixed(tok, h, kv_cache, abs_pos, key_mask, cur_pos, active)

                h, kv_cache, abs_pos, key_mask, cur_pos = jax.lax.cond(
                    should_forward,
                    do_forward,
                    lambda args: args,
                    (h, kv_cache, abs_pos, key_mask, cur_pos),
                )
                return step_idx + 1, rng, h, kv_cache, abs_pos, key_mask, cur_pos, fast_buf, fast_m, done

            fast_start = jnp.asarray(idx, dtype=jnp.int32)
            _, _, _, _, _, _, _, fast_buf, fast_m, _ = jax.lax.while_loop(
                fast_cond,
                fast_body,
                (fast_start, rng, h, kv_cache, abs_pos, key_mask, cur_pos, fast_buf, fast_m, done_f),
            )
            pos_f = jnp.arange(mf, dtype=jnp.int32)[None, :]
            matches_pipe = (fast_buf == pipe_token_id) & fast_m
            has_pipe = jnp.any(matches_pipe, axis=-1)
            first_pipe = jnp.min(jnp.where(matches_pipe, pos_f, mf), axis=-1)
            total_len_f = jnp.where(has_pipe, first_pipe + 1, jnp.sum(fast_m, axis=-1, dtype=jnp.int32))
            fast_m = pos_f < total_len_f[:, None]
            fast_buf = jnp.where(fast_m, fast_buf, 0)

        return rea_buf, rea_m, sub_buf, sub_m, fast_buf, fast_m

    def _sample_cot_core(
        self,
        rng: at.KeyArrayLike,
        prefix_tokens: jnp.ndarray,
        prefix_mask: jnp.ndarray,
        prefix_ar: jnp.ndarray,
        *,
        mr: int,
        ms: int,
        temperature: float,
    ):
        return self._cot_module_jit(
            "sample_cot_core",
            self._sample_cot_core_impl,
            static_argnums=(5, 6, 7, 8),
        )(rng, prefix_tokens, prefix_mask, prefix_ar, mr, ms, mf, temperature)

    # ------------------------------------------------------------------
    # Inference: sample_cot (autoregressive CoT, prompt-only conditioning)
    # ------------------------------------------------------------------

    def sample_cot(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        temperature: float = 0.0,
        max_subtask_len: int | None = None,
        max_reasoning_len: int | None = None,
        image_keys: Sequence[str] | None = None,
    ) -> dict[str, jnp.ndarray]:
        """Autoregressively sample reasoning, subtask, and optional FAST tokens from images + prompt."""
        keys = tuple(image_keys) if image_keys is not None else self._preprocess_image_keys
        observation = _model.preprocess_observation(None, observation, train=False, image_keys=keys)
        if observation.tokenized_prompt is None or observation.tokenized_prompt_mask is None:
            raise ValueError("sample_cot requires tokenized_prompt and tokenized_prompt_mask")

        ms = max_subtask_len if max_subtask_len is not None else self.max_subtask_len
        mr = max_reasoning_len if max_reasoning_len is not None else self.max_reasoning_len
        mf = int(self.max_fast_len) if self._use_fast_tokens else 0
        img_tokens, img_masks, img_ar = self._embed_images(observation)

        prompt_emb = self._embed_text_tokens(observation.tokenized_prompt)
        prompt_mask = observation.tokenized_prompt_mask

        prefix_tokens = jnp.concatenate(img_tokens + [prompt_emb], axis=1)
        prefix_mask = jnp.concatenate(img_masks + [prompt_mask], axis=1)
        prefix_ar = jnp.array(img_ar + [False] * prompt_emb.shape[1])
        rea_buf, rea_m, sub_buf, sub_m, fast_buf, fast_m = self._sample_cot_core(
            rng,
            prefix_tokens,
            prefix_mask,
            prefix_ar,
            mr=int(mr),
            ms=int(ms),
            mf=int(mf),
            temperature=float(temperature),
        )

        out = {
            "tokenized_subtask": sub_buf,
            "tokenized_subtask_mask": sub_m,
            "tokenized_reasoning": rea_buf,
            "tokenized_reasoning_mask": rea_m,
        }
        if self._use_fast_tokens:
            out["tokenized_fast"] = fast_buf
            out["tokenized_fast_mask"] = fast_m
        return out

    @nnx.jit
    def _denoise_flow_step(
        self,
        observation: _model.Observation,
        kv_cache,
        prefix_mask: jnp.ndarray,
        prefix_mask_no_reasoning: jnp.ndarray,
        dt: jnp.ndarray,
        x_t: jnp.ndarray,
        t: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Single flow-matching step (suffix forward under frozen prefix ``kv_cache``)."""
        b = observation.state.shape[0]
        t_b = jnp.broadcast_to(t, (b,))
        suffix_tokens, suffix_mask, suffix_ar_list, adarms_cond = self._embed_action_suffix(
            observation, x_t, t_b
        )
        suffix_ar = jnp.array(suffix_ar_list)
        suffix_attn_mask = pi0.make_attn_mask(suffix_mask, suffix_ar)
        action_to_prefix = einops.repeat(
            prefix_mask_no_reasoning, "b p -> b s p", s=suffix_tokens.shape[1]
        )
        full_attn_mask = jnp.concatenate([action_to_prefix, suffix_attn_mask], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
        return x_t + dt * v_t, t + dt

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        image_keys: Sequence[str] | None = None,
    ) -> _model.Actions:
        """Flow-match denoise from prefix KV + action suffix.

        ``jit_denoise_steps``: Python loop over ``num_steps``, each iteration calling this
        module's jitted :meth:`_denoise_flow_step` (separate XLA program per step; usually
        between fused ``while_loop`` and fully eager in peak VRAM and speed).

        ``low_memory_denoise``: fully eager Python loop (no per-step ``nnx.jit``). Do not wrap
        the whole :meth:`sample_actions` in ``nnx.jit`` when using either loop mode.
        """
        keys = tuple(image_keys) if image_keys is not None else self._preprocess_image_keys
        observation = _model.preprocess_observation(None, observation, train=False, image_keys=keys)
        batch_size = observation.state.shape[0]

        # --- Build prefix: images + prompt + reasoning + subtask ---
        img_tokens, img_masks, img_ar = self._embed_images(observation)
        n_img = sum(t.shape[1] for t in img_tokens)

        prompt_emb = self._embed_text_tokens(observation.tokenized_prompt)
        prompt_mask = observation.tokenized_prompt_mask
        n_prompt = prompt_emb.shape[1]

        # During inference, subtask/reasoning may be provided as ground-truth
        # (for teacher-forced eval) or could be autoregressively generated.
        # For now, we support teacher-forced inference with provided tokens.
        reasoning_emb = self._embed_text_tokens(observation.tokenized_reasoning)
        reasoning_mask = observation.tokenized_reasoning_mask
        n_reasoning = reasoning_emb.shape[1]

        subtask_emb = self._embed_text_tokens(observation.tokenized_subtask)
        subtask_mask = observation.tokenized_subtask_mask
        n_subtask = subtask_emb.shape[1]

        prefix_parts = img_tokens + [prompt_emb, reasoning_emb, subtask_emb]
        prefix_mask_parts = img_masks + [prompt_mask, reasoning_mask, subtask_mask]
        prefix_ar_list = img_ar + [False] * n_prompt + [True] * n_reasoning + [True] * n_subtask
        if self._use_fast_tokens and observation.tokenized_fast is not None:
            fast_emb = self._embed_text_tokens(observation.tokenized_fast)
            fast_mask = observation.tokenized_fast_mask
            prefix_parts.append(fast_emb)
            prefix_mask_parts.append(fast_mask)
            prefix_ar_list += [True] * fast_emb.shape[1]

        prefix_tokens = jnp.concatenate(prefix_parts, axis=1)
        prefix_mask = jnp.concatenate(prefix_mask_parts, axis=1)
        prefix_ar = jnp.array(prefix_ar_list)

        # Prefix attention (blocks FAST/action from reasoning when fast tokens are present).
        n_fast = 0
        if self._use_fast_tokens and observation.tokenized_fast is not None:
            n_fast = observation.tokenized_fast.shape[1]
        empty_suffix_mask = jnp.zeros((batch_size, 0), dtype=jnp.bool_)
        empty_suffix_ar = jnp.array([], dtype=jnp.bool_)
        prefix_attn_mask = self._build_attention_mask(
            prefix_mask,
            prefix_ar,
            empty_suffix_mask,
            empty_suffix_ar,
            n_img,
            n_prompt,
            n_subtask,
            n_reasoning,
            n_fast,
            0,
        )
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        # --- Denoising loop ---
        n_steps_i = int(num_steps)
        dt = -1.0 / float(n_steps_i)
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # Build a per-column mask that blocks reasoning tokens from action attention
        prefix_len = prefix_mask.shape[1]
        reasoning_start = n_img + n_prompt
        reasoning_end = reasoning_start + n_reasoning
        col_is_reasoning = (jnp.arange(prefix_len) >= reasoning_start) & (jnp.arange(prefix_len) < reasoning_end)
        # Shape (b, prefix_len): valid prefix tokens minus reasoning columns
        prefix_mask_no_reasoning = prefix_mask & ~col_is_reasoning[None, :]

        def step(carry):
            x_t, t = carry
            suffix_tokens, suffix_mask, suffix_ar_list, adarms_cond = self._embed_action_suffix(
                observation, x_t, jnp.broadcast_to(t, (batch_size,))
            )
            suffix_ar = jnp.array(suffix_ar_list)
            suffix_attn_mask = pi0.make_attn_mask(suffix_mask, suffix_ar)

            # (b, suffix_len, prefix_len): each action token sees prefix minus reasoning
            action_to_prefix = einops.repeat(
                prefix_mask_no_reasoning, "b p -> b s p", s=suffix_tokens.shape[1]
            )
            full_attn_mask = jnp.concatenate([action_to_prefix, suffix_attn_mask], axis=-1)

            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
            return x_t + dt * v_t, t + dt


        t0 = jnp.asarray(1.0, dtype=noise.dtype)

        def cond(carry):
            _, t = carry
            return t >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, t0))
        return x_0
