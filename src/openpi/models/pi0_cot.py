"""Pi0.5 with Chain-of-Thought: reasoning + subtask generation before action decoding.

Prefix layout (language, after image tokens; see ``CoTPaligemmaTokenizer``):

    Prompt:...;Reasoning:<start_of_reasoning>
    |--- bidirectional ---|...|--- causal: reasoning body + <end_of_reasoning> ---|
    ;Subtask:<start_of_subtask>...<end_of_subtask>
    |--- causal ---|

Suffix (Action expert, adaRMS):
    [action tokens] — attend to images + prompt + subtask (not reasoning).

Losses: CE on reasoning + subtask (VLM), flow-matching on actions.
"""

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
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")

# Must match ``CoTPaligemmaTokenizer._cot_skip_tokens`` and special-token layout.
_COT_SKIP_LAST = 128

START_OF_SUBTASK_ID = _gemma.PALIGEMMA_VOCAB_SIZE - 1 - _COT_SKIP_LAST - 1
END_OF_SUBTASK_ID = _gemma.PALIGEMMA_VOCAB_SIZE - 1 - _COT_SKIP_LAST - 2
START_OF_REASONING_ID = _gemma.PALIGEMMA_VOCAB_SIZE - 1 - _COT_SKIP_LAST - 3
END_OF_REASONING_ID = _gemma.PALIGEMMA_VOCAB_SIZE - 1 - _COT_SKIP_LAST - 4


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
        self._preprocess_image_keys: tuple[str, ...] = (
            tuple(config.inference_image_keys)
            if config.inference_image_keys is not None
            else tuple(_model.IMAGE_KEYS)
        )

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
        n_action: int,
    ) -> jnp.ndarray:
        """Build a (b, total, total) attention mask implementing:

        - Images + prompt: bidirectional among themselves
        - Reasoning: causal, attending to images + prompt + earlier reasoning
        - Subtask: causal, attending to images + prompt + reasoning + earlier subtask
        - Actions: causal among themselves, attend to images + prompt + subtask (NOT reasoning)
        """
        total = prefix_mask.shape[1] + suffix_mask.shape[1]
        batch = prefix_mask.shape[0]

        combined_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        combined_ar = jnp.concatenate([prefix_ar, suffix_ar], axis=0)

        # Start with the standard cumsum-based mask from pi0
        cumsum = jnp.cumsum(combined_ar, axis=0)
        attn = cumsum[None, None, :] <= cumsum[None, :, None]
        valid = combined_mask[:, None, :] * combined_mask[:, :, None]
        base_mask = jnp.logical_and(attn, valid)

        # Now zero out action-tokens → reasoning-tokens attention.
        # Prefix order: images, prompt, reasoning, subtask (reasoning before subtask).
        reasoning_start = n_img + n_prompt
        reasoning_end = reasoning_start + n_reasoning
        action_start = reasoning_end + n_subtask  # suffix starts after full prefix

        # Mask: for rows [action_start : action_start + n_action],
        #        zero out columns [reasoning_start : reasoning_end]
        row_is_action = (jnp.arange(total) >= action_start) & (jnp.arange(total) < action_start + n_action)
        col_is_reasoning = (jnp.arange(total) >= reasoning_start) & (jnp.arange(total) < reasoning_end)
        block_mask = ~(row_is_action[:, None] & col_is_reasoning[None, :])

        return base_mask & block_mask[None, :, :]

    # ------------------------------------------------------------------
    # Training: compute_loss
    # ------------------------------------------------------------------

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
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

        prefix_tokens = jnp.concatenate(img_tokens + [prompt_emb, reasoning_emb, subtask_emb], axis=1)
        prefix_mask = jnp.concatenate(img_masks + [prompt_mask, reasoning_mask, subtask_mask], axis=1)
        # AR mask: bidirectional for images+prompt, causal for reasoning then subtask
        prefix_ar = jnp.array(
            img_ar
            + [False] * n_prompt
            + [True] * n_reasoning
            + [True] * n_subtask
        )

        # --- Build suffix tokens (action expert) ---
        suffix_tokens, suffix_mask, suffix_ar_list, adarms_cond = self._embed_action_suffix(observation, x_t, time)
        suffix_ar = jnp.array(suffix_ar_list)
        n_action = suffix_tokens.shape[1]

        # --- Custom attention mask ---
        attn_mask = self._build_attention_mask(
            prefix_mask, prefix_ar, suffix_mask, suffix_ar,
            n_img, n_prompt, n_subtask, n_reasoning, n_action,
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

        # --- VLM cross-entropy loss on reasoning and subtask ---
        # Project VLM outputs back to vocab logits via the shared embedder
        reasoning_start = n_img + n_prompt
        reasoning_out = prefix_out[:, reasoning_start:reasoning_start + n_reasoning]
        subtask_start = reasoning_start + n_reasoning
        subtask_out = prefix_out[:, subtask_start:subtask_start + n_subtask]

        # Logits via embedder (shared embedding weights); do not use method="decode" (that returns argmax ids).
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

        # Boundary terms omitted by the slices above (same next-token layout as ``sample_cot``):
        # - first reasoning body token is predicted from the last valid image+prompt hidden
        #   (after ``<start_of_reasoning>``, not from reasoning_logits[:, 0]);
        # - first subtask body token is predicted from the last valid reasoning hidden
        #   (after ``<start_of_subtask>``), i.e. not from subtask_logits[:, 0].
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

        cot_loss = reasoning_ce + subtask_ce + first_reasoning_ce + first_subtask_ce

        # Combine: flow loss is per-timestep (batch, horizon), cot_loss is scalar per batch
        # Broadcast cot_loss to match flow_loss shape for the return
        combined = flow_loss + self.cot_loss_weight * cot_loss[:, None]
        return combined

    @staticmethod
    def _token_cross_entropy(logits, targets, mask):
        """Per-example cross-entropy loss with masking."""
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        target_log_probs = jnp.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]
        masked = target_log_probs * mask
        return -jnp.sum(masked, axis=-1) / jnp.clip(jnp.sum(mask, axis=-1), 1.0)

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
        """Autoregressively sample reasoning then subtask from images + prompt prefix only.

        ``observation.tokenized_prompt`` must be set (through ``<start_of_reasoning>``, as in
        training). Ground-truth subtask/reasoning fields are ignored.

        Returns token ID buffers and boolean masks (True = generated timestep), same layout
        as training keys ``tokenized_subtask*`` / ``tokenized_reasoning*``.

        Args:
            image_keys: Subset of camera keys to resize/embed. If ``None``, uses
                ``Pi0CoTConfig.inference_image_keys`` when set, else all ``IMAGE_KEYS``.
                CARLA single-camera inference can pass ``("base_0_rgb",)`` or set that on the config.
        """
        keys = tuple(image_keys) if image_keys is not None else self._preprocess_image_keys
        observation = _model.preprocess_observation(None, observation, train=False, image_keys=keys)
        if observation.tokenized_prompt is None or observation.tokenized_prompt_mask is None:
            raise ValueError("sample_cot requires tokenized_prompt and tokenized_prompt_mask")

        batch_size = observation.state.shape[0]
        ms = max_subtask_len if max_subtask_len is not None else self.max_subtask_len
        mr = max_reasoning_len if max_reasoning_len is not None else self.max_reasoning_len

        # Embed images
        img_tokens, img_masks, img_ar = self._embed_images(observation)
        
        # Embed prompt
        prompt_emb = self._embed_text_tokens(observation.tokenized_prompt)
        prompt_mask = observation.tokenized_prompt_mask
        
        # Construct prefix
        prefix_tokens = jnp.concatenate(img_tokens + [prompt_emb], axis=1)
        prefix_mask = jnp.concatenate(img_masks + [prompt_mask], axis=1)
        prefix_ar = jnp.array(img_ar + [False] * prompt_emb.shape[1])
        
        # Build attention mask
        prefix_attn_mask = pi0.make_attn_mask(prefix_mask, prefix_ar)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        
        # Prefill prefix (keep KV snapshot so we can replay truncated reasoning + <start_of_subtask>.)
        (prefix_out, _), kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
        )
        assert prefix_out is not None
        kv_prompt = kv_cache
        prefix_out_prompt = prefix_out

        h = self._gather_last_valid_hidden(prefix_out, prefix_mask)
        key_mask = prefix_mask
        abs_pos = jnp.sum(prefix_mask, axis=1, keepdims=True).astype(jnp.int32)

        rea_buf = jnp.zeros((batch_size, mr), dtype=jnp.int32)
        rea_m = jnp.zeros((batch_size, mr), dtype=jnp.bool_)
        rng_cur = rng

        # Generate reasoning (first causal segment after the prompt)
        for j in range(mr):
            logits = self.PaliGemma.llm(h, method="decode_logits")[:, 0, :]
            if temperature and temperature > 0:
                rng_cur, step_rng = jax.random.split(rng_cur)
                tok = jax.random.categorical(step_rng, logits / jnp.maximum(temperature, 1e-6))
            else:
                tok = jnp.argmax(logits, axis=-1)
            rea_buf = rea_buf.at[:, j].set(tok)
            rea_m = rea_m.at[:, j].set(True)
            if j == mr - 1:
                break
            emb = self._embed_text_tokens(tok[:, None])
            key_mask = jnp.concatenate([key_mask, jnp.ones((batch_size, 1), dtype=jnp.bool_)], axis=1)
            attn_mask = key_mask[:, None, :]
            (out, _), kv_cache = self.PaliGemma.llm(
                [emb, None],
                mask=attn_mask,
                positions=abs_pos,
                kv_cache=kv_cache,
            )
            assert out is not None
            h = out
            abs_pos = abs_pos + 1

        pos_r = jnp.arange(mr, dtype=jnp.int32)[None, :]
        matches_end_r = (rea_buf == END_OF_REASONING_ID) & rea_m
        has_end_r = jnp.any(matches_end_r, axis=-1)
        first_end_r = jnp.min(jnp.where(matches_end_r, pos_r, mr), axis=-1)
        body_len_r = jnp.where(has_end_r, first_end_r + 1, mr)
        last_tok_r = jnp.take_along_axis(rea_buf, jnp.clip(body_len_r - 1, 0)[:, None], axis=1).squeeze(-1)
        need_sos = last_tok_r != START_OF_SUBTASK_ID
        total_len_r = jnp.minimum(body_len_r + need_sos.astype(jnp.int32), mr)
        rr = jnp.arange(mr, dtype=jnp.int32)[None, :]
        rea_m = rr < total_len_r[:, None]
        rea_buf = jnp.where(
            rr < body_len_r[:, None],
            rea_buf,
            jnp.where(
                (rr == body_len_r[:, None]) & need_sos[:, None] & (total_len_r[:, None] > body_len_r[:, None]),
                START_OF_SUBTASK_ID,
                0,
            ),
        )

        # Replay reasoning + optional <start_of_subtask> from the prompt KV so h / cache match ``rea_buf``.
        kv_cache = kv_prompt
        key_mask = prefix_mask
        abs_pos = jnp.sum(prefix_mask, axis=1, keepdims=True).astype(jnp.int32)
        h = self._gather_last_valid_hidden(prefix_out_prompt, prefix_mask)
        for t in range(mr):
            tok = rea_buf[:, t]
            step_ok = rea_m[:, t]
            emb = self._embed_text_tokens(tok[:, None])
            key_mask = jnp.concatenate([key_mask, step_ok[:, None]], axis=1)
            attn_mask = key_mask[:, None, :]
            (out, _), kv_cache = self.PaliGemma.llm(
                [emb, None],
                mask=attn_mask,
                positions=abs_pos,
                kv_cache=kv_cache,
            )
            assert out is not None
            h = jnp.where(step_ok[:, None, None], out, h)
            abs_pos = abs_pos + 1

        sub_buf = jnp.zeros((batch_size, ms), dtype=jnp.int32)
        sub_m = jnp.zeros((batch_size, ms), dtype=jnp.bool_)

        for i in range(ms):
            logits = self.PaliGemma.llm(h, method="decode_logits")[:, 0, :]
            if temperature and temperature > 0:
                rng_cur, step_rng = jax.random.split(rng_cur)
                tok = jax.random.categorical(step_rng, logits / jnp.maximum(temperature, 1e-6))
            else:
                tok = jnp.argmax(logits, axis=-1)
            sub_buf = sub_buf.at[:, i].set(tok)
            sub_m = sub_m.at[:, i].set(True)
            if i == ms - 1:
                break
            emb = self._embed_text_tokens(tok[:, None])
            key_mask = jnp.concatenate([key_mask, jnp.ones((batch_size, 1), dtype=jnp.bool_)], axis=1)
            attn_mask = key_mask[:, None, :]
            (out, _), kv_cache = self.PaliGemma.llm(
                [emb, None],
                mask=attn_mask,
                positions=abs_pos,
                kv_cache=kv_cache,
            )
            assert out is not None
            h = out
            abs_pos = abs_pos + 1

        return {
            "tokenized_subtask": sub_buf,
            "tokenized_subtask_mask": sub_m,
            "tokenized_reasoning": rea_buf,
            "tokenized_reasoning_mask": rea_m,
        }

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

        prefix_tokens = jnp.concatenate(img_tokens + [prompt_emb, reasoning_emb, subtask_emb], axis=1)
        prefix_mask = jnp.concatenate(img_masks + [prompt_mask, reasoning_mask, subtask_mask], axis=1)
        prefix_ar = jnp.array(
            img_ar + [False] * n_prompt + [True] * n_reasoning + [True] * n_subtask
        )

        # Fill KV cache with prefix
        prefix_attn_mask = pi0.make_attn_mask(prefix_mask, prefix_ar)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        # --- Denoising loop ---
        dt = -1.0 / num_steps
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
                observation, x_t, jnp.broadcast_to(t, batch_size)
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

        def cond(carry):
            _, t = carry
            return t >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
