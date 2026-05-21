import logging
import os

import jax
import numpy as np
import orbax.checkpoint as ocp
import sentencepiece
from transformers import AutoProcessor

import openpi.models.utils.fsq_tokenizer as fsq_tokenizer
import openpi.shared.download as download

# Last N token ids in the PaliGemma vocab are reserved (special tokens, FAST, CoT).
PALIGEMMA_VOCAB_SKIP_TOKENS = 128
# CoT delimiter ids occupy the first slots of that region (before FAST action codes).
COT_DELIMITER_TOKEN_SLOTS = 4


class PaligemmaTokenizer:
    def __init__(self, max_len: int = 48):
        self._max_len = max_len

        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    def tokenize(self, prompt: str, state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        if state is not None:
            # This is the Pi05 format, where the state is part of the discrete language input.
            discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
            state_str = " ".join(map(str, discretized_state))
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            tokens = self._tokenizer.encode(full_prompt, add_bos=True)
        else:
            # This is the Pi0 format, where the state is part of the continuous action expert input.
            # tokenize "\n" separately as the "start of answer" token
            tokens = self._tokenizer.encode(cleaned_text, add_bos=True) + self._tokenizer.encode("\n")
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + padding
            tokens = tokens + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len

        return np.asarray(tokens), np.asarray(mask)


class CoTPaligemmaTokenizer:
    """Tokenizer for Chain-of-Thought Pi0.5 that produces three separate token
    sequences: prompt (input), reasoning (generated), and subtask (generated).

    Full logical layout (concatenated at train/inference time):

        Prompt:"task";State:"state";
        <start_of_reasoning>"reasoning"<end_of_reasoning>;
        <start_of_subtask>"subtask"<end_of_subtask>

    Attention (see ``Pi0CoT``): images + prompt segment bidirectional; reasoning and
    subtask segments causal; optional FAST action tokens are causal and attend to
    images + prompt + subtask (not reasoning); the action expert does not attend to
    reasoning.
    """

    def __init__(
        self,
        max_prompt_len: int = 64,
        max_subtask_len: int = 48,
        max_reasoning_len: int = 96,
        max_fast_len: int = 64,
        use_fast_tokens: bool = False,
        fast_tokenizer_path: str = "physical-intelligence/fast",
    ):
        self._max_prompt_len = max_prompt_len
        self._max_subtask_len = max_subtask_len
        self._max_reasoning_len = max_reasoning_len
        self._max_fast_len = max_fast_len
        self.use_fast_tokens = use_fast_tokens

        self._cot_skip_tokens = PALIGEMMA_VOCAB_SKIP_TOKENS

        path = download.maybe_download(
            "gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        self._fast_tokenizer: FASTTokenizer | None = None
        if use_fast_tokens:
            self._fast_tokenizer = FASTTokenizer(
                max_len=max_fast_len,
                fast_tokenizer_path=fast_tokenizer_path,
                reserved_vocab_slots=COT_DELIMITER_TOKEN_SLOTS,
            )

    def _pad_or_truncate(
        self,
        tokens: list[int],
        mask: list[bool],
        max_len: int,
        *,
        segment_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(tokens) < max_len:
            pad = [False] * (max_len - len(tokens))
            tokens = tokens + pad
            mask = mask + pad
        elif len(tokens) > max_len:
            logging.warning(
                f"{segment_name} token length ({len(tokens)}) exceeds max ({max_len}), truncating."
            )
            tokens = tokens[:max_len]
            mask = mask[:max_len]
        return np.asarray(tokens, dtype=np.int32), np.asarray(mask, dtype=bool)

    def tokenize_prompt(
        self,
        prompt: str,
        state: np.ndarray,
        *,
        state_dim: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Tokenize the bidirectional prefix (prompt + state; no CoT delimiters).

        ``state_dim`` limits how many proprio dimensions are embedded in the prompt.
        Use this when ``state`` has been zero-padded to ``model.action_dim`` but only
        the leading values are meaningful (e.g. SteerVLA speed/course).
        """
        cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
        state = np.asarray(state)
        if state_dim is not None:
            state = state[..., :state_dim]
        discretized = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        state_str = " ".join(map(str, discretized))
        text = (
            f"Prompt:{cleaned};State:{state_str};"
        )
        tokens = self._tokenizer.encode(text, add_bos=True)
        mask = [True] * len(tokens)
        return self._pad_or_truncate(tokens, mask, self._max_prompt_len, segment_name="prompt")

    def tokenize_subtask(self, subtask: str) -> tuple[np.ndarray, np.ndarray]:
        """Tokenize subtask segment (causal; follows reasoning in the prefix)."""
        cleaned = subtask.strip().replace("_", " ").replace("\n", " ")
        text = f"{cleaned};"
        tokens = (
            [self._start_of_subtask()]
            + self._tokenizer.encode(text)
            + [self._end_of_subtask()]
            + [self._tokenizer.eos_id()]
        )
        mask = [True] * len(tokens)
        return self._pad_or_truncate(tokens, mask, self._max_subtask_len, segment_name="subtask")

    def tokenize_reasoning(self, reasoning: str) -> tuple[np.ndarray, np.ndarray]:
        """Tokenize reasoning segment (causal; first generated segment after the prompt)."""
        cleaned = reasoning.strip().replace("_", " ").replace("\n", " ")
        text = f"{cleaned};"
        tokens = [self._start_of_reasoning()] + self._tokenizer.encode(text) + [self._end_of_reasoning()]
        mask = [True] * len(tokens)
        return self._pad_or_truncate(tokens, mask, self._max_reasoning_len, segment_name="reasoning")

    def tokenize_fast_actions(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Tokenize FAST-discretized actions for VLM supervision (after subtask in the prefix).

        Layout matches ``FASTTokenizer``: ``Action: <fast tokens> |`` with EOS.
        """
        if self._fast_tokenizer is None:
            raise RuntimeError("FAST tokenizer is disabled; set use_fast_tokens=True on CoTPaligemmaTokenizer.")
        action_tokens = self._fast_tokenizer._fast_tokenizer(actions[None])[0]
        action_tokens_in_pg = self._fast_tokenizer._act_tokens_to_paligemma_tokens(action_tokens)
        tokens = (
            self._tokenizer.encode("Action: ")
            + action_tokens_in_pg.tolist()
            + self._tokenizer.encode("|", add_eos=True)
        )
        mask = [True] * len(tokens)
        return self._pad_or_truncate(tokens, mask, self._max_fast_len, segment_name="fast")

    def extract_fast_actions(
        self,
        tokens: np.ndarray,
        action_horizon: int,
        action_dim: int,
        *,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Decode FAST actions from a tokenized fast segment (for eval / visualization)."""
        if self._fast_tokenizer is None:
            raise RuntimeError("FAST tokenizer is disabled; set use_fast_tokens=True on CoTPaligemmaTokenizer.")
        if mask is not None:
            tokens = tokens[mask.astype(bool)]
        return self._fast_tokenizer.extract_actions_from_fast_segment(
            tokens, action_horizon, action_dim
        )

    @property
    def max_fast_len(self) -> int:
        return self._max_fast_len

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.vocab_size()
    
    def _start_of_subtask(self) -> int:
        return self._cot_reserved_token_id(1)

    def _end_of_subtask(self) -> int:
        return self._cot_reserved_token_id(2)

    def _start_of_reasoning(self) -> int:
        return self._cot_reserved_token_id(3)

    def _end_of_reasoning(self) -> int:
        return self._cot_reserved_token_id(4)

    def _cot_reserved_token_id(self, slot: int) -> int:
        """Map CoT delimiter slot (1..COT_DELIMITER_TOKEN_SLOTS) to a PaliGemma token id."""
        if not 1 <= slot <= COT_DELIMITER_TOKEN_SLOTS:
            raise ValueError(f"CoT reserved slot must be in 1..{COT_DELIMITER_TOKEN_SLOTS}, got {slot}")
        return self.vocab_size - 1 - self._cot_skip_tokens - slot
    
    


class FASTTokenizer:
    def __init__(
        self,
        max_len: int = 256,
        fast_tokenizer_path: str = "physical-intelligence/fast",
        *,
        reserved_vocab_slots: int = 0,
    ):
        self._max_len = max_len
        self._fast_skip_tokens = PALIGEMMA_VOCAB_SKIP_TOKENS
        # Extra ids reserved before FAST codes (e.g. CoT delimiter tokens in Pi0CoT).
        self._reserved_vocab_slots = reserved_vocab_slots

        # Download base PaliGemma tokenizer
        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        # Instantiate FAST tokenizer
        self._fast_tokenizer = AutoProcessor.from_pretrained(fast_tokenizer_path, trust_remote_code=True)

    def tokenize(
        self, prompt: str, state: np.ndarray, actions: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cleaned_text = prompt.lower().strip().replace("_", " ")

        # Convention: state gets discretized into 256 discrete bins (assumed range after normalization: [-1, 1])
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        # Convention: prefix includes prompt and string-representation of state, followed by ';'
        state_str = " ".join(map(str, discretized_state))
        prefix = f"Task: {cleaned_text}, State: {state_str};\n"
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        if actions is not None:
            # Tokenize actions with FAST tokenizer --> map to last tokens in PaliGemma vocab
            action_tokens = self._fast_tokenizer(actions[None])[0]
            action_tokens_in_pg = self._act_tokens_to_paligemma_tokens(action_tokens)

            # Convention: postfix contains 'Action:' followed by FAST tokens, followed by '|'
            postfix_tokens = (
                self._paligemma_tokenizer.encode("Action: ")
                + action_tokens_in_pg.tolist()
                + self._paligemma_tokenizer.encode("|", add_eos=True)
            )
        else:
            postfix_tokens = []

        # Create output token sequence & masks
        # AR mask is 0 on prefix (bidirectional attention) and 1 on postfix (causal attention to all previous tokens)
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)  # Loss on postfix only

        # Pad tokens to max length
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        return np.asarray(tokens), np.asarray(token_mask), np.asarray(ar_mask), np.asarray(loss_mask)

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        return self.extract_actions_from_fast_segment(tokens, action_horizon, action_dim)

    def _fast_pg_anchor(self) -> int:
        return (
            self._paligemma_tokenizer.vocab_size()
            - 1
            - self._fast_skip_tokens
            - self._reserved_vocab_slots
        )

    def _pg_tokens_to_fast_token_ids(self, pg_tokens: np.ndarray) -> np.ndarray:
        anchor = self._fast_pg_anchor()
        fast_ids = anchor - np.asarray(pg_tokens, dtype=np.int64)
        return fast_ids[(fast_ids >= 0) & (fast_ids < 4096)]

    def _pg_token_ids_between_action_and_pipe(self, token_ids: np.ndarray) -> np.ndarray:
        """Return PaliGemma token ids for the FAST action body (between ``Action:`` and ``|``)."""
        ids = np.asarray(token_ids, dtype=np.int32).reshape(-1)
        if ids.size == 0:
            return np.array([], dtype=np.int32)

        action_prefix = self._paligemma_tokenizer.encode("Action: ")
        pipe_token = self._paligemma_tokenizer.encode("|")[0]

        start = None
        for i in range(len(ids) - len(action_prefix) + 1):
            if ids[i : i + len(action_prefix)].tolist() == action_prefix:
                start = i + len(action_prefix)
                break
        if start is None:
            return np.array([], dtype=np.int32)

        end = len(ids)
        for j in range(start, len(ids)):
            if int(ids[j]) == pipe_token:
                end = j
                break
        return ids[start:end]

    def extract_actions_from_fast_segment(
        self, tokens: np.ndarray, action_horizon: int, action_dim: int
    ) -> np.ndarray:
        """Decode continuous actions from a ``Action: <FAST pg tokens> |`` segment."""
        pg_body = self._pg_token_ids_between_action_and_pipe(tokens)
        fast_ids = self._pg_tokens_to_fast_token_ids(pg_body)
        if fast_ids.size == 0:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)
        return self._fast_tokenizer.decode(
            [fast_ids.astype(np.int32).tolist()],
            time_horizon=action_horizon,
            action_dim=action_dim,
        )[0]

    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return (
            self._paligemma_tokenizer.vocab_size()
            - 1
            - self._fast_skip_tokens
            - self._reserved_vocab_slots
            - tokens
        )


###########################################################################
## The tokenizers below are used for RoboArena baseline implementations. ##
## They are *not* used for pi0-style models.                             ##
###########################################################################


class BinningTokenizer:
    """
    Standard RT-2 / OpenVLA style binning tokenizer.
    """

    def __init__(self, max_len: int = 256, n_bins: int = 256):
        self._max_len = max_len
        self._n_bins = n_bins

        # Download base PaliGemma tokenizer
        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        self._fast_skip_tokens = 128  # Skip last 128 tokens in PaliGemma vocab since they are special tokens

    def tokenize(
        self, prompt: str, state: np.ndarray, actions: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Tokenize a prompt and state into a sequence of tokens.

        Args:
            prompt: The text prompt to tokenize.
            state: The state array to discretize and tokenize.
            actions: Must be None. Action encoding is not currently supported.

        Returns:
            A tuple of (tokens, token_mask, ar_mask, targets).

        Raises:
            NotImplementedError: If actions is not None.
        """
        cleaned_text = prompt.lower().strip().replace("_", " ")

        # Convention: state gets discretized into 256 discrete bins (assumed range after normalization: [-1, 1])
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        # Convention: prefix includes prompt and string-representation of state, followed by ';'
        state_str = " ".join(map(str, discretized_state))
        prefix = f"Task: {cleaned_text}, State: {state_str};\n"
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        if actions is not None:
            raise NotImplementedError("BinningTokenizer does not support encoding actions atm (only for inference use)")
        postfix_tokens = []

        # Create output token sequence & masks
        # AR mask is 0 on prefix (bidirectional attention) and 1 on postfix (causal attention to all previous tokens)
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)  # Loss on postfix only

        # Pad tokens to max length
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        return np.asarray(tokens), np.asarray(token_mask), np.asarray(ar_mask), np.asarray(loss_mask)

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        # Decode predicted output tokens
        decoded_tokens = self._paligemma_tokenizer.decode(tokens.tolist())

        # Extract actions from FAST model outputs
        if "Action: " not in decoded_tokens:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

        # Extract actions from decoded tokens
        raw_action_tokens = np.array(
            self._paligemma_tokenizer.encode(decoded_tokens.split("Action: ")[1].split("|")[0].strip())
        )
        action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)
        if len(action_tokens) < action_horizon * action_dim:
            return np.zeros([action_horizon, action_dim], dtype=np.float32)
        action_tokens = action_tokens[: (action_horizon * action_dim)].reshape([action_horizon, action_dim])
        return action_tokens / self._n_bins * 2 - 1

    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return self._paligemma_tokenizer.vocab_size() - 1 - self._fast_skip_tokens - tokens


class FSQTokenizer:
    """
    FSQ tokenizer from the FAST paper baselines.
    """

    def __init__(self, max_len: int = 256, fsq_tokenizer_path: str | None = None):
        self._max_len = max_len

        assert fsq_tokenizer_path is not None, "fsq_tokenizer_path must be provided"
        # Download tokenizer
        path = download.maybe_download(fsq_tokenizer_path)
        tok_path = os.path.join(path, os.listdir(path)[0])

        # Split step from path
        step = int(tok_path.split("/")[-1])
        base_path = tok_path.rsplit("/", 1)[0]

        mgr = ocp.CheckpointManager(
            base_path,
            item_handlers={
                "params": ocp.StandardCheckpointHandler(),
                "opt_state": ocp.StandardCheckpointHandler(),
                "config": ocp.JsonCheckpointHandler(),
            },
            options=ocp.CheckpointManagerOptions(max_to_keep=1),
        )

        try:
            restored = mgr.restore(
                step, args=ocp.args.Composite(config=ocp.args.JsonRestore(), params=ocp.args.StandardRestore())
            )
            config = restored["config"]
            self._params = restored["params"]
            self._fsq_tokenizer = fsq_tokenizer.FsqAttentionTokenizer(**config)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load FSQ tokenizer checkpoint from {fsq_tokenizer_path}. Error: {e!s}"
            ) from e

        # Compile tokenize and detokenize functions
        self._tokenize_fn = jax.jit(
            lambda params, x: self._fsq_tokenizer.apply({"params": params}, x, method=self._fsq_tokenizer.tokenize)
        )
        self._detokenize_fn = jax.jit(
            lambda params, x: self._fsq_tokenizer.apply({"params": params}, x, method=self._fsq_tokenizer.detokenize)
        )

        # Download base PaliGemma tokenizer
        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        self._fast_skip_tokens = 128  # Skip last 128 tokens in PaliGemma vocab since they are special tokens

    def tokenize(
        self, prompt: str, state: np.ndarray, actions: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cleaned_text = prompt.lower().strip().replace("_", " ")

        # Convention: state gets discretized into 256 discrete bins (assumed range after normalization: [-1, 1])
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        # Convention: prefix includes prompt and string-representation of state, followed by ';'
        state_str = " ".join(map(str, discretized_state))
        prefix = f"Task: {cleaned_text}, State: {state_str};\n"
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        if actions is not None:
            raise NotImplementedError("FSQTokenizer does not support encoding actions atm (only for inference use)")
        postfix_tokens = []

        # Create output token sequence & masks
        # AR mask is 0 on prefix (bidirectional attention) and 1 on postfix (causal attention to all previous tokens)
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)  # Loss on postfix only

        # Pad tokens to max length
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        return np.asarray(tokens), np.asarray(token_mask), np.asarray(ar_mask), np.asarray(loss_mask)

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        # Decode predicted output tokens
        decoded_tokens = self._paligemma_tokenizer.decode(tokens.tolist())

        # Extract actions from FAST model outputs
        if "Action: " not in decoded_tokens:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

        # Extract actions from decoded tokens
        raw_action_tokens = np.array(
            self._paligemma_tokenizer.encode(decoded_tokens.split("Action: ")[1].split("|")[0].strip())
        )
        action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)
        try:
            # Move computation to CPU and compile on-demand
            device = jax.devices("cpu")[0]
            with jax.default_device(device):
                detok_act = self._detokenize_fn(self._params, action_tokens[None, ...])[0]
            return detok_act[: action_horizon * action_dim].reshape([action_horizon, action_dim])
        except Exception as e:
            logging.warning(f"Error decoding FSQ: {e}")
            return np.zeros((action_horizon, action_dim))

    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return self._paligemma_tokenizer.vocab_size() - 1 - self._fast_skip_tokens - tokens
