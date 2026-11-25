import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


@dataclasses.dataclass(frozen=True)
class Gemma3WeightLoader(WeightLoader):
    """Loads the Gemma 3 VLM (language model + vision tower) from a Hugging Face checkpoint.

    Like ``PaliGemmaWeightLoader``, every parameter the checkpoint lacks (the action expert, action/time
    projections) keeps its fresh initialization. The vision position embedding is resized to the model's
    configured image resolution; any other shape mismatch is an error.
    """

    # Hugging Face repo id (gated; needs HF_TOKEN) or a local directory holding the safetensors checkpoint.
    repo_or_dir: str = "google/gemma-3-4b-pt"

    def load(self, params: at.Params) -> at.Params:
        import openpi.models.gemma3_weights as gemma3_weights

        flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
        loaded = gemma3_weights.load_gemma3_params(self.repo_or_dir)
        posemb_key = "PaliGemma/img/siglip/pos_embedding"
        if posemb_key in loaded and posemb_key in flat_ref:
            loaded[posemb_key] = gemma3_weights.resize_posemb(loaded[posemb_key], flat_ref[posemb_key].shape[1])
        if missing := sorted(k for k in loaded if k not in flat_ref):
            raise ValueError(f"Gemma 3 weights have no place in this model (wrong backbone?): {missing[:5]}")
        if mismatched := [(k, v.shape, flat_ref[k].shape) for k, v in loaded.items() if v.shape != flat_ref[k].shape]:
            raise ValueError(f"Gemma 3 weight shape mismatch: {mismatched[:5]}")
        logger.info(
            "Loaded %d Gemma 3 tensors; %d params keep their initialization.", len(loaded), len(flat_ref) - len(loaded)
        )
        return _merge_params(flax.traverse_util.unflatten_dict(loaded, sep="/"), params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")
