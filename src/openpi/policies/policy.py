from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, Callable, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        cot_sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            cot_sample_kwargs: Keyword arguments for :meth:`infer_with_cot` (passed to
                ``model.sample_cot``), e.g. ``temperature`` or ``image_keys``.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._cot_sample_kwargs = cot_sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
            self._sample_cot: Callable[..., Any] | None = getattr(model, "sample_cot", None)
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            # Eager CoT sampling avoids compiling the full autoregressive graph (see Pi0-CoT / SteerVLA notes).
            self._sample_cot = getattr(model, "sample_cot", None)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    def infer_with_cot(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        """Run ``sample_cot`` then ``sample_actions`` (Pi0-CoT and similar models).

        Requires ``tokenized_prompt`` / ``tokenized_prompt_mask`` in the transformed inputs
        (same as training). Returns sampled reasoning/subtask token buffers plus actions.

        ``cot_sample_kwargs`` can include ``timing=True`` (and ``timing_per_step=True``,
        ``timing_sync=True``) to print ``[Pi0CoT.sample_cot timing]`` breakdowns to stdout
        (e.g. in Jupyter).

        JAX only; PyTorch models raise :class:`NotImplementedError`.
        """
        if self._is_pytorch_model:
            raise NotImplementedError("infer_with_cot is only implemented for JAX models with sample_cot.")
        if self._sample_cot is None:
            raise TypeError(
                "infer_with_cot requires a model implementing sample_cot (e.g. Pi0CoT). "
                f"Got {type(self._model).__name__}."
            )

        print(f"Inferring with cot for obs: {obs}")
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        self._rng, rng_cot, rng_act = jax.random.split(self._rng, 3)

        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = jnp.asarray(noise)
            if noise.ndim == 2:
                noise = noise[None, ...]
            sample_kwargs["noise"] = noise
        
        print(f"Inputs: {inputs}")

        observation = _model.Observation.from_dict(inputs)
        cot_kwargs = dict(self._cot_sample_kwargs)
        print(f"Sampling cot with kwargs: {cot_kwargs}")

        t_cot = time.monotonic()
        print(f"Sampling cot...")
        cot_out = self._sample_cot(rng_cot, observation, **cot_kwargs)
        cot_ms = (time.monotonic() - t_cot) * 1000

        observation = observation.replace(
            tokenized_reasoning=cot_out["tokenized_reasoning"],
            tokenized_reasoning_mask=cot_out["tokenized_reasoning_mask"],
            tokenized_subtask=cot_out["tokenized_subtask"],
            tokenized_subtask_mask=cot_out["tokenized_subtask_mask"],
        )

        t_act = time.monotonic()
        print(f"Sampling actions...")
        actions = self._sample_actions(rng_act, observation, **sample_kwargs)
        act_ms = (time.monotonic() - t_act) * 1000

        def _batch0(x: Any) -> Any:
            return np.asarray(x[0, ...])

        # Only state/actions go through output transforms (e.g. Unnormalize with strict norm_stats keys).
        outputs = self._output_transform({"state": _batch0(inputs["state"]), "actions": _batch0(actions)})
        outputs["tokenized_reasoning"] = _batch0(cot_out["tokenized_reasoning"])
        outputs["tokenized_reasoning_mask"] = _batch0(cot_out["tokenized_reasoning_mask"])
        outputs["tokenized_subtask"] = _batch0(cot_out["tokenized_subtask"])
        outputs["tokenized_subtask_mask"] = _batch0(cot_out["tokenized_subtask_mask"])
        outputs["policy_timing"] = {
            "infer_cot_ms": cot_ms,
            "infer_actions_ms": act_ms,
            "infer_ms": cot_ms + act_ms,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results

    def infer_with_cot(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        infer_fn = getattr(self._policy, "infer_with_cot", None)
        if infer_fn is None:
            raise TypeError("Wrapped policy has no infer_with_cot (expected openpi.policies.policy.Policy).")
        results = infer_fn(obs, noise=noise)
        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")
        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1
        np.save(output_path, np.asarray(data))
        return results
