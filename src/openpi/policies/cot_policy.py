"""Policy wrappers for chain-of-thought models (``Pi0CoT``).

Kept separate from :mod:`openpi.policies.policy` so upstream's ``Policy`` stays untouched.
:func:`openpi.policies.policy_config.create_trained_policy` returns a :class:`CoTPolicy` whenever
the model implements ``sample_cot``.
"""

import pathlib
import time
from typing import Any

import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.policies import policy as _policy
from openpi.shared import nnx_utils


def _hashable_image_keys(kwargs: dict[str, Any] | None) -> dict[str, Any]:
    # ``image_keys`` is a static jit argument, so it must be hashable (tuple, not list).
    kwargs = dict(kwargs or {})
    if kwargs.get("image_keys") is not None:
        kwargs["image_keys"] = tuple(kwargs["image_keys"])
    return kwargs


class CoTPolicy(_policy.Policy):
    """``Policy`` plus :meth:`infer_with_cot`: sample reasoning + subtask, then actions.

    :meth:`infer` (inherited) runs only the action expert, conditioning on whatever
    reasoning/subtask tokens the input transforms produced (e.g. teacher-forced ground truth).
    """

    def __init__(self, model: _model.BaseModel, *, cot_sample_kwargs: dict[str, Any] | None = None, **kwargs):
        kwargs["sample_kwargs"] = _hashable_image_keys(kwargs.get("sample_kwargs"))
        super().__init__(model, **kwargs)
        self._cot_sample_kwargs = _hashable_image_keys(cot_sample_kwargs)
        if not self._is_pytorch_model:
            # Pi0CoT.sample_actions unrolls a Python loop over ``num_steps`` and selects cameras by
            # ``image_keys``, so both must be compile-time constants.
            self._sample_actions = nnx_utils.module_jit(
                model.sample_actions, static_argnames=("num_steps", "image_keys")
            )
        # Eager: sample_cot jits its own per-token kernels instead of compiling the whole decode loop.
        self._sample_cot = model.sample_cot

    def infer_with_cot(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:
        """Run ``sample_cot`` then ``sample_actions`` on a single (unbatched) observation.

        Requires ``tokenized_prompt`` / ``tokenized_prompt_mask`` in the transformed inputs (same as
        training). Returns actions plus the sampled reasoning/subtask (and FAST, if enabled) token
        buffers and per-phase timings. JAX only.
        """
        if self._is_pytorch_model:
            raise NotImplementedError("infer_with_cot is only implemented for JAX models.")

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

        observation = _model.Observation.from_dict(inputs)

        t_cot = time.monotonic()
        cot_out = self._sample_cot(rng_cot, observation, **self._cot_sample_kwargs)
        cot_ms = (time.monotonic() - t_cot) * 1000

        observation = observation.replace(
            tokenized_reasoning=cot_out["tokenized_reasoning"],
            tokenized_reasoning_mask=cot_out["tokenized_reasoning_mask"],
            tokenized_subtask=cot_out["tokenized_subtask"],
            tokenized_subtask_mask=cot_out["tokenized_subtask_mask"],
        )
        if "tokenized_fast" in cot_out and "tokenized_fast_mask" in cot_out:
            observation = observation.replace(
                tokenized_fast=cot_out["tokenized_fast"],
                tokenized_fast_mask=cot_out["tokenized_fast_mask"],
            )

        t_act = time.monotonic()
        actions = self._sample_actions(rng_act, observation, **sample_kwargs)
        act_ms = (time.monotonic() - t_act) * 1000

        def _batch0(x: Any) -> Any:
            return np.asarray(x[0, ...])

        # Only state/actions go through output transforms (e.g. Unnormalize with strict norm_stats keys).
        outputs = self._output_transform({"state": _batch0(inputs["state"]), "actions": _batch0(actions)})
        for key in ("tokenized_reasoning", "tokenized_reasoning_mask", "tokenized_subtask", "tokenized_subtask_mask"):
            outputs[key] = _batch0(cot_out[key])
        if "tokenized_fast" in cot_out:
            outputs["tokenized_fast"] = _batch0(cot_out["tokenized_fast"])
            outputs["tokenized_fast_mask"] = _batch0(cot_out["tokenized_fast_mask"])
        outputs["policy_timing"] = {
            "infer_cot_ms": cot_ms,
            "infer_actions_ms": act_ms,
            "infer_ms": cot_ms + act_ms,
        }
        return outputs


class CoTPolicyRecorder(_policy.PolicyRecorder):
    """``PolicyRecorder`` that also records :meth:`CoTPolicy.infer_with_cot` calls."""

    def __init__(self, policy: CoTPolicy, record_dir: str):
        super().__init__(policy, record_dir)

    def infer_with_cot(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:
        results = self._policy.infer_with_cot(obs, noise=noise)
        data = flax.traverse_util.flatten_dict({"inputs": obs, "outputs": results}, sep="/")
        output_path = pathlib.Path(self._record_dir) / f"step_{self._record_step}"
        self._record_step += 1
        np.save(output_path, np.asarray(data))
        return results
