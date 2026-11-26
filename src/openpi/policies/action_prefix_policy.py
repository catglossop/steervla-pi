"""Policy that supports real-time chunking (RTC) for models trained with action-prefix conditioning.

A request may carry ``actions`` (the already-committed actions, padded to the chunk length, in the same raw space
as the dataset) and ``action_prefix_len``. They go through the usual input transforms (so they are normalized and
padded exactly like training targets) and are passed to ``sample_actions`` as a fixed prefix. Requests without a
prefix behave exactly like ``Policy.infer``.
"""

import jax
import jax.numpy as jnp
import numpy as np

from openpi.policies import policy as _policy


class ActionPrefixPolicy(_policy.Policy):
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        if "action_prefix_len" not in obs:
            return super().infer(obs, noise=noise)
        if self._is_pytorch_model:
            raise NotImplementedError("Action-prefix (RTC) inference is JAX-only.")
        transformed = self._input_transform(jax.tree.map(lambda x: x, obs))
        prefix = jnp.asarray(transformed["actions"])[None]
        prefix_len = jnp.asarray(transformed["action_prefix_len"]).reshape(1)
        plain_obs = {k: v for k, v in obs.items() if k not in ("actions", "action_prefix_len")}
        saved = self._sample_kwargs
        self._sample_kwargs = {**saved, "action_prefix": prefix, "action_prefix_len": prefix_len}
        try:
            return super().infer(plain_obs, noise=noise)
        finally:
            self._sample_kwargs = saved
