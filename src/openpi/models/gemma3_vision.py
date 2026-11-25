"""Gemma 3 vision tower: SigLIP-So400m/14 + Gemma 3's multimodal projector.

Same encoder as PaliGemma's (``siglip.Module(variant="So400m/14")``) but without the ``head`` Dense. Gemma 3
instead average-pools the patch grid down to ``tokens_per_side**2`` tokens, applies an RMSNorm
(``mm_soft_emb_norm``) and a bias-free linear projection into the LLM width. Matches
``transformers.models.gemma3.Gemma3MultiModalProjector``.

The call signature mirrors ``siglip.Module`` (``(tokens, aux)``), so models can swap it in for ``PaliGemma.img``.
"""

import flax.linen as nn
import jax.numpy as jnp

import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip

# Gemma 3 always emits 256 image tokens (16x16), whatever the input resolution.
TOKENS_PER_SIDE = 16
PATCH_SIZE = 14
NATIVE_RESOLUTION = 896


def check_resolution(resolution: int) -> None:
    if resolution % (PATCH_SIZE * TOKENS_PER_SIDE) != 0:
        raise ValueError(
            f"Gemma 3 image resolution must be a multiple of {PATCH_SIZE * TOKENS_PER_SIDE} "
            f"(patch {PATCH_SIZE} x {TOKENS_PER_SIDE} tokens per side), got {resolution}."
        )


class VisionTower(nn.Module):
    out_width: int  # LLM width the image tokens are projected into.
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, image, *, train=False):
        x, out = _siglip.Module(
            num_classes=None,
            variant="So400m/14",
            pool_type="none",
            scan=True,
            dtype_mm=self.dtype_mm,
            name="siglip",
        )(image, train=train)

        n, length, c = x.shape
        side = round(length**0.5)
        assert side * side == length, f"non-square patch grid: {length} tokens"
        k = side // TOKENS_PER_SIDE
        # Row-major patch grid -> k x k average pooling -> (n, 256, c), as in Gemma3MultiModalProjector.
        x = x.reshape(n, TOKENS_PER_SIDE, k, TOKENS_PER_SIDE, k, c).mean(axis=(2, 4))
        x = x.reshape(n, TOKENS_PER_SIDE * TOKENS_PER_SIDE, c)

        x, _ = _gemma.RMSNorm(name="mm_soft_emb_norm")(x, None)
        projection = self.param(
            "mm_input_projection", nn.initializers.lecun_normal(in_axis=-2, out_axis=-1), (c, self.out_width)
        )
        x = jnp.dot(x, projection.astype(x.dtype))
        out["projected"] = x
        return x.astype(self.dtype_mm), out
