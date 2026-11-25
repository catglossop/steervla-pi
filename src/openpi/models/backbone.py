"""VLM backbone selection: PaliGemma (default) or Gemma 3 4B.

Model configs carry ``backbone`` and ``image_resolution``; everything backbone-specific that the models and the
model transforms need (image encoder, input resolution, tokenizers, variant checks) is resolved here, so the
PaliGemma code paths stay exactly as upstream builds them.
"""

from typing import Any, Literal

import flax.linen as nn

Backbone = Literal["paligemma", "gemma3"]

DEFAULT_RESOLUTION: dict[str, int] = {"paligemma": 224, "gemma3": 896}
# LLM variants (gemma.py / gemma_fast.py) and action-expert variants that belong to each backbone.
_GEMMA3_VARIANTS = ("gemma3_4b", "dummy_gemma3")
_GEMMA3_EXPERT_VARIANTS = ("gemma3_expert_1024", "dummy_gemma3")


def resolution(config: Any) -> tuple[int, int]:
    """(height, width) the model consumes, from ``config.image_resolution`` or the backbone default."""
    backbone = getattr(config, "backbone", "paligemma")
    side = getattr(config, "image_resolution", None) or DEFAULT_RESOLUTION[backbone]
    return side, side


def validate(config: Any, *, vlm_variant: str, expert_variant: str | None = None) -> None:
    """Fail early when the variants don't belong to the selected backbone."""
    backbone = getattr(config, "backbone", "paligemma")
    if backbone not in DEFAULT_RESOLUTION:
        raise ValueError(f"Unknown backbone {backbone!r}; expected one of {sorted(DEFAULT_RESOLUTION)}")
    is_gemma3 = backbone == "gemma3"
    if (vlm_variant in _GEMMA3_VARIANTS) != is_gemma3:
        raise ValueError(
            f"backbone={backbone!r} does not match paligemma_variant={vlm_variant!r}. "
            f"Gemma 3 needs paligemma_variant in {_GEMMA3_VARIANTS}; PaliGemma needs a gemma_* variant."
        )
    if expert_variant is not None and (expert_variant in _GEMMA3_EXPERT_VARIANTS) != is_gemma3:
        raise ValueError(
            f"backbone={backbone!r} does not match action_expert_variant={expert_variant!r}. The action expert "
            f"shares every attention layer with the VLM, so Gemma 3 needs one of {_GEMMA3_EXPERT_VARIANTS}."
        )
    if is_gemma3:
        from openpi.models import gemma3_vision

        gemma3_vision.check_resolution(resolution(config)[0])


def image_encoder(config: Any, *, llm_width: int) -> nn.Module:
    """The ``PaliGemma.img`` module: SigLIP + linear head (PaliGemma) or SigLIP + Gemma 3 projector."""
    if getattr(config, "backbone", "paligemma") == "gemma3":
        from openpi.models import gemma3_vision

        return gemma3_vision.VisionTower(out_width=llm_width, dtype_mm=config.dtype)

    from openpi.models import siglip

    return siglip.Module(
        num_classes=llm_width,
        variant="So400m/14",
        pool_type="none",
        scan=True,
        dtype_mm=config.dtype,
    )


def prompt_tokenizer(config: Any, max_len: int):
    from openpi.models import tokenizer

    if getattr(config, "backbone", "paligemma") == "gemma3":
        return tokenizer.Gemma3Tokenizer(max_len)
    return tokenizer.PaligemmaTokenizer(max_len)


def fast_tokenizer_cls(config: Any):
    from openpi.models import tokenizer

    if getattr(config, "backbone", "paligemma") == "gemma3":
        return tokenizer.Gemma3FASTTokenizer
    return tokenizer.FASTTokenizer
