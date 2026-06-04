"""Dump VLM-backbone prefix embeddings (the RL Token paper's z_{1:M}) from a frozen Pi0.5-CoT SteerVLA checkpoint."""
import dataclasses
import logging
import time

import etils.epath as epath
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import tqdm_loggable.auto as tqdm
import tyro

import openpi.models.model as _model
from openpi.models.pi0 import make_attn_mask
from openpi.models.pi0_cot import Pi0CoT
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.jax_distributed as jax_distributed
import openpi.training.sharding as sharding

DEFAULT_CHECKPOINT = (
    "gs://cat-logs/pi05_steervla_cot_simplified_reasoning/pi05_steervla_cot_simplified_reasoning/"
    "pi05_steervla_cot_simplified_reasoning_20260523_222304/20000"
)


@dataclasses.dataclass
class Args:
    config_name: str = "pi05_steervla_cot_simplified_reasoning_traffic_light_only"
    checkpoint_dir: str = DEFAULT_CHECKPOINT
    output_dir: str = "./rl_token_embeddings/traffic_light_v0"
    num_batches: int = 500
    split: str = "train"
    shuffle: bool = False
    save_metadata: bool = True
    save_images: bool = True
    jpeg_images: bool = True
    jpeg_quality: int = 95


def _prefix_forward(model: Pi0CoT, observation: _model.Observation):
    observation = _model.preprocess_observation(None, observation, train=False)
    img_tokens, img_masks, img_ar = model._embed_images(observation)
    n_img = sum(t.shape[1] for t in img_tokens)

    prompt = model._embed_text_tokens(observation.tokenized_prompt)
    reasoning = model._embed_text_tokens(observation.tokenized_reasoning)
    subtask = model._embed_text_tokens(observation.tokenized_subtask)
    n_prompt, n_reasoning, n_subtask = prompt.shape[1], reasoning.shape[1], subtask.shape[1]

    parts = list(img_tokens) + [prompt, reasoning, subtask]
    masks = list(img_masks) + [observation.tokenized_prompt_mask, observation.tokenized_reasoning_mask, observation.tokenized_subtask_mask]
    ar = list(img_ar) + [False] * n_prompt + [True] * n_reasoning + [True] * n_subtask

    n_fast = 0
    if model._use_fast_tokens and observation.tokenized_fast is not None:
        fast = model._embed_text_tokens(observation.tokenized_fast)
        n_fast = fast.shape[1]
        parts.append(fast)
        masks.append(observation.tokenized_fast_mask)
        ar += [True] * n_fast

    tokens = jnp.concatenate(parts, axis=1)
    prefix_mask = jnp.concatenate(masks, axis=1)
    attn_mask = make_attn_mask(prefix_mask, jnp.array(ar))
    if n_fast > 0:
        total = prefix_mask.shape[1]
        rstart, rend = n_img + n_prompt, n_img + n_prompt + n_reasoning
        fstart = rend + n_subtask
        col_r = (jnp.arange(total) >= rstart) & (jnp.arange(total) < rend)
        row_f = (jnp.arange(total) >= fstart) & (jnp.arange(total) < fstart + n_fast)
        attn_mask = attn_mask & (~(row_f[:, None] & col_r[None, :]))[None]
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    (prefix_out, _), _ = model.PaliGemma.llm([tokens, None], mask=attn_mask, positions=positions)
    sizes = jnp.array([n_img, n_prompt, n_reasoning, n_subtask, n_fast], jnp.int32)
    return prefix_out, prefix_mask, sizes


def _encode_jpeg(frame: np.ndarray, quality: int) -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _extract_metadata(obs: _model.Observation, *, save_images: bool, jpeg_images: bool, jpeg_quality: int) -> dict:
    """Per-row raw inputs aligned to the embeddings: tokenized text ids (+masks) and base image.

    Images are analysis-only (the AE trains on embeddings, not images). With ``jpeg_images`` we store
    each frame as a JPEG blob in a flat uint8 buffer + per-row lengths (no pickle, much smaller on disk).
    """
    def g(x):
        return None if x is None else np.asarray(jax.device_get(x))

    meta = {
        "tokenized_prompt": g(obs.tokenized_prompt),
        "tokenized_prompt_mask": g(obs.tokenized_prompt_mask),
        "tokenized_reasoning": g(obs.tokenized_reasoning),
        "tokenized_reasoning_mask": g(obs.tokenized_reasoning_mask),
        "tokenized_subtask": g(obs.tokenized_subtask),
        "tokenized_subtask_mask": g(obs.tokenized_subtask_mask),
        "tokenized_fast": g(obs.tokenized_fast),
        "tokenized_fast_mask": g(obs.tokenized_fast_mask),
    }
    if save_images and "base_0_rgb" in obs.images:
        # Loader images are float in [-1, 1]; convert to uint8 for viewable frames.
        img = np.asarray(jax.device_get(obs.images["base_0_rgb"])).astype(np.float32)
        img = np.clip((img + 1.0) * 127.5, 0, 255).astype(np.uint8)  # (B, H, W, 3)
        if jpeg_images:
            blobs = [_encode_jpeg(frame, jpeg_quality) for frame in img]
            meta["base_image_jpeg"] = np.frombuffer(b"".join(blobs), dtype=np.uint8).copy()
            meta["base_image_jpeg_lengths"] = np.asarray([len(b) for b in blobs], dtype=np.int64)
        else:
            meta["base_image"] = img
    return {k: v for k, v in meta.items() if v is not None}


def main(args: Args):
    jax_distributed.initialize_if_needed()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    logging.info("JAX devices=%d processes=%d", jax.device_count(), jax.process_count())

    config = _config.get_config(args.config_name)
    output_dir = epath.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    data_loader = _data_loader.create_data_loader(config, sharding=data_sharding, shuffle=args.shuffle, skip_norm_stats=True, split=args.split)
    data_iter = iter(data_loader)

    logging.info("Loading checkpoint %s", args.checkpoint_dir)
    t0 = time.monotonic()
    params = _model.restore_params(args.checkpoint_dir + "/params", dtype=jnp.bfloat16)
    model = config.model.load(params)
    if not isinstance(model, Pi0CoT):
        raise ValueError(f"Expected Pi0CoT, got {type(model).__name__}")
    logging.info("Model loaded in %.1fs", time.monotonic() - t0)

    graphdef, state = nnx.split(model)

    def _jit_fn(state, obs):
        m = nnx.merge(graphdef, state)
        m.eval()
        return _prefix_forward(m, obs)

    jitted = jax.jit(_jit_fn, in_shardings=(None, data_sharding))

    for i in tqdm.tqdm(range(args.num_batches), desc="dump"):
        try:
            obs, actions = next(data_iter)
        except StopIteration:
            logging.info("Iterator exhausted at batch %d", i)
            break

        prefix_out, prefix_mask, sizes = jitted(state, obs)
        out = np.asarray(jax.device_get(prefix_out.astype(jnp.float16)))
        mask = np.asarray(jax.device_get(prefix_mask))
        sizes = np.asarray(jax.device_get(sizes))
        acts = np.asarray(jax.device_get(actions.astype(jnp.float32)))

        # Drop always-masked wrist-camera tokens (SteerVLA uses only base_0_rgb).
        n_img = int(sizes[0])
        assert n_img % 3 == 0, f"n_img={n_img} not divisible by 3"
        per_cam = n_img // 3
        keep = np.r_[np.arange(per_cam), np.arange(3 * per_cam, out.shape[1])]

        payload = dict(
            prefix_out=out[:, keep, :],
            prefix_mask=mask[:, keep],
            actions=acts,
            tokens_per_cam=np.int32(per_cam),
            n_prompt=np.int32(sizes[1]),
            n_reasoning=np.int32(sizes[2]),
            n_subtask=np.int32(sizes[3]),
            n_fast=np.int32(sizes[4]),
        )
        if args.save_metadata:
            payload.update(_extract_metadata(
                obs, save_images=args.save_images, jpeg_images=args.jpeg_images, jpeg_quality=args.jpeg_quality
            ))
        np.savez(output_dir / f"shard_{i:06d}.npz", **payload)
        if i < 3:
            logging.info("batch %d shape=%s density=%.3f", i, out[:, keep, :].shape, float(mask[:, keep].mean()))

    logging.info("Done. Wrote shards to %s", output_dir)


if __name__ == "__main__":
    main(tyro.cli(Args))
