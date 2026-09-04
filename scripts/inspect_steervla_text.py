"""Sample examples from the SteerVLA RLDS datasets and show every text field.

Two views, either of which can be skipped:

1. RAW   -- N frames pulled straight out of each TFDS dataset under ``--data-dir``.
            Every ``Text`` feature declared in that dataset's ``features.json`` is printed
            verbatim, so datasets with different schemas each show whatever they carry.

2. BATCH -- the same data after the *full* training transform stack of a ``TrainConfig``
            (repack -> data transforms -> Normalize -> model transforms), i.e. exactly the
            dict ``train.py`` turns into an ``Observation``. By that point the text has become
            token ids, so the ids are decoded back to text to show what the model actually reads,
            including the CoT delimiters and the ``Prompt:...;State:...;`` framing.

Usage:

    uv run --group rlds scripts/inspect_steervla_text.py
    uv run --group rlds scripts/inspect_steervla_text.py --datasets simlingo_dataset_all_img512_1116
    uv run --group rlds scripts/inspect_steervla_text.py --skip-raw --batch-config pi05_steervla_cot_simplified_reasoning
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import logging
import pathlib
import random
import sys
import textwrap

DEFAULT_DATA_DIR = "/raid/datasets/steervla"
DEFAULT_BATCH_CONFIG = "pi05_steervla_cot_simplified_reasoning_csp"

# ---------------------------------------------------------------------------- helpers


def _hr(title: str, char: str = "=") -> str:
    return f"\n{char * 100}\n{title}\n{char * 100}"


def _wrap(text: str, indent: str = "      ") -> str:
    if not text:
        return f"{indent}<empty>"
    return "\n".join(
        textwrap.fill(line, width=110, initial_indent=indent, subsequent_indent=indent) or indent
        for line in text.splitlines()
    )


def _as_str(value) -> str:
    """bytes / np.bytes_ / tf tensor -> str."""
    if hasattr(value, "numpy"):
        value = value.numpy()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _jpeg_data_uri(image) -> str | None:
    """Encode a uint8 HWC array or raw JPEG bytes as a data: URI for the HTML report."""
    if image is None:
        return None
    if hasattr(image, "numpy"):
        image = image.numpy()
    if isinstance(image, bytes):
        return "data:image/jpeg;base64," + base64.b64encode(image).decode("ascii")
    try:
        import numpy as np
        from PIL import Image

        arr = np.asarray(image)
        if arr.dtype != np.uint8:  # e.g. the [-1, 1] float image out of the model transforms
            arr = np.clip((arr + 1.0) * 127.5, 0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG", quality=85)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


# ---------------------------------------------------------------------------- raw view


def discover_datasets(data_dir: pathlib.Path) -> list[tuple[str, pathlib.Path]]:
    """Every ``<name>/<version>/features.json`` under ``data_dir``, sorted by name."""
    found: dict[str, pathlib.Path] = {}
    for features in sorted(data_dir.glob("*/*/features.json")):
        found.setdefault(features.parent.parent.name, features)
    return sorted(found.items())


def text_fields(features_path: pathlib.Path) -> dict[str, list[str]]:
    """Text feature paths in a features.json, grouped by top-level section."""
    spec = json.loads(features_path.read_text())
    out: dict[str, list[str]] = {}

    def walk(node, path: list[str], section: str | None):
        fdict = node.get("featuresDict", {}).get("features")
        if fdict:
            for key, child in fdict.items():
                walk(child, [*path, key], section or key)
            return
        seq = node.get("sequence", {}).get("feature")
        if seq:
            walk(seq, path, section)
            return
        if node.get("pythonClassName", "").endswith("text_feature.Text"):
            out.setdefault(section or "", []).append("/".join(path[1:]))

    walk(spec, [], None)
    return out


def _get_path(mapping, path: str):
    node = mapping
    for key in path.split("/"):
        if key not in node:
            return None
        node = node[key]
    return node


def sample_raw(
    name: str,
    features_path: pathlib.Path,
    data_dir: pathlib.Path,
    *,
    num_samples: int,
    split: str,
    seed: int,
) -> list[dict]:
    """Pull ``num_samples`` frames, each from a different episode, out of one TFDS dataset."""
    import tensorflow_datasets as tfds

    fields = text_fields(features_path)
    step_fields = fields.get("steps", [])
    episode_fields = fields.get("episode_metadata", [])

    builder = tfds.builder(name, data_dir=str(data_dir), version=features_path.parent.name)
    # Skip JPEG decode: the raw bytes go straight into the HTML report and decoding 512x512
    # images we mostly do not look at is the slowest part of this script.
    decoders = {"steps": {"observation": {"image": tfds.decode.SkipDecoding()}}}
    try:
        episodes = builder.as_dataset(split=split, shuffle_files=True, decoders=decoders)
    except Exception:
        episodes = builder.as_dataset(split=split, shuffle_files=True)

    rng = random.Random(seed)
    samples: list[dict] = []
    # Over-take episodes so a short/empty one does not cost us a sample.
    for episode in episodes.take(num_samples * 3):
        if len(samples) >= num_samples:
            break
        steps = list(episode["steps"].take(rng.randint(1, 8)))
        if not steps:
            continue
        step = steps[-1]

        record = {
            "episode": {f: _as_str(_get_path(episode.get("episode_metadata", {}), f)) for f in episode_fields},
            "text": {f: _as_str(_get_path(step, f)) for f in step_fields},
            "image": None,
            "scalars": {},
        }
        image = _get_path(step, "observation/image")
        if image is not None:
            record["image"] = image.numpy() if hasattr(image, "numpy") else image
        for key in ("speed", "global_course", "local_course", "timestamp"):
            if key in step:
                record["scalars"][key] = float(step[key].numpy())
        samples.append(record)

    return samples


def print_raw(name: str, samples: list[dict]) -> None:
    print(_hr(f"RAW DATASET: {name}   ({len(samples)} samples)"))
    if not samples:
        print("  (no samples -- empty split?)")
        return
    for i, sample in enumerate(samples):
        print(f"\n  --- sample {i} " + "-" * 82)
        for key, value in sample["episode"].items():
            print(f"    [episode_metadata/{key}]\n{_wrap(value)}")
        if sample["scalars"]:
            scalars = "  ".join(f"{k}={v:.3f}" for k, v in sample["scalars"].items())
            print(f"    [scalars] {scalars}")
        for key, value in sample["text"].items():
            print(f"    [{key}]  ({len(value)} chars)\n{_wrap(value)}")


# ---------------------------------------------------------------------------- batch view


def build_token_decoder(tokenizer):
    """Return ``ids -> str`` that renders CoT delimiter ids as named tags."""
    specials = {
        tokenizer._start_of_reasoning(): "<start_of_reasoning>",  # noqa: SLF001
        tokenizer._end_of_reasoning(): "<end_of_reasoning>",  # noqa: SLF001
        tokenizer._start_of_subtask(): "<start_of_subtask>",  # noqa: SLF001
        tokenizer._end_of_subtask(): "<end_of_subtask>",  # noqa: SLF001
        tokenizer._tokenizer.eos_id(): "<eos>",  # noqa: SLF001
        tokenizer._tokenizer.bos_id(): "<bos>",  # noqa: SLF001
    }

    def decode(ids) -> str:
        pieces: list[str] = []
        run: list[int] = []
        for tid in (int(t) for t in ids):
            if tid in specials:
                if run:
                    pieces.append(tokenizer._tokenizer.decode(run))  # noqa: SLF001
                    run = []
                pieces.append(specials[tid])
            else:
                run.append(tid)
        if run:
            pieces.append(tokenizer._tokenizer.decode(run))  # noqa: SLF001
        return "".join(pieces)

    return decode


def sample_batch(config_name: str, *, batch_size: int, split: str, shuffle: bool, skip_norm_stats: bool | None):
    """One batch straight out of the training pipeline, plus the metadata needed to read it."""
    from openpi.training import config as _config
    from openpi.training import data_loader as _data_loader

    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    rlds = _data_loader.create_rlds_dataset(
        data_config,
        action_horizon=config.model.action_horizon,
        batch_size=batch_size,
        shuffle=shuffle,
        split=split,
    )
    dataset_names = getattr(rlds, "dataset_names", [])

    transformed = _data_loader.transform_iterable_dataset(
        rlds,
        data_config,
        skip_norm_stats=config.skip_norm_stats if skip_norm_stats is None else skip_norm_stats,
        is_batched=True,
    )
    batch = next(iter(transformed))
    return config, data_config, batch, dataset_names


def print_batch(config, data_config, batch, dataset_names, *, num_samples: int, decode) -> None:
    import numpy as np

    from openpi.models import model as _model

    print(_hr(f"TRAINING BATCH: config={config.name}"))
    print("\n  Batch keys (this dict is what `Observation.from_dict` consumes in train.py):")
    for key in sorted(batch):
        value = batch[key]
        if isinstance(value, dict):
            for sub in sorted(value):
                arr = np.asarray(value[sub])
                print(f"    {key}/{sub:<22} {arr.dtype!s:<8} {arr.shape}")
        else:
            arr = np.asarray(value)
            print(f"    {key:<30} {arr.dtype!s:<8} {arr.shape}")

    observation = _model.Observation.from_dict(batch)
    print(f"\n  Observation.from_dict OK -- images={sorted(observation.images)}")
    print("  NOTE: from_dict rewrites `image` in place, uint8 -> float32 in [-1, 1]; the per-element")
    print("        image lines below therefore show the post-Observation dtype, not the table above.")

    n = min(num_samples, int(np.asarray(batch["actions"]).shape[0]))
    for i in range(n):
        print(f"\n  --- batch element {i} " + "-" * 76)

        ds_id = int(np.asarray(batch["dataset_id"])[i]) if "dataset_id" in batch else None
        if ds_id is not None:
            source = dataset_names[ds_id] if ds_id < len(dataset_names) else f"<id {ds_id}>"
            print(f"    source dataset : [{ds_id}] {source}")

        for field, mask_field in (
            ("tokenized_prompt", "tokenized_prompt_mask"),
            ("tokenized_reasoning", "tokenized_reasoning_mask"),
            ("tokenized_subtask", "tokenized_subtask_mask"),
            ("tokenized_fast", "tokenized_fast_mask"),
        ):
            if field not in batch:
                continue
            ids = np.asarray(batch[field])[i]
            mask = np.asarray(batch[mask_field])[i].astype(bool)
            used = ids[mask]
            print(f"\n    [{field}]  {int(mask.sum())}/{len(ids)} tokens used (rest is padding)")
            print(_wrap(decode(used)))
            print(f"      ids: {used.tolist()}")

        if "state" in batch:
            state = np.asarray(batch["state"])[i]
            nonzero = int(np.count_nonzero(state))
            print(f"\n    [state] dim={state.shape[0]} ({nonzero} non-zero, rest is action_dim padding)")
            print(f"      {np.array2string(state[:8], precision=4)} ...")

        actions = np.asarray(batch["actions"])[i]
        print(f"    [actions] {actions.shape}  range=[{actions.min():.3f}, {actions.max():.3f}]")
        if "action_loss_mask" in batch:
            alm = np.asarray(batch["action_loss_mask"])[i]
            print(f"    [action_loss_mask] {alm.shape} all={bool(alm.all())} any={bool(alm.any())}")
        if "cot_loss_mask" in batch:
            print(f"    [cot_loss_mask] {bool(np.asarray(batch['cot_loss_mask'])[i])}")
        for cam in sorted(batch["image"]):
            img = np.asarray(batch["image"][cam])[i]
            keep = bool(np.asarray(batch["image_mask"][cam])[i])
            print(f"    [image/{cam}] {img.shape} {img.dtype} range=[{img.min():.2f}, {img.max():.2f}] mask={keep}")


# ---------------------------------------------------------------------------- html report


def write_report(path: pathlib.Path, raw_sections, batch_section) -> None:
    def esc(text) -> str:
        return html.escape(str(text))

    parts = [
        "<!doctype html><meta charset='utf-8'><title>SteerVLA dataset text fields</title>",
        "<style>",
        "body{font:14px/1.5 system-ui,sans-serif;margin:24px;background:#fafafa;color:#111}",
        "h1{font-size:22px}h2{font-size:17px;margin-top:34px;border-bottom:2px solid #ddd;padding-bottom:4px}",
        ".sample{display:flex;gap:16px;border:1px solid #ddd;background:#fff;border-radius:8px;",
        "padding:12px;margin:12px 0;align-items:flex-start}",
        ".sample img{width:260px;border-radius:6px;flex:none}",
        ".fields{flex:1;min-width:0}",
        ".f{margin-bottom:10px}.k{font:12px ui-monospace,monospace;color:#0a58ca;font-weight:600}",
        ".v{white-space:pre-wrap;word-break:break-word;background:#f6f8fa;border-radius:4px;padding:6px 8px}",
        ".meta{font:12px ui-monospace,monospace;color:#555;margin-bottom:8px}",
        "</style>",
        "<h1>SteerVLA dataset text fields</h1>",
    ]

    for name, samples in raw_sections:
        parts.append(f"<h2>{esc(name)} &mdash; {len(samples)} samples</h2>")
        for i, sample in enumerate(samples):
            uri = _jpeg_data_uri(sample["image"])
            parts.append("<div class='sample'>")
            if uri:
                parts.append(f"<img src='{uri}' alt='frame'>")
            parts.append("<div class='fields'>")
            meta = [f"sample {i}"]
            meta += [f"{k}={v:.3f}" for k, v in sample["scalars"].items()]
            parts.append(f"<div class='meta'>{esc('  |  '.join(meta))}</div>")
            for key, value in {**sample["episode"], **sample["text"]}.items():
                parts.append(f"<div class='f'><div class='k'>{esc(key)}</div><div class='v'>{esc(value)}</div></div>")
            parts.append("</div></div>")

    if batch_section:
        config_name, elements = batch_section
        parts.append(f"<h2>Training batch &mdash; config <code>{esc(config_name)}</code></h2>")
        for i, element in enumerate(elements):
            parts.append("<div class='sample'>")
            if element["image_uri"]:
                parts.append(f"<img src='{element['image_uri']}' alt='batch image'>")
            parts.append("<div class='fields'>")
            meta = f"batch element {i}  |  {element['source']}"
            parts.append(f"<div class='meta'>{esc(meta)}</div>")
            for key, value in element["text"].items():
                parts.append(f"<div class='f'><div class='k'>{esc(key)}</div><div class='v'>{esc(value)}</div></div>")
            parts.append("</div></div>")

    path.write_text("".join(parts))
    print(f"\nWrote HTML report -> {path}")


# ---------------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, type=pathlib.Path)
    parser.add_argument("--num-samples", type=int, default=5, help="Samples per dataset / batch elements to show.")
    parser.add_argument("--datasets", nargs="*", default=None, help="Restrict to these dataset names.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-raw", action="store_true")
    parser.add_argument("--skip-batch", action="store_true")
    parser.add_argument("--batch-config", default=DEFAULT_BATCH_CONFIG)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--skip-norm-stats",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Override the config's skip_norm_stats when building the batch.",
    )
    parser.add_argument("--report", type=pathlib.Path, default=None, help="Write an HTML report here.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    raw_sections: list[tuple[str, list[dict]]] = []
    if not args.skip_raw:
        available = discover_datasets(args.data_dir)
        if args.datasets:
            wanted = set(args.datasets)
            available = [(n, p) for n, p in available if n in wanted]
            missing = wanted - {n for n, _ in available}
            if missing:
                print(f"WARNING: not found under {args.data_dir}: {sorted(missing)}", file=sys.stderr)
        if not available:
            print(f"No TFDS datasets found under {args.data_dir}", file=sys.stderr)
            return 1

        print(f"Found {len(available)} datasets under {args.data_dir}")
        for name, features_path in available:
            try:
                samples = sample_raw(
                    name,
                    features_path,
                    args.data_dir,
                    num_samples=args.num_samples,
                    split=args.split,
                    seed=args.seed,
                )
            except Exception as exc:
                print(f"\nERROR sampling {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
                continue
            print_raw(name, samples)
            raw_sections.append((name, samples))

    batch_section = None
    if not args.skip_batch:
        import numpy as np

        config, data_config, batch, dataset_names = sample_batch(
            args.batch_config,
            batch_size=args.batch_size,
            split=args.split,
            shuffle=True,
            skip_norm_stats=args.skip_norm_stats,
        )
        from openpi.models.tokenizer import CoTPaligemmaTokenizer

        tokenizer = CoTPaligemmaTokenizer()
        decode = build_token_decoder(tokenizer)
        print_batch(config, data_config, batch, dataset_names, num_samples=args.num_samples, decode=decode)

        elements = []
        for i in range(min(args.num_samples, int(np.asarray(batch["actions"]).shape[0]))):
            ds_id = int(np.asarray(batch["dataset_id"])[i]) if "dataset_id" in batch else -1
            source = dataset_names[ds_id] if 0 <= ds_id < len(dataset_names) else f"dataset_id={ds_id}"
            text = {}
            for field, mask_field in (
                ("tokenized_prompt", "tokenized_prompt_mask"),
                ("tokenized_reasoning", "tokenized_reasoning_mask"),
                ("tokenized_subtask", "tokenized_subtask_mask"),
                ("tokenized_fast", "tokenized_fast_mask"),
            ):
                if field in batch:
                    ids = np.asarray(batch[field])[i]
                    mask = np.asarray(batch[mask_field])[i].astype(bool)
                    text[f"{field} ({int(mask.sum())}/{len(ids)} tokens)"] = decode(ids[mask])
            cam = sorted(batch["image"])[0]
            elements.append(
                {
                    "source": f"source: {source}",
                    "text": text,
                    "image_uri": _jpeg_data_uri(np.asarray(batch["image"][cam])[i]),
                }
            )
        batch_section = (args.batch_config, elements)

    if args.report:
        write_report(args.report, raw_sections, batch_section)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
