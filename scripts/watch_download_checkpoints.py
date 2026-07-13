#!/usr/bin/env python3
"""Poll GCS for new training checkpoints and download them into the OpenPI cache.

Uses :func:`openpi.shared.download.maybe_download`, which stores data under
``~/.cache/openpi/<bucket>/<path>`` and skips paths that already exist locally.

Example (background):

    cd /home/carla/steervla-pi
    nohup uv run scripts/watch_download_checkpoints.py \\
        > /home/carla/logs/watch_checkpoints.log 2>&1 &
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib
import re
import shutil
import time
import urllib.parse

import fsspec
import tyro

from openpi.shared import download as openpi_download

logger = logging.getLogger(__name__)

_STEP_DIR_RE = re.compile(r"^\d+$")
_COMPLETE_MARKER = "commit_success.txt"


@dataclasses.dataclass
class Args:
    """Watch a GCS run directory and download new step checkpoints as they appear."""

    run_url: str = (
        "gs://cat-logs/pi05_steervla_cot_simplified_reasoning/"
        "pi05_steervla_cot_simplified_reasoning/"
        "pi05_steervla_cot_simplified_reasoning_20260523_222304"
    )
    """GCS prefix containing numeric step subdirectories (2000, 4000, ...)."""

    poll_interval_sec: float = 300.0
    """Seconds to sleep between remote listing attempts."""

    min_step: int = 4000
    """Only download checkpoints with step >= this value."""

    step_stride: int = 1
    """Only download steps where ``(step - min_step) % step_stride == 0``."""

    once: bool = False
    """Exit after one poll instead of looping forever."""


def local_checkpoint_path(remote_step_url: str) -> pathlib.Path:
    """Mirror :func:`openpi.shared.download.maybe_download` cache layout."""
    parsed = urllib.parse.urlparse(remote_step_url)
    cache_dir = openpi_download.get_cache_dir()
    return cache_dir / parsed.netloc / parsed.path.strip("/")


def is_checkpoint_complete(local_path: pathlib.Path) -> bool:
    return (local_path / _COMPLETE_MARKER).is_file()


def list_remote_steps(run_url: str) -> list[int]:
    fs, path = fsspec.core.url_to_fs(run_url)
    try:
        entries = fs.ls(path.rstrip("/"))
    except FileNotFoundError:
        return []

    steps: list[int] = []
    for entry in entries:
        name = str(entry).rstrip("/").split("/")[-1]
        if _STEP_DIR_RE.match(name):
            steps.append(int(name))
    return sorted(steps)


def cleanup_incomplete(local_path: pathlib.Path) -> None:
    """Remove a partial local checkpoint so maybe_download can retry."""
    lock_path = local_path.with_suffix(".lock")
    partial_path = local_path.with_suffix(".partial")

    if lock_path.exists():
        logger.info("Download in progress for %s (lock present); skipping.", local_path)
        return

    logger.warning("Removing incomplete cached checkpoint: %s", local_path)
    if local_path.exists():
        shutil.rmtree(local_path)
    if partial_path.exists():
        shutil.rmtree(partial_path)


def download_step_if_needed(run_url: str, step: int) -> bool:
    remote_url = f"{run_url.rstrip('/')}/{step}"
    local_path = local_checkpoint_path(remote_url)

    if is_checkpoint_complete(local_path):
        logger.info("Step %s already cached at %s", step, local_path)
        return False

    if local_path.exists():
        cleanup_incomplete(local_path)
        if local_path.exists():
            return False

    logger.info("Downloading step %s from %s", step, remote_url)
    openpi_download.maybe_download(remote_url)
    if not is_checkpoint_complete(local_path):
        raise RuntimeError(
            f"Download finished but {_COMPLETE_MARKER} missing at {local_path}. "
            "Checkpoint may still be uploading on GCS."
        )
    logger.info("Finished step %s -> %s", step, local_path)
    return True


def poll_once(args: Args) -> int:
    steps = list_remote_steps(args.run_url)
    if not steps:
        logger.info("No step directories found yet under %s", args.run_url)
        return 0

    downloaded = 0
    for step in steps:
        if step < args.min_step:
            continue
        if args.step_stride > 1 and (step - args.min_step) % args.step_stride != 0:
            continue
        if download_step_if_needed(args.run_url, step):
            downloaded += 1

    logger.info(
        "Poll complete: remote_steps=%s downloaded_now=%s cache_root=%s",
        steps,
        downloaded,
        openpi_download.get_cache_dir(),
    )
    return downloaded


def main(args: Args) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.info("Watching %s (poll every %.0fs)", args.run_url, args.poll_interval_sec)

    while True:
        try:
            poll_once(args)
        except Exception:
            logger.exception("Poll failed")

        if args.once:
            break
        time.sleep(args.poll_interval_sec)


if __name__ == "__main__":
    main(tyro.cli(Args))
