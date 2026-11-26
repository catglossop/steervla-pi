"""Convert raw ABC sim episodes (MCAP) into LeRobot shards for openpi.

Mirrors ABC's own exporter (abc_minimal/export_mcap.py): a fixed 30 Hz clock over the overlap of all
streams (ticks from t0 + 1/30 s), causal floor resampling of cameras, states and actions, and 14-D
absolute joint state/action vectors. Images stay at their native 224x168; the training transforms
resize-pad them. Episodes are split train/val deterministically by uuid hash and written as many small
LeRobot datasets ("shards") in parallel; a shard with a DONE marker is skipped on rerun.

    uv run --group abc examples/abc_sim/convert_abc_sim_to_lerobot.py \
        --raw-root /raid/datasets/data/sim --out-root /raid/users/cglossop/datasets/abc_sim_lerobot --workers 150
"""

import dataclasses
import io
import json
import logging
import multiprocessing as mp
import pathlib
import shutil
import traceback

import numpy as np
import tyro

from openpi.training import abc_sim_dataset as abc

TICK_NS = 33_333_333  # int(1e9 / 30), as in ABC
CAMERA_TOPICS = {"top": "/top-camera", "left_wrist": "/left-wrist-camera", "right_wrist": "/right-wrist-camera"}
STATE_TOPICS = ("/left-arm-state", "/left-ee-state", "/right-arm-state", "/right-ee-state")
ACTION_TOPICS = ("/left-arm-action", "/left-ee-action", "/right-arm-action", "/right-ee-action")
IMAGE_SHAPE = (168, 224, 3)


@dataclasses.dataclass
class Args:
    raw_root: pathlib.Path = pathlib.Path("/raid/datasets/data/sim")
    out_root: pathlib.Path = pathlib.Path("/raid/users/cglossop/datasets/abc_sim_lerobot")
    workers: int = 64
    episodes_per_shard: int = 200
    # Held-out fraction (per mille), chosen by uuid hash within every task.
    val_permille: int = 20
    # Only convert these tasks (default: all).
    tasks: tuple[str, ...] = abc.TASKS
    # Stop after this many episodes per task (for quick tests).
    max_episodes_per_task: int | None = None


# ----------------------------------------------------------------------------- MCAP reading


def _protobuf_decoder():
    """Decode protobuf messages from the FileDescriptorSet embedded in each MCAP schema."""
    from google.protobuf import descriptor_pb2
    from google.protobuf import descriptor_pool
    from google.protobuf import message_factory

    cache = {}

    def decode(schema, data: bytes):
        if schema.id not in cache:
            fds = descriptor_pb2.FileDescriptorSet.FromString(schema.data)
            pool = descriptor_pool.DescriptorPool()
            for fd in fds.file:
                pool.Add(fd)
            descriptor = pool.FindMessageTypeByName(schema.name)
            if hasattr(message_factory, "GetMessageClass"):
                cls = message_factory.GetMessageClass(descriptor)
            else:  # older protobuf
                cls = message_factory.MessageFactory(pool).GetPrototype(descriptor)
            cache[schema.id] = cls
        return cache[schema.id].FromString(data)

    return decode


def read_episode(mcap_path: pathlib.Path) -> tuple[dict, dict]:
    """Return ({camera: [(log_time, h264 bytes)]}, {topic: [(log_time, position vector)]})."""
    from mcap.reader import make_reader

    decode = _protobuf_decoder()
    camera_topic_to_key = {t: k for k, t in CAMERA_TOPICS.items()}
    scalar_topics = set(STATE_TOPICS + ACTION_TOPICS)
    cams: dict[str, list] = {k: [] for k in CAMERA_TOPICS}
    scalars: dict[str, list] = {t: [] for t in scalar_topics}
    with mcap_path.open("rb") as f:
        for schema, channel, message in make_reader(f).iter_messages():
            if channel.topic in camera_topic_to_key:
                cams[camera_topic_to_key[channel.topic]].append((message.log_time, decode(schema, message.data).data))
            elif channel.topic in scalar_topics:
                position = np.asarray(decode(schema, message.data).position, np.float64)
                scalars[channel.topic].append((message.log_time, position))
    for msgs in (*cams.values(), *scalars.values()):
        msgs.sort(key=lambda x: x[0])
    return cams, scalars


def decode_h264(chunks: list[bytes]) -> np.ndarray:
    import av

    with av.open(io.BytesIO(b"".join(chunks)), format="h264") as container:
        return np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)])


def floor_indices(source_ts: np.ndarray, target_ts: np.ndarray) -> np.ndarray:
    """Index of the latest source message at or before each target tick (ABC's causal resampling)."""
    return np.clip(np.searchsorted(source_ts, target_ts, side="right") - 1, 0, len(source_ts) - 1)


def load_episode(episode_dir: pathlib.Path) -> dict | None:
    """Resample one raw episode onto ABC's 30 Hz grid."""
    cams, scalars = read_episode(episode_dir / "episode.mcap")
    if any(not v for v in (*cams.values(), *scalars.values())):
        logging.warning("skip %s: missing stream", episode_dir.name)
        return None
    streams = [*cams.values(), *scalars.values()]
    t0 = max(s[0][0] for s in streams)
    t_end = min(s[-1][0] for s in streams)
    ticks = np.arange(t0 + TICK_NS, t_end + 1, TICK_NS, dtype=np.int64)
    if len(ticks) < abc.CHUNK_LENGTH:
        logging.warning("skip %s: too short (%d steps)", episode_dir.name, len(ticks))
        return None

    def resample(topics):
        parts = []
        for topic in topics:
            ts = np.array([t for t, _ in scalars[topic]], np.int64)
            parts.append(np.stack([v for _, v in scalars[topic]])[floor_indices(ts, ticks)])
        return np.concatenate(parts, axis=-1).astype(np.float32)

    images = {}
    for key, msgs in cams.items():
        frames = decode_h264([data for _, data in msgs])
        ts = np.array([t for t, _ in msgs], np.int64)
        if len(frames) != len(ts):  # h264 messages != frames: respace timestamps, as ABC does
            ts = np.linspace(ts[0], ts[-1], len(frames), dtype=np.int64)
        if frames.shape[1:] != IMAGE_SHAPE:
            raise ValueError(f"{episode_dir}: {key} frames are {frames.shape[1:]}, expected {IMAGE_SHAPE}")
        images[key] = frames[floor_indices(ts, ticks)]
    state, actions = resample(STATE_TOPICS), resample(ACTION_TOPICS)
    assert state.shape[-1] == actions.shape[-1] == abc.ACTION_DIM
    return {"images": images, "state": state, "actions": actions}


# ----------------------------------------------------------------------------- shard writing

FEATURES = {
    **{
        f"observation.images.{cam}": {"dtype": "video", "shape": IMAGE_SHAPE, "names": ["height", "width", "channel"]}
        for cam in abc.CAMERAS
    },
    "observation.state": {"dtype": "float32", "shape": (abc.ACTION_DIM,), "names": ["state"]},
    "action": {"dtype": "float32", "shape": (abc.ACTION_DIM,), "names": ["action"]},
    "task_group": {"dtype": "int64", "shape": (1,), "names": None},
}


SHARD_SKIPPED, SHARD_FAILED = -1, -2


def write_shard(job: tuple[str, list[tuple[str, str]], str]) -> tuple[str, int, int]:
    """Write one shard. Any failure (e.g. a video encode under heavy load) discards the partial shard so the
    caller can retry it; a finished shard carries a DONE marker and is skipped on reruns."""
    shard_name, episodes, out_root = job
    try:
        return _write_shard(shard_name, episodes, out_root)
    except Exception:
        logging.error("shard %s failed, will retry:\n%s", shard_name, traceback.format_exc())
        shutil.rmtree(pathlib.Path(out_root) / shard_name, ignore_errors=True)
        return shard_name, SHARD_FAILED, 0


def _write_shard(shard_name: str, episodes: list[tuple[str, str]], out_root: str) -> tuple[str, int, int]:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    shard_dir = pathlib.Path(out_root) / shard_name
    if (shard_dir / "DONE").exists():
        return shard_name, SHARD_SKIPPED, 0
    if shard_dir.exists():
        shutil.rmtree(shard_dir)  # partial shard from an interrupted run
    dataset = LeRobotDataset.create(
        repo_id=f"abc_sim/{shard_name}",
        root=shard_dir,
        fps=abc.FPS,
        robot_type="abc_bimanual",
        features=FEATURES,
        use_videos=True,
        image_writer_threads=4,
    )
    written, failed, manifest = 0, 0, []
    for task, episode_path in episodes:
        episode_dir = pathlib.Path(episode_path)
        try:
            episode = load_episode(episode_dir)
        except Exception:  # one corrupt episode must not kill the shard
            logging.error("failed %s:\n%s", episode_dir, traceback.format_exc())
            episode = None
        if episode is None:
            failed += 1
            continue
        prompt = abc.task_prompt(task, abc.episode_directive(episode_dir))
        group = np.array([abc.TASKS.index(task)], np.int64)
        for i in range(len(episode["state"])):
            dataset.add_frame(
                {
                    **{f"observation.images.{cam}": episode["images"][cam][i] for cam in abc.CAMERAS},
                    "observation.state": episode["state"][i],
                    "action": episode["actions"][i],
                    "task_group": group,
                    "task": prompt,
                }
            )
        dataset.save_episode()
        manifest.append({"episode": episode_dir.name, "task": task, "prompt": prompt, "length": len(episode["state"])})
        written += 1
    (shard_dir / "episodes_manifest.json").write_text(json.dumps(manifest))
    (shard_dir / "DONE").write_text(json.dumps({"written": written, "failed": failed}))
    return shard_name, written, failed


def plan_shards(args: Args) -> list[tuple[str, list[tuple[str, str]], str]]:
    per_split: dict[str, list[tuple[str, str]]] = {"train": [], "val": []}
    for task in args.tasks:
        episodes = sorted(p for p in (args.raw_root / task).glob("episode_*") if (p / "episode.mcap").exists())
        if args.max_episodes_per_task is not None:
            episodes = episodes[: args.max_episodes_per_task]
        for ep in episodes:
            split = "val" if abc.is_val_episode(ep.name, args.val_permille) else "train"
            per_split[split].append((task, str(ep)))
    jobs = []
    for split, episodes in per_split.items():
        # Interleave tasks across shards so a partially converted dataset is still task-balanced.
        rng = np.random.default_rng(0)
        shuffled = [episodes[i] for i in rng.permutation(len(episodes))]
        for start in range(0, len(shuffled), args.episodes_per_shard):
            name = f"{split}_{start // args.episodes_per_shard:04d}"
            jobs.append((name, shuffled[start : start + args.episodes_per_shard], str(args.out_root)))
    return jobs


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out_root.mkdir(parents=True, exist_ok=True)
    jobs = plan_shards(args)
    n_eps = sum(len(j[1]) for j in jobs)
    logging.info("%d episodes -> %d shards (%d workers)", n_eps, len(jobs), args.workers)
    totals = {"written": 0, "failed_episodes": 0, "skipped_shards": 0, "failed_shards": []}
    workers = args.workers
    for attempt in range(3):  # failed shards are retried at lower concurrency
        failed_jobs = []
        with mp.get_context("spawn").Pool(workers, maxtasksperchild=1) as pool:
            for job, (name, written, failed) in zip(jobs, pool.imap(write_shard, jobs), strict=True):
                if written == SHARD_SKIPPED:
                    totals["skipped_shards"] += 1
                elif written == SHARD_FAILED:
                    failed_jobs.append(job)
                else:
                    totals["written"] += written
                    totals["failed_episodes"] += failed
                logging.info(
                    "attempt %d shard %s (written=%d failed=%d) totals=%s", attempt, name, written, failed, totals
                )
        if not failed_jobs:
            break
        jobs, workers = failed_jobs, max(4, workers // 4)
        logging.warning("%d shards failed; retrying with %d workers", len(jobs), workers)
    totals["failed_shards"] = [j[0] for j in failed_jobs]
    (args.out_root / "CONVERSION_SUMMARY.json").write_text(json.dumps(totals, indent=1))


if __name__ == "__main__":
    main(tyro.cli(Args))
