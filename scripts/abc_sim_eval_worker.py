"""Evaluate one checkpoint in the ABC simulator (launched by ``abc_sim_eval.AbcSimEval``, or by hand).

Serves the checkpoint over openpi's websocket policy server and runs ``examples/abc_sim/sim_client.py`` in ABC's
own Python environment (it needs mujoco 3.8 / torch 2.11, incompatible with openpi's). Writes
``<out-dir>/results.json`` with per-rollout records and a summary.

    uv run scripts/abc_sim_eval_worker.py --config-name pi05_gemma3_abc_sim \
        --checkpoint-dir <run_dir>/<step> --out-dir /tmp/sim_eval --sim-args '{"worlds_per_task": 2}'
"""

import argparse
import dataclasses
import json
import logging
import os
import pathlib
import pickle
import socket
import subprocess
import threading
import time

REPO = pathlib.Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_checkpoint(path: pathlib.Path, timeout_s: float = 3600) -> None:
    # Orbax writes to a temporary directory and renames it to <step> on commit.
    deadline = time.time() + timeout_s
    while not (path / "params").exists():
        if time.time() > deadline:
            raise TimeoutError(f"checkpoint never appeared: {path}")
        time.sleep(10)


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config-name")
    group.add_argument("--config-pickle")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sim-args", default="{}", help="JSON of AbcSimEval fields")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)

    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import abc_sim_eval
    from openpi.training import config as _config

    sim = {**dataclasses.asdict(abc_sim_eval.AbcSimEval()), **json.loads(args.sim_args)}
    if args.config_pickle:
        train_config = pickle.loads(pathlib.Path(args.config_pickle).read_bytes())
    else:
        train_config = _config.get_config(args.config_name)
    checkpoint = pathlib.Path(args.checkpoint_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _wait_for_checkpoint(checkpoint)

    policy = policy_config.create_trained_policy(
        train_config,
        checkpoint,
        sample_kwargs={"num_steps": int(sim["num_flow_steps"])},
        # A run trained without normalization saves no norm stats; empty stats make (Un)Normalize the identity.
        norm_stats={} if train_config.skip_norm_stats else None,
    )
    port = _free_port()
    server = websocket_policy_server.WebsocketPolicyServer(policy, host="127.0.0.1", port=port, metadata={})
    threading.Thread(target=server.serve_forever, daemon=True).start()

    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", str(sim["gpu"])).split(",")[0]
    client_env = {
        **os.environ,
        "MUJOCO_GL": "egl",  # OpenGL renderer, headless
        "MUJOCO_EGL_DEVICE_ID": gpu,
        "PYTHONPATH": str(REPO / "src"),  # for openpi.training.abc_sim_dataset (pure Python, no openpi deps)
    }
    results_path = out_dir / "rollouts.json"
    cmd = [
        sim["abc_python"],
        str(REPO / "examples" / "abc_sim" / "sim_client.py"),
        "--port",
        str(port),
        "--out",
        str(results_path),
        "--sim-args",
        json.dumps(sim),
    ]
    logging.info("Running sim client: %s", " ".join(cmd))
    subprocess.run(cmd, env=client_env, check=True)

    rollouts = json.loads(results_path.read_text())
    summary = abc_sim_eval.summarize_rollouts(rollouts)
    (out_dir / "results.json").write_text(json.dumps({"summary": summary, "rollouts": rollouts}, indent=1))
    logging.info("Sim eval summary: %s", json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
