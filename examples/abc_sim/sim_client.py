"""ABC simulator rollouts against an openpi websocket policy. Runs in ABC's Python env, not openpi's.

Mirrors ABC's eval (abc_minimal/eval_policy.py): world i is ``env.reset(seed=seed + i, randomize=True)``; the policy
returns a 30-step absolute joint chunk and ``execute_steps`` (15) actions run before re-querying; an episode ends
on ``ever_success`` (or ``ever_failed`` for ball_tray) or after ``max_chunks`` chunks.

* Synchronous mode (ABC's --no-rtc): execute 15, observe, query again.
* RTC mode: after ``execute_steps - rtc_prefix_length`` actions, observe and query the next chunk conditioned on the
  ``rtc_prefix_length`` already-committed actions (sent as a clean action prefix); those run, then the new chunk
  continues from its own position ``rtc_prefix_length``. Needs a model trained with action-prefix conditioning.

Rendering uses ABC's ``camera_backend`` (``mujoco`` = OpenGL; set MUJOCO_GL=egl headless).
"""

import argparse
import importlib.util
import json
import pathlib
import time

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "abc_sim_dataset",
    pathlib.Path(__file__).resolve().parents[2] / "src" / "openpi" / "training" / "abc_sim_dataset.py",
)
abc_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(abc_data)

CHUNK = abc_data.CHUNK_LENGTH


def resolve_spec(abc_sim, task: str):
    # Four raw task names only resolve with ABC's "sim_" release prefix.
    for name in (task, f"sim_{task}"):
        try:
            return abc_sim.get_task_spec(name)
        except (KeyError, ValueError):
            continue
    raise ValueError(f"ABC has no sim task spec for {task!r}")


def to_policy_obs(obs: dict, prompt: str) -> dict:
    images = obs["images"]
    return {
        "observation/images/top": images["top"],
        "observation/images/left_wrist": images["left"],
        "observation/images/right_wrist": images["right"],
        "observation/state": np.asarray(obs["state"], np.float32),
        "prompt": prompt,
    }


def rollout(env, policy, *, seed: int, task: str, sim: dict, drop_prompt: bool) -> dict:
    obs, _ = env.reset(seed=seed, randomize=True)
    policy.reset()
    execute, prefix_len = int(sim["execute_steps"]), int(sim["rtc_prefix_length"])
    best = {"success": False, "failed": False, "max_reward": 0.0, "steps": 0}

    def query(observation, prefix=None):
        request = to_policy_obs(observation, "" if drop_prompt else observation["prompt"])
        if prefix is not None:
            padded = np.zeros((CHUNK, abc_data.ACTION_DIM), np.float32)
            padded[: len(prefix)] = prefix
            request["actions"] = padded
            request["action_prefix_len"] = np.int32(len(prefix))
        return np.asarray(policy.infer(request)["actions"], np.float32)

    def run(actions) -> bool:
        for a in actions:
            env.step(a, render_obs=False)
            best["steps"] += 1
            info = env.evaluate_task().to_info(squeeze=True)
            best["max_reward"] = max(best["max_reward"], float(info.get("reward", 0.0)))
            best["success"] |= bool(info.get("ever_success", False))
            best["failed"] |= bool(info.get("ever_failed", False))
            if best["success"] or (best["failed"] and task == "ball_tray_balancing"):
                return True
        return False

    chunk = query(obs)
    start = time.time()
    for _ in range(int(sim["max_chunks"])):
        if not sim["rtc"]:
            if run(chunk[:execute]):
                break
            chunk = query(env.get_obs())
            continue
        # RTC: run up to the early re-query point, then commit the prefix while the next chunk is "computed".
        if run(chunk[: execute - prefix_len]):
            break
        committed = chunk[execute - prefix_len : execute]
        next_chunk = query(env.get_obs(), prefix=committed)
        if run(committed):
            break
        chunk = next_chunk[prefix_len:]
        if len(chunk) < execute:  # model returned fewer free actions than needed; requery synchronously
            chunk = query(env.get_obs())
    return {**best, "seconds": time.time() - start}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sim-args", required=True)
    args = parser.parse_args()
    sim = json.loads(args.sim_args)

    import abc_sim
    from openpi_client import websocket_client_policy

    policy = websocket_client_policy.WebsocketClientPolicy(host="127.0.0.1", port=args.port)
    conditions = [("task_prompt", False)] + ([("no_prompt", True)] if sim.get("unconditional") else [])
    records = []
    for task in sim["tasks"]:
        spec = resolve_spec(abc_sim, task)
        # Directive tasks: the env replaces this with the episode's own directive on reset (as in the dataset).
        env = abc_sim.make_env(
            task=spec.name,
            prompt=abc_data.task_prompt(task),
            render_cameras=True,
            camera_backend=sim["camera_backend"],
            camera_height=168,
            camera_width=224,
        )
        try:
            for condition, drop_prompt in conditions:
                for world in range(int(sim["worlds_per_task"])):
                    seed = int(sim["seed"]) + world
                    r = rollout(env, policy, seed=seed, task=task, sim=sim, drop_prompt=drop_prompt)
                    records.append({"task": task, "condition": condition, "world": world, "seed": seed, **r})
                    print(json.dumps(records[-1]), flush=True)
        finally:
            env.close()
    pathlib.Path(args.out).write_text(json.dumps(records))


if __name__ == "__main__":
    main()
