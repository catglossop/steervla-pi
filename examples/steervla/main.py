"""
SteerVLA inference client.

Sends camera images and ego state to the openpi policy server and receives
driving actions (delta_speed, course_angle). Can be used with nuScenes replay
or a live simulator (e.g., CARLA).

Usage:
    python main.py --remote_host=<server_ip> --remote_port=8000
"""

import dataclasses
import datetime
import io
import os
import time

import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
from PIL import Image
import tyro


@dataclasses.dataclass
class Args:
    # Policy server address.
    remote_host: str = "0.0.0.0"
    remote_port: int = 8000

    # Action space settings.
    action_horizon: int = 6
    action_dim: int = 2
    fps: int = 2

    # Ego state settings.
    include_ego_history: bool = True
    proprio_norm: bool = True
    speed_in_prompt: bool = True

    # Output settings.
    output_dir: str = "./steervla_output"


def pad_ego_history(history: list[list[float]], max_len: int = 4) -> np.ndarray:
    """Pad ego state history to fixed length. Each entry is [speed, course]."""
    history_arr = np.asarray(history, dtype=np.float32)
    if len(history_arr) >= max_len:
        padded = history_arr[-max_len:]
    else:
        padding = np.repeat(history_arr[[0]], max_len - len(history_arr), axis=0)
        padded = np.concatenate([padding, history_arr], axis=0)
    return padded.flatten()


def build_request(
    args: Args,
    image: np.ndarray,
    ego_state: np.ndarray,
    current_speed: float,
    instruction: str,
) -> dict:
    """Build a request dict for the openpi policy server."""
    resized_image = image_tools.resize_with_pad(image, 224, 224)

    request = {
        "observation/image": resized_image,
        "observation/state": ego_state.astype(np.float32),
        "prompt": instruction,
    }

    if args.speed_in_prompt:
        request["observation/current_speed"] = np.float32(current_speed)

    return request


def denormalize_actions(actions: np.ndarray) -> np.ndarray:
    """Convert normalized actions back to physical units.

    Input: actions[:, 0] = delta_speed / 10, actions[:, 1] = course / 180
    Output: actions[:, 0] = delta_speed (m/s), actions[:, 1] = course (degrees)
    """
    result = actions.copy()
    result[:, 0] *= 10.0
    result[:, 1] *= 180.0
    return result


def main(args: Args):
    os.makedirs(args.output_dir, exist_ok=True)

    policy_client = websocket_client_policy.WebsocketClientPolicy(
        args.remote_host, args.remote_port
    )

    print("Connected to policy server. Starting inference loop.")
    print("This is a template -- integrate with your simulator or data replay.")

    ego_history: list[list[float]] = []
    prev_yaw = 0.0

    instruction = input("Enter driving instruction (or press Enter for default): ").strip()
    if not instruction:
        instruction = "The car is driving on a highway."

    dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    dummy_speed = 10.0
    dummy_yaw = 0.0

    for tick in range(100):
        start_time = time.time()

        current_speed = dummy_speed + np.random.randn() * 0.5
        current_yaw = dummy_yaw + tick * 0.5

        local_course = (current_yaw - prev_yaw + 360) % 360
        if local_course > 180:
            local_course -= 360.0
        ego_history.append([current_speed, local_course])
        prev_yaw = current_yaw

        ego_state = pad_ego_history(ego_history, max_len=4)

        request_data = build_request(
            args,
            dummy_image,
            ego_state,
            current_speed,
            instruction,
        )

        result = policy_client.infer(request_data)
        pred_actions = result["actions"]

        physical_actions = denormalize_actions(pred_actions)
        print(
            f"Tick {tick}: speed={current_speed:.1f} m/s, "
            f"predicted delta_speed={physical_actions[0, 0]:.2f} m/s, "
            f"course={physical_actions[0, 1]:.1f} deg"
        )

        elapsed = time.time() - start_time
        sleep_time = max(0, 1.0 / args.fps - elapsed)
        time.sleep(sleep_time)

    print("Inference complete.")


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)
