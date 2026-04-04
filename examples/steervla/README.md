# SteerVLA: Driving Policies in openpi

SteerVLA adapts the [bigvision-palivla-drive](https://github.com/user/bigvision-palivla-drive) autonomous driving VLA pipeline to the openpi framework, enabling training of pi0/pi0.5/pi0-FAST models on nuScenes driving data.

We offer instructions for:
- [Running inference with a trained SteerVLA policy](#running-steervla-inference)
- [Training on the full nuScenes RLDS dataset](./README_train.md#training-on-nuscenes-rlds)
- [Fine-tuning on a custom driving dataset](./README_train.md#fine-tuning-on-custom-driving-datasets)

## Overview

SteerVLA trains vision-language-action models for autonomous driving. The model takes:
- **Input**: Front camera image + ego vehicle state history (speed, heading) + language instruction
- **Output**: Driving action chunks (delta speed + course angle changes)

### Action Space

| Dimension | Description | Normalization |
|-----------|-------------|---------------|
| `delta_speed` | Speed change (m/s) | Divided by 10 |
| `course` | Heading angle change (degrees) | Centered to (-180, 180], divided by 180 |
| `delta_x` (optional) | Ego-frame forward displacement | Divided by 15 |
| `delta_y` (optional) | Ego-frame lateral displacement | Divided by 15 |

### Ego State

The ego state consists of interleaved `[speed, heading_course]` pairs for the vehicle's recent history. With `include_ego_history=True` (default), the last 4 timesteps (8 values) are used.

## Running SteerVLA Inference

### Step 1: Start a policy server

On a machine with a GPU:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_steervla_inference \
    --policy.dir=<path_to_your_checkpoint>
```

### Step 2: Run inference

Install the openpi client on the inference machine:

```bash
pip install openpi-client tyro
```

Run the example inference script:

```bash
python examples/steervla/main.py --remote_host=<server_ip> --remote_port=8000
```

The `main.py` script is a template that demonstrates the API. Integrate it with your simulator (e.g., CARLA) or data replay pipeline by replacing the dummy observations with real data.

### Inference API

The policy server expects these keys in each request:

| Key | Shape | Description |
|-----|-------|-------------|
| `observation/image` | `(H, W, 3)` uint8 | Front camera RGB image |
| `observation/state` | `(8,)` float32 | Ego state history (4 × [speed, course]) |
| `observation/current_speed` | scalar float32 | Current speed for prompt injection |
| `prompt` | string | Driving instruction |

The server returns:

| Key | Shape | Description |
|-----|-------|-------------|
| `actions` | `(action_horizon, 2)` float32 | Normalized driving actions |

## Available Configs

| Config Name | Model | Description |
|-------------|-------|-------------|
| `pi05_steervla` | pi0.5 | Full RLDS training on nuScenes |
| `pi0_steervla` | pi0 | Full RLDS training on nuScenes |
| `pi0_fast_steervla` | pi0-FAST | Full RLDS training on nuScenes |
| `pi05_steervla_finetune` | pi0.5 | Fine-tuning on LeRobot dataset |
| `pi05_steervla_inference` | pi0.5 | Inference-only config |
