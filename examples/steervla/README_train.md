# Training SteerVLA

## Training on nuScenes RLDS

Train pi0.5 (or pi0/pi0-FAST) on the full nuScenes driving dataset using the RLDS data format.

### Install

Install RLDS dependencies:

```bash
uv sync --group rlds
```

### Prepare Dataset

Your nuScenes RLDS dataset should be in the standard TFDS format. The dataset is expected to have these fields per trajectory step:

| Field | Description |
|-------|-------------|
| `observation/front_image` | Front camera image (encoded JPEG) |
| `observation/state` | Ego state: interleaved `[speed, course]` pairs |
| `action_chunk` | Pre-chunked actions: `[delta_speed, course, ...]` |
| `language_instruction` | Text description of driving behavior |
| `global_course` | Global heading (for optional XY action conversion) |

### Configure

Update the `rlds_data_dir` in your training config. Edit [src/openpi/training/config.py](../../src/openpi/training/config.py) and find the `pi05_steervla` config:

```python
data=RLDSSteerVLADataConfig(
    repo_id="steervla",
    rlds_data_dir="<path_to_nuscenes_rlds_dataset>",  # <-- Set this
    ...
),
```

### Compute Normalization Statistics

```bash
uv run --group rlds scripts/compute_norm_stats.py --config-name pi05_steervla --max-frames 1_000_000
```

### Run Training

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run --group rlds scripts/train.py pi05_steervla \
    --exp-name=my_steervla_experiment --overwrite
```

For pi0 or pi0-FAST, replace `pi05_steervla` with `pi0_steervla` or `pi0_fast_steervla`.

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `action_horizon` | 6 | Number of future timesteps to predict (fps × 3s) |
| `action_dim` | 2 | Action dimensions (delta_speed, course) |
| `include_ego_history` | True | Use 4-step ego state history |
| `speed_in_prompt` | True | Prepend current speed to language prompt |
| `proprio_norm` | True | Normalize state (speed/20, course/180) |
| `include_xy_action` | False | Include ego-frame XY waypoints (4D actions) |
| `batch_size` | 96 | Global batch size |
| `num_train_steps` | 5000 | Total training steps |

### Using 4D Actions (XY Waypoints)

To train with XY waypoint actions in addition to speed/course, set `include_xy_action=True` and `action_dim=4` in your config. This requires the dataset to include `global_course` and XY deltas in `action_chunk`.


## Fine-Tuning on Custom Driving Datasets

For smaller custom datasets, convert to LeRobot format first, then fine-tune.

### Step 1: Convert to LeRobot

If your data is already in RLDS format:

```bash
uv run --group rlds examples/steervla/convert_nuscenes_to_lerobot.py \
    --rlds_data_dir /path/to/your/data \
    --dataset_name your_dataset_name
```

Edit `REPO_NAME` in the script to set your HuggingFace dataset name.

### Step 2: Fine-Tune

Update the `repo_id` in the `pi05_steervla_finetune` config in `config.py`, then run:

```bash
uv run scripts/train.py pi05_steervla_finetune --exp-name=my_finetune --overwrite
```

### Step 3: Serve and Evaluate

Once trained, serve the policy:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_steervla_inference \
    --policy.dir=./checkpoints/pi05_steervla_finetune/my_finetune/<step>/
```

Then use `examples/steervla/main.py` (or your own integration) to run inference.


## Mapping from bigvision-palivla-drive

This table maps key concepts from the original bigvision-palivla-drive codebase to their openpi equivalents:

| bigvision-palivla-drive | openpi (SteerVLA) |
|-------------------------|-------------------|
| `nuscenes_refined_config_tpu.py` | `pi05_steervla` / `pi0_steervla` in `config.py` |
| `nuscenes_dataset_transform()` | `SteerVLARldsDataset.restructure()` + `SteerVLAInputs` |
| `PaliVLAModel` | pi0 / pi0.5 / pi0-FAST models |
| `scripts/train.py` | `scripts/train.py pi05_steervla` |
| `scripts/inference_server.py` | `scripts/serve_policy.py` |
| `octo/data/dataset.py` | `steervla_rlds_dataset.py` |
| `action_tokenizer.bin(...)` | Flow matching (pi0) or FAST tokenizer (pi0-FAST) |
| `sequence_builder.default(...)` | Built-in pi0 sequence handling |
