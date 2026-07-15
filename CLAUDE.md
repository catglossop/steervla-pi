# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A fork of [openpi](https://github.com/Physical-Intelligence/openpi) (Physical Intelligence's VLA models: π₀, π₀-FAST, π₀.₅) extended with **SteerVLA** — autonomous-driving policies trained on nuScenes/SimLingo RLDS data — and **Pi0CoT**, a π₀.₅ variant that generates chain-of-thought reasoning and a subtask before decoding actions.

Upstream openpi remains intact underneath; when changing shared code (`transforms.py`, `data_loader.py`, `model.py`, `pi0.py`), keep the ALOHA/DROID/LIBERO paths working.

## Commands

Environment is managed with `uv`; everything runs through `uv run`.

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync            # install (GIT_LFS_SKIP_SMUDGE is required for LeRobot)
uv sync --group rlds                     # adds TF/TFDS — required for any RLDS (SteerVLA/DROID) config

uv run pytest --strict-markers -m "not manual"        # full suite (what CI runs)
uv run pytest src/openpi/models/pi0_test.py           # single file
uv run pytest src/openpi/models/pi0_test.py -k csp    # single test
# `manual` marker = slow/manual tests, excluded by default.

ruff check . && ruff format .            # lint + format (line-length 120); pre-commit runs both
```

Training / inference (see "Config-driven everything" below — `<config_name>` is a key in `_CONFIGS`):

```bash
uv run --group rlds scripts/compute_norm_stats.py --config-name <config_name> [--max-frames 1_000_000]
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run --group rlds scripts/train.py <config_name> --exp-name=<run> --overwrite
uv run scripts/serve_policy.py policy:checkpoint --policy.config=<config_name> --policy.dir=<ckpt_dir>
```

Norm stats are **mandatory** before training (enforced in `data_loader.py`) unless the config sets `skip_norm_stats=True`. They are keyed by config `name`, so renaming a config orphans its stats in `assets/<name>/` — point `AssetsConfig` at the old name instead (the CSP config at the bottom of `config.py` does exactly this).

TPU runs: `./launch_tpu_job.sh <vm-name> <config-name>` rsyncs the repo to a TPU VM and runs `start_tpu_job.sh` there (needs `WANDB_API_KEY` and `HF_TOKEN` exported).

## Architecture

### The transform pipeline is the core abstraction

Data flows through the same ordered transform list in training and inference:

```
raw sample → repack_transforms → data_transforms (robot-specific) → Normalize → model_transforms (tokenize/resize/pad) → Observation.from_dict
```

Inference (`policies/policy_config.py`) reuses that exact input list and applies the **mirror image** on the way out: `model_transforms.outputs → Unnormalize → data_transforms.outputs → repack.outputs`. If you add an input transform, ask whether it needs an inverse.

- `RepackTransform` is **training-only** — it renames dataset keys to match what the inference environment sends.
- Data transforms (`policies/<robot>_policy.py`) are the robot-specific layer: they emit the canonical `{state, image, image_mask, actions, prompt}` schema documented at `models/model.py:50-80`.
- `Normalize` uses quantile norm (q01/q99) for everything except `ModelType.PI0`, auto-set in `config.py`.

### Config-driven everything

`TrainConfig` **is** the model factory: `config.model` is a `BaseModelConfig` whose `.create(rng)` returns the model. Configs are looked up by name from `_CONFIGS` in `src/openpi/training/config.py` (that list is the registry; names must be unique).

| Model config | `model_type` | Class |
|---|---|---|
| `Pi0Config(pi05=False)` | `PI0` | `Pi0` |
| `Pi0Config(pi05=True)` | `PI05` | `Pi0` (state-as-token + adaRMS) |
| `Pi0FASTConfig` | `PI0_FAST` | `Pi0FAST` |
| `Pi0CoTConfig` | `PI05` | `Pi0CoT` |

`Pi0CoTConfig` deliberately reports `model_type == PI05` but **bypasses `ModelTransformFactory`** — `RLDSSteerVLACoTDataConfig` hand-builds its model transforms with a `CoTPaligemmaTokenizer`. That's why no `PI0_COT` enum member exists; don't add one reflexively.

### Adding a new dataset/robot — three touch points

1. `src/openpi/policies/<robot>_policy.py`: an `Inputs` transform (must branch on `model_type` to pick image keys/masks) and an `Outputs` transform that slices model actions back to the true action dim.
2. A `DataConfigFactory` subclass in `training/config.py` wiring those into `data_transforms`. `LeRobotLiberoDataConfig` is the documented template; `LeRobotSteerVLADataConfig` is the driving analogue.
3. A `TrainConfig` entry appended to `_CONFIGS`, then run `compute_norm_stats.py`.

For large datasets you also add an RLDS dataset class (see `steervla_rlds_dataset.py`) and a branch in `create_rlds_dataset`. **RLDS requires `num_workers=0`** — it batches and shards internally. The loader path is chosen purely by whether `rlds_data_dir` is set.

### Pi0CoT

`models/pi0_cot.py` has a detailed module docstring covering the prefix layout and every loss term — **read it before touching the model.** In short, the VLM prefix is `[prompt+state (bidirectional)] [reasoning (causal)] [subtask (causal)] [optional FAST action tokens]`, and the action expert attends to images + prompt + subtask but *not* reasoning.

- CoT delimiter tokens (`<start_of_reasoning>` etc.) are carved out of the top of the PaliGemma vocab. The IDs in `pi0_cot.py` and the reserved-slot layout in `tokenizer.py` (`COT_DELIMITER_TOKEN_SLOTS`, `PALIGEMMA_VOCAB_SKIP_TOKENS`) **must stay in sync**.
- Inference is `Policy.infer_with_cot`: autoregressively `sample_cot`, stuff the generated tokens back into the `Observation`, then `sample_actions`.
- Context-Smoothed Pre-training (`models/context_smoothing.py`) noises the SigLIP image tokens in the prefix with a per-sample `t_context`, fed to the action expert via adaRMS alongside the flow timestep. `context_smoothing=None` disables it with zero param/behavior change. Its params (`ctx_time_mlp_*`) postdate `pi05_base`, so the weight loader needs `missing_regex=".*(lora|ctx_time_mlp).*"` for them to survive the checkpoint merge.

### Fork-specific gotchas

- **`speed_in_prompt` double-injection.** `RLDSSteerVLADataConfig` injects the speed string in the *RLDS loader* and passes `speed_in_prompt=False` to `SteerVLAInputs`. `RLDSSteerVLACoTDataConfig` does the **opposite** (injects in the policy transform). Easy to get backwards and end up with the speed in the prompt twice, or not at all.
- **High-level (reasoning-only) datasets.** `steervla_hl_datasets` are mixed in by weight with `action_supervision=False`, which becomes an all-False `action_loss_mask` — CoT loss applies, action loss doesn't.
- **Batch size must be divisible by `jax.device_count()`** (asserted in `train.py`). This bites on non-power-of-two meshes: with `fsdp_devices=3`, a batch of 512 is invalid. Note the eval loader derives its own batch size from `config.batch_size` and must satisfy the same constraint.

### PyTorch path

`models_pytorch/` mirrors π₀/π₀.₅ in PyTorch (no FAST, no LoRA/FSDP/EMA/mixed-precision). It requires patching the installed `transformers` in-place:

```bash
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

With uv's default hardlink mode this **mutates the shared uv cache** and can leak into other projects; undo with `uv cache clean transformers`.
