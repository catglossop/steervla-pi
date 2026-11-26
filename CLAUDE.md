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

Norm stats are **mandatory** before training (enforced in `data_loader.py`) unless the config sets `skip_norm_stats=True`. They are keyed by config `name`, so renaming a config orphans its stats in `assets/<name>/` — point `AssetsConfig` at the old name instead.

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

`TrainConfig` **is** the model factory: `config.model` is a `BaseModelConfig` whose `.create(rng)` returns the model. Configs are looked up by name from `_CONFIGS` in `src/openpi/training/config.py` (that list is the registry; names must be unique). SteerVLA/SimLingo `TrainConfig`s live in `training/misc/steervla_config.py` and are spliced into `_CONFIGS` via `get_steervla_configs()` (same pattern as upstream's roboarena/polaris configs); their data-config factories stay in `config.py`.

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
3. A `TrainConfig` entry appended to `_CONFIGS` (SteerVLA ones go in `training/misc/steervla_config.py`), then run `compute_norm_stats.py`.

For large datasets you also add an RLDS dataset class (see `steervla_rlds_dataset.py`) and a branch in `create_rlds_dataset`. **RLDS requires `num_workers=0`** — it batches and shards internally. The loader path is chosen purely by whether `rlds_data_dir` is set.

### Pi0CoT

`models/pi0_cot.py` has a detailed module docstring covering the prefix layout and every loss term — **read it before touching the model.** In short, the VLM prefix is `[prompt+state (bidirectional)] [reasoning (causal)] [subtask (causal)] [optional FAST action tokens]`, and the action expert attends to images + prompt + subtask but *not* reasoning.

- CoT delimiter tokens (`<start_of_reasoning>` etc.) are carved out of the top of the PaliGemma vocab. The IDs in `pi0_cot.py` and the reserved-slot layout in `tokenizer.py` (`COT_DELIMITER_TOKEN_SLOTS`, `PALIGEMMA_VOCAB_SKIP_TOKENS`) **must stay in sync**.
- Inference is `CoTPolicy.infer_with_cot` (`policies/cot_policy.py`; `create_trained_policy` returns a `CoTPolicy` for any model with `sample_cot`): autoregressively `sample_cot`, stuff the generated tokens back into the `Observation`, then `sample_actions`. Upstream's `Policy` is untouched.
- The FAST tokens in the CoT prefix use `CoTFASTTokenizer`, which shifts FAST codes down by `COT_DELIMITER_TOKEN_SLOTS` so they don't collide with the delimiters. Upstream `FASTTokenizer` is untouched.
- Knowledge insulation (`knowledge_insulation=True`, on `Pi0Config` and `Pi0CoTConfig`) lives in `gemma.Attention`: when both experts run without a KV cache (i.e. training), action-expert queries read the backbone's K/V through `stop_gradient`, so the flow loss never updates the VLM. It reuses upstream's attention math (`_attend`) on the two query groups; `models/gemma_test.py` pins the behavior.

### VLM backbone: PaliGemma or Gemma 3 4B

`Pi0Config`, `Pi0CoTConfig` and `Pi0FASTConfig` take `backbone="paligemma"` (default; unchanged upstream behavior) or `"gemma3"`, plus `image_resolution` (None = 224 / 896; Gemma 3 needs a multiple of 224). Everything backbone-specific is resolved in `models/backbone.py`. Gemma 3 also needs matching variants, which `backbone.validate` enforces at model creation: `paligemma_variant="gemma3_4b"` and, for the flow models, `action_expert_variant="gemma3_expert_1024"`. See `gemma3_steervla_cot_simplified_reasoning_norm` in `training/misc/steervla_config.py`.

- The action expert shares every attention layer with the VLM, so it must match Gemma 3's geometry (34 layers, 8 heads, 4 KV heads, head_dim 256). No pretrained expert exists: it starts from scratch, and there is no π₀.₅ robot pretraining.
- Gemma 3 layers are options on `gemma.Config` / `gemma_fast` (QK-norm, post norms, local/global RoPE); all default to PaliGemma. The 1024-token sliding window is not masked: sequences are asserted to fit inside it, where local == global attention.
- Weights: `Gemma3WeightLoader` converts `google/gemma-3-4b-pt` safetensors in memory (gated: `HF_TOKEN` with the Gemma license accepted) and resizes the vision position embedding to `image_resolution`. The Gemma 3 tokenizer is also fetched from that repo.
- Token ids: CoT delimiters and FAST codes are counted down from the top of the SentencePiece vocab (`tokenizer.SP_VOCAB_SIZE`), which for Gemma 3 is all `<unused>` tokens. Model and tokenizer both read `tokenizer.cot_delimiter_ids(backbone)`; note the SentencePiece size (262,144) differs from Gemma 3's embedding rows (262,208).
- `gemma3_weights_test.py` checks the port against `transformers` (marked `manual`; set `OPENPI_GEMMA3_CKPT` to a local copy). No LoRA variants or PyTorch path for Gemma 3.

### ABC bimanual sim (`pi05_gemma3_abc_sim`)

Policy for the ABC simulator (https://github.com/amazon-far/abc): task prompt + current observation (scene camera, 2 wrist cameras, 14-D joint state), 30-step absolute-joint chunks at 30 Hz. Config in `training/misc/abc_sim_config.py`.

- **Data:** raw MCAP episodes (`/raid/datasets/data/sim/<task>/episode_*`) are converted once into LeRobot shards (`uv run --group abc examples/abc_sim/convert_abc_sim_to_lerobot.py`), resampled onto ABC's 30 Hz grid exactly like ABC's exporter. A deterministic uuid-hash split gives `train_*` / `val_*` shards, and `AbcSimDataset` stitches them. Prompts are the task name, or the per-episode directive from `randomization.json` for put_relative / grab_clutter / count_into_opaque_box / multi_drawer_search.
- **Dropout** (`AbcSimDataConfig`): independent training-only probabilities for the prompt and each camera. It runs in `repack_transforms`, so never at inference. A dropped camera is zeroed and its image mask set False; a dropped prompt is `""`, which is how the unconditional policy is evaluated.
- **Periodic evals:** `TrainConfig.periodic_evals` hooks (`training/periodic_eval.py`), called by train.py right after any checkpoint save.
  - `AbcActionMseEval`: held-out MSE in raw joint space at 1 s / 0.5 s, per task, task mean, worst-k, with and without the prompt.
  - `AbcSimEval`: asynchronous. It launches `scripts/abc_sim_eval_worker.py` on its own GPU, which serves the checkpoint over websocket and runs `examples/abc_sim/sim_client.py` in ABC's separate venv (Python 3.12, mujoco 3.8, torch 2.11; incompatible with openpi's env) with the OpenGL renderer (`camera_backend="mujoco"`, `MUJOCO_GL=egl`). Results are logged on a later eval step under `sim_eval/*` together with `sim_eval/checkpoint_step`. Leave memory headroom on that GPU (e.g. train with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.75`).
- **RTC** (ABC's default eval) is available but off by default. It needs `Pi0Config.action_prefix_max > 0`: training feeds a clean action prefix at flow time 0 with no loss on it, using per-token adaRMS time, and `ActionPrefixPolicy` holds a given prefix fixed while sampling. Enable it in eval with `AbcSimEval(rtc=True)`. With RTC off, eval is ABC's `--no-rtc` loop (execute 15 of 30, replan).

### Fork-specific gotchas

- **`speed_in_prompt` double-injection.** `RLDSSteerVLADataConfig` injects the speed string in the *RLDS loader* and passes `speed_in_prompt=False` to `SteerVLAInputs`. `RLDSSteerVLACoTDataConfig` does the **opposite** (injects in the policy transform). Easy to get backwards and end up with the speed in the prompt twice, or not at all.
- **SteerVLA loader options** live on one field, `DataConfig.steervla: SteerVLALoaderConfig | None` (defined in `steervla_rlds_dataset.py`); when it is set, the RLDS loader builds a `SteerVLARldsDataset` instead of DROID.
- **High-level (reasoning-only) datasets.** `steervla.hl_datasets` are mixed in by weight with `action_supervision=False`, which becomes an all-False `action_loss_mask` — CoT loss applies, action loss doesn't.
- **Camera framing differs between RLDS builds.** All the SimLingo-derived datasets store 512x512 JPEGs, but they were built from the same 1024x512 CARLA front camera under two framings: `simlingo_dataset_*_img512_1116` crop `(170, 0, 852, 359)` first, `simplified_reasoning_dataset` does not crop at all. Mixing them raw trains on two different cameras. `DATASET_IMAGE_CROPS` in `steervla_rlds_dataset.py` re-crops the offenders at decode time, and `resolve_image_crop` logs a **warning** for any dataset whose framing has never been measured — register a new corpus there, or set `image_crop=()` to record that it needs no crop, rather than letting it default silently. Verify a new build by solving for the crop that reproduces a stored frame from its source under `simlingo_rgb_zips` — a correct match lands at <1/255 mean abs error.
- **Batch size must be divisible by `jax.device_count()`** (asserted in `train.py`). This bites on non-power-of-two meshes: with `fsdp_devices=3`, a batch of 512 is invalid.

### Staying close to upstream

`steerpi` tracks openpi (`git remote add upstream https://github.com/Physical-Intelligence/openpi.git`, merge `upstream/main`). Keep fork logic out of upstream code paths: prefer new modules or subclasses (`cot_policy.py`, `CoTFASTTokenizer`, `misc/steervla_config.py`), additive optional fields, and flags that default to upstream behavior. Work kept off `steerpi` on purpose: full in-training eval/visualization (`steervla-cot-norm-compact-eval`), CSP (`tmrl`), yay-robot/CAST configs (`yay-robot-configs`), VLM-activation exposure (`surya/expose-vlm-activations`).

### PyTorch path

`models_pytorch/` mirrors π₀/π₀.₅ in PyTorch (no FAST, no LoRA/FSDP/EMA/mixed-precision). It requires patching the installed `transformers` in-place:

```bash
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

With uv's default hardlink mode this **mutates the shared uv cache** and can leak into other projects; undo with `uv cache clean transformers`.
