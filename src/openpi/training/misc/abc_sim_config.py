"""ABC bimanual simulation configs (https://github.com/amazon-far/abc).

Data: raw episodes under /raid/datasets/data/sim, converted once with
``examples/abc_sim/convert_abc_sim_to_lerobot.py``. Policy setup follows ABC: task prompt + current observation
(scene camera, two wrist cameras, 14-D joint state), 30-step action chunks at 30 Hz (1 s), absolute joint targets.
"""

import openpi.models.pi0_config as pi0_config
import openpi.training.abc_sim_dataset as abc_sim_dataset
import openpi.training.abc_sim_eval as abc_sim_eval
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders


def get_abc_sim_configs():
    # Import here to avoid circular imports.
    from openpi.training.config import AbcSimDataConfig
    from openpi.training.config import TrainConfig

    return [
        TrainConfig(
            name="pi05_gemma3_abc_sim",
            model=pi0_config.Pi0Config(
                pi05=True,
                backbone="gemma3",
                paligemma_variant="gemma3_4b",
                action_expert_variant="gemma3_expert_1024",
                image_resolution=448,  # source frames are 224x168, resize-padded to 448x448
                action_dim=32,
                action_horizon=abc_sim_dataset.CHUNK_LENGTH,
                # Prompt + discretized state. With 3 x 256 image tokens and 30 action tokens the sequence stays
                # inside Gemma 3's 1024-token sliding window (asserted at trace time).
                max_token_len=128,
            ),
            data=AbcSimDataConfig(
                loader=abc_sim_dataset.AbcSimLoaderConfig(
                    lerobot_root="/raid/users/cglossop/datasets/abc_sim_lerobot",
                ),
                # Independent training-only dropout per conditioning input. Prompt dropout trains the
                # unconditional policy that the "no_prompt" evals measure.
                prompt_dropout=0.1,
                top_camera_dropout=0.0,
                left_wrist_camera_dropout=0.0,
                right_wrist_camera_dropout=0.0,
            ),
            weight_loader=weight_loaders.Gemma3WeightLoader("google/gemma-3-4b-pt"),
            # ABC: AdamW(0.9, 0.95), wd 0.01, grad clip 10, lr 1e-4 with 1k warmup, 75k steps.
            optimizer=_optimizer.AdamW(b1=0.9, b2=0.95, weight_decay=0.01, clip_gradient_norm=10.0),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_000, peak_lr=1e-4, decay_steps=75_000, decay_lr=1e-5
            ),
            num_train_steps=75_000,
            batch_size=256,
            fsdp_devices=8,
            num_workers=32,
            log_interval=100,
            save_interval=5_000,
            keep_period=10_000,
            max_to_keep=4,
            periodic_evals=(
                abc_sim_eval.AbcActionMseEval(interval=2_000),
                abc_sim_eval.AbcSimEval(interval=10_000, gpu=7),
            ),
        ),
    ]
