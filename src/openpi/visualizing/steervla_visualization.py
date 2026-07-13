"""Trajectory visualization and evaluation for SteerVLA training.

Adapted from bigvision-palivla-drive/scripts/visualization.py for the openpi framework.
Produces wandb-logged trajectory plots and computes ADE/FDE metrics by running
the model's flow-matching sampler on a held-out batch during training.
"""

import logging
import re
import tempfile
from collections.abc import Sequence

import flax.nnx as nnx
import jax
import jax.experimental.multihost_utils as multihost_utils
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.training.utils as training_utils
import openpi.models.pi0_cot as pi0_cot
from openpi.models.tokenizer import CoTPaligemmaTokenizer

matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Action denormalization
# ---------------------------------------------------------------------------

def denormalize_actions(
    actions: np.ndarray,
    action_dim: int,
    output_action_format: str | None = None,
) -> np.ndarray:
    """Reverse the normalization applied in steervla_rlds_dataset.py."""
    actions = actions[..., :action_dim]

    if output_action_format in (
        "delta_speed_t_delta_course_t_delta_course_space",
        "DELTA_SPEED_T_DELTA_COURSE_T_DELTA_COURSE_SPACE",
    ):
        out = np.empty_like(actions)
        out[..., 0] = actions[..., 0] * 10.0
        out[..., 1] = actions[..., 1] * 180.0
        out[..., 2] = actions[..., 2] * 180.0
        return out

    if output_action_format in (
        "delta_xy_t_delta_xy_space",
        "DELTA_XY_T_DELTA_XY_SPACE",
    ):
        out = np.empty_like(actions)
        out[..., :2] = actions[..., :2] * 7.0
        out[..., 2:] = actions[..., 2:]
        return out

    if output_action_format in (
        "delta_xy_t_delta_course_space",
        "DELTA_XY_T_DELTA_COURSE_SPACE",
    ):
        out = np.empty_like(actions)
        out[..., :2] = actions[..., :2] * 7.0
        out[..., 2] = actions[..., 2] * 180.0
        return out

    # Default nuScenes format: [delta_speed/10, course/180, ...]
    out = np.empty_like(actions)
    out[..., 0] = actions[..., 0] * 10.0
    out[..., 1] = actions[..., 1] * 180.0
    if action_dim > 2:
        out[..., 2:] = actions[..., 2:] * 15.0
    return out


# ---------------------------------------------------------------------------
# Waypoint computation
# ---------------------------------------------------------------------------

def compute_waypoints(
    denorm_actions: np.ndarray,
    initial_speeds: np.ndarray,
    dt: float = 0.5,
    output_action_format: str | None = None,
) -> np.ndarray:
    """Convert denormalized action sequences to (x, y) waypoints.

    Returns array of shape (batch, horizon, 2).
    """
    fmt = (output_action_format or "").upper()
    if fmt in ("DELTA_XY_T_DELTA_XY_SPACE", "DELTA_XY_T_DELTA_COURSE_SPACE"):
        dx = denorm_actions[..., 0]
        dy = denorm_actions[..., 1]
    else:
        delta_speeds = denorm_actions[..., 0]
        delta_courses = denorm_actions[..., 1]
        speeds = np.cumsum(delta_speeds, axis=-1) + initial_speeds[..., None]
        headings = np.cumsum(delta_courses, axis=-1)
        dx = speeds * np.cos(np.deg2rad(headings)) * dt
        dy = speeds * np.sin(np.deg2rad(headings)) * dt

    x = np.cumsum(dx, axis=-1)
    y = np.cumsum(dy, axis=-1)
    return np.stack([x, y], axis=-1)


# ---------------------------------------------------------------------------
# Trajectory metrics
# ---------------------------------------------------------------------------

def compute_trajectory_metrics(
    pred_wp: np.ndarray,
    gt_wp: np.ndarray,
) -> dict[str, float]:
    """ADE and FDE at different horizon lengths."""
    metrics: dict[str, float] = {}
    for n in [2, 4, 6]:
        if n > pred_wp.shape[1]:
            continue
        dists = np.sqrt(np.sum((pred_wp[:, :n] - gt_wp[:, :n]) ** 2, axis=-1))
        metrics[f"eval/ade_wp{n}"] = float(np.mean(dists))
        metrics[f"eval/fde_wp{n}"] = float(np.mean(dists[:, -1]))
    return metrics


def _short_dataset_name(name: str) -> str:
    """Compact wandb key from a full RLDS dataset name."""
    short = name.removeprefix("simlingo_dataset_")
    short = re.sub(r"_img512_\d+$", "", short)
    return short or name


def _decode_subtask_from_tokens(
    subtask_ids: np.ndarray,
    subtask_mask: np.ndarray,
    tokenizer: CoTPaligemmaTokenizer,
) -> str:
    text = _decode_cot_segment(
        subtask_ids,
        subtask_mask,
        tokenizer._tokenizer,
        start_id=tokenizer._start_of_subtask(),
        end_id=tokenizer._end_of_subtask(),
    )
    return _strip_loc_spans(text)


def _per_step_displacement_magnitude(
    denorm_actions: np.ndarray,
    output_action_format: str | None,
) -> np.ndarray:
    """Per-timestep scalar motion magnitude from denormalized actions."""
    fmt = (output_action_format or "").upper()
    if fmt in ("DELTA_XY_T_DELTA_XY_SPACE", "DELTA_XY_T_DELTA_COURSE_SPACE"):
        return np.sqrt(denorm_actions[..., 0] ** 2 + denorm_actions[..., 1] ** 2)
    return np.abs(denorm_actions[..., 0])


def _infer_motion_label(
    denorm_actions: np.ndarray,
    output_action_format: str | None,
    *,
    rel_threshold: float = 0.05,
) -> str:
    """Classify predicted motion as accelerate / decelerate / constant."""
    mag = _per_step_displacement_magnitude(denorm_actions, output_action_format)
    if mag.ndim > 1:
        mag = np.mean(mag, axis=-1)
    n = mag.shape[-1]
    k = max(1, n // 3)
    early = float(np.mean(mag[:k]))
    late = float(np.mean(mag[-k:]))
    rel_change = (late - early) / (early + 1e-6)
    if rel_change < -rel_threshold:
        return "decelerate"
    if rel_change > rel_threshold:
        return "accelerate"
    return "constant"


def _expected_motion_from_dataset(dataset_name: str) -> str | None:
    if "acceleration_negative" in dataset_name:
        return "decelerate"
    if "acceleration_positive" in dataset_name:
        return "accelerate"
    return None


def _compute_per_dataset_loss_metrics(
    per_sample_loss: np.ndarray,
    dataset_ids: np.ndarray,
    dataset_names: Sequence[str],
) -> dict[str, float]:
    """Mean eval loss and sample count per dataset source present in the batch."""
    metrics: dict[str, float] = {}
    for dataset_id in np.unique(dataset_ids):
        idx = int(dataset_id)
        if idx < 0 or idx >= len(dataset_names):
            continue
        mask = dataset_ids == idx
        count = int(mask.sum())
        if count == 0:
            continue
        short = _short_dataset_name(dataset_names[idx])
        metrics[f"eval/dataset/{short}/loss"] = float(np.mean(per_sample_loss[mask]))
        metrics[f"eval/dataset/{short}/count"] = float(count)
    return metrics


def _prepare_image_for_display(img: np.ndarray) -> np.ndarray:
    if np.issubdtype(img.dtype, np.floating):
        if img.max() <= 1.0:
            return np.clip(img * 255, 0, 255).astype(np.uint8)
        return np.clip(img, 0, 255).astype(np.uint8)
    return img


def _make_trajectory_figure(
    gt_wp: np.ndarray,
    pred_wp: np.ndarray,
    image: np.ndarray,
    *,
    sample_i: int,
    step: int,
    group_label: str,
    subtask_text: str = "",
    extra_title: str = "",
    caption_suffix: str = "",
) -> wandb.Image:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(gt_wp[:, 1], gt_wp[:, 0], "g.-", label="Ground Truth", linewidth=2)
    ax.plot(pred_wp[:, 1], pred_wp[:, 0], "b.--", label="Predicted", linewidth=2)
    ax.plot(0, 0, "ko", markersize=8)
    ax.set_xlabel("Y (m)")
    ax.set_ylabel("X (m)")
    title = f"Trajectory — {group_label} sample {sample_i}"
    if extra_title:
        title = f"{title}\n{extra_title}"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis("equal")

    ax = axes[1]
    ax.imshow(_prepare_image_for_display(image))
    ax.set_title("Camera view")
    ax.axis("off")

    if subtask_text:
        fig.text(
            0.5,
            -0.02,
            f"Subtask:\n{subtask_text}",
            ha="center",
            va="top",
            fontsize=8,
            family="monospace",
            wrap=True,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9),
        )

    fig.tight_layout()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        fig.savefig(tmp.name, format="png", bbox_inches="tight", dpi=100)
        plt.close(fig)
        caption = f"step {step} {group_label} sample {sample_i}"
        if caption_suffix:
            caption = f"{caption} — {caption_suffix}"
        return wandb.Image(tmp.name, caption=caption)


# ---------------------------------------------------------------------------
# Eval step: run model sampling and collect predictions
# ---------------------------------------------------------------------------

def eval_step(
    state: training_utils.TrainState,
    rng: at.KeyArrayLike,
    batch: tuple[_model.Observation, _model.Actions],
) -> dict[str, np.ndarray | float]:
    """Run one eval step: compute loss and sample actions.

    Both compute_loss and sample_actions handle observation preprocessing
    internally, so we pass the raw observation directly.
    """
    model = nnx.merge(state.model_def, state.params)
    model.eval()

    observation, gt_actions = batch

    per_timestep_loss = model.compute_loss(rng, observation, gt_actions, train=False)
    per_sample_loss = jnp.mean(per_timestep_loss, axis=-1)
    loss = jnp.mean(per_sample_loss)
    pred_actions = model.sample_actions(rng, observation)

    return jax.device_get({
        "loss": loss,
        "per_sample_loss": per_sample_loss,
        "pred_actions": pred_actions,
        "gt_actions": gt_actions,
    })


# ---------------------------------------------------------------------------
# Main visualization entry-point
# ---------------------------------------------------------------------------

def run_visualization_evaluation(
    state: training_utils.TrainState,
    rng: at.KeyArrayLike,
    batch: tuple[_model.Observation, _model.Actions],
    *,
    step: int,
    action_dim: int = 4,
    output_action_format: str | None = None,
    dt: float = 0.5,
    vis_samples: int = 5,
    dataset_names: Sequence[str] | None = None,
    accel_decel_vis_samples: int = 3,
) -> dict[str, float]:
    """Run evaluation on process 0 only; other hosts barrier-wait (no duplicate GPU eval)."""
    if jax.process_index() != 0:
        multihost_utils.sync_global_devices("steervla_eval_traj")
        return {}

    try:
        eval_info = eval_step(state, rng, batch)

        pred_actions = np.asarray(eval_info["pred_actions"])
        gt_actions = np.asarray(eval_info["gt_actions"])

        # Determine which samples carry action supervision (i.e., are NOT high-level / CoT-only).
        # HL samples have action_loss_mask=False for all timesteps and should be excluded
        # from action-level eval metrics.
        observation = batch[0]
        if observation.action_loss_mask is not None:
            action_loss_mask = np.asarray(jax.device_get(observation.action_loss_mask))
            non_hl_mask = np.any(action_loss_mask, axis=-1)
        else:
            non_hl_mask = np.ones(pred_actions.shape[0], dtype=bool)

        # Denormalize
        pred_denorm = denormalize_actions(pred_actions, action_dim, output_action_format)
        gt_denorm = denormalize_actions(gt_actions, action_dim, output_action_format)

        # Extract initial speeds from state (last value before normalization, ×20 to undo /20)
        states = np.asarray(jax.device_get(observation.state))
        initial_speeds = states[:, -2] * 20.0 if states.shape[-1] >= 2 else np.zeros(states.shape[0])

        # Compute waypoints
        pred_wp = compute_waypoints(pred_denorm, initial_speeds, dt, output_action_format)
        gt_wp = compute_waypoints(gt_denorm, initial_speeds, dt, output_action_format)

        # Restrict action-level eval metrics to non-HL samples only.
        metrics: dict[str, float] = {}
        n_non_hl = int(non_hl_mask.sum())
        metrics["eval/num_non_hl_samples"] = float(n_non_hl)
        if n_non_hl > 0:
            pred_wp_eval = pred_wp[non_hl_mask]
            gt_wp_eval = gt_wp[non_hl_mask]
            metrics.update(compute_trajectory_metrics(pred_wp_eval, gt_wp_eval))

            act_mse = np.mean(
                (pred_actions[non_hl_mask, ..., :action_dim] - gt_actions[non_hl_mask, ..., :action_dim]) ** 2
            )
            metrics["eval/action_mse"] = float(act_mse)
        else:
            logging.warning("run_visualization_evaluation: no non-HL samples in eval batch; skipping action metrics.")

        metrics["eval/loss"] = float(eval_info["loss"])

        per_sample_loss = np.asarray(eval_info["per_sample_loss"])
        if dataset_names and observation.dataset_id is not None:
            dataset_ids = np.asarray(jax.device_get(observation.dataset_id))
            metrics.update(_compute_per_dataset_loss_metrics(per_sample_loss, dataset_ids, dataset_names))

        subtask_texts: list[str] = []
        if (
            observation.tokenized_subtask is not None
            and observation.tokenized_subtask_mask is not None
        ):
            cot_tokenizer = CoTPaligemmaTokenizer(use_fast_tokens=False)
            subtask_ids = np.asarray(jax.device_get(observation.tokenized_subtask))
            subtask_masks = np.asarray(jax.device_get(observation.tokenized_subtask_mask))
            subtask_texts = [
                _decode_subtask_from_tokens(subtask_ids[i], subtask_masks[i], cot_tokenizer)
                for i in range(subtask_ids.shape[0])
            ]

        images_dict = jax.device_get(observation.images)
        base_key = next(iter(images_dict))
        base_images = np.asarray(images_dict[base_key])

        def _subtask_for(i: int) -> str:
            return subtask_texts[i] if subtask_texts else ""

        def _make_trajectory_figures(sample_indices: np.ndarray, group_label: str) -> list:
            figs = []
            for sample_i in sample_indices:
                sample_i = int(sample_i)
                figs.append(
                    _make_trajectory_figure(
                        gt_wp[sample_i],
                        pred_wp[sample_i],
                        base_images[sample_i],
                        sample_i=sample_i,
                        step=step,
                        group_label=group_label,
                        subtask_text=_subtask_for(sample_i),
                    )
                )
            return figs

        non_hl_indices = np.flatnonzero(non_hl_mask)[:vis_samples]
        hl_indices = np.flatnonzero(~non_hl_mask)[:vis_samples]

        log_dict: dict = {**metrics}
        if non_hl_indices.size > 0:
            log_dict["eval/trajectories_non_hl"] = _make_trajectory_figures(non_hl_indices, "non-HL")
        if hl_indices.size > 0:
            log_dict["eval/trajectories_hl"] = _make_trajectory_figures(hl_indices, "HL")

        if dataset_names and observation.dataset_id is not None:
            dataset_ids = np.asarray(jax.device_get(observation.dataset_id))
            accel_figs = []
            accel_correct = 0
            accel_total = 0
            for sample_i in range(pred_actions.shape[0]):
                if not non_hl_mask[sample_i]:
                    continue
                dataset_id = int(dataset_ids[sample_i])
                if dataset_id < 0 or dataset_id >= len(dataset_names):
                    continue
                dataset_name = dataset_names[dataset_id]
                expected_motion = _expected_motion_from_dataset(dataset_name)
                if expected_motion is None:
                    continue
                pred_motion = _infer_motion_label(
                    pred_denorm[sample_i],
                    output_action_format,
                )
                matches = pred_motion == expected_motion
                accel_total += 1
                accel_correct += int(matches)

            if accel_total > 0:
                metrics["eval/accel_decel_policy_correct_rate"] = float(accel_correct / accel_total)
                metrics["eval/accel_decel_policy_correct_count"] = float(accel_correct)
                metrics["eval/accel_decel_policy_total"] = float(accel_total)

            per_dataset_accel: dict[str, list[int]] = {}
            for sample_i in range(pred_actions.shape[0]):
                if not non_hl_mask[sample_i]:
                    continue
                dataset_id = int(dataset_ids[sample_i])
                if dataset_id < 0 or dataset_id >= len(dataset_names):
                    continue
                dataset_name = dataset_names[dataset_id]
                if _expected_motion_from_dataset(dataset_name) is None:
                    continue
                short = _short_dataset_name(dataset_name)
                per_dataset_accel.setdefault(short, []).append(sample_i)

            for short, indices in per_dataset_accel.items():
                correct = 0
                for sample_i in indices:
                    expected_motion = _expected_motion_from_dataset(dataset_names[int(dataset_ids[sample_i])])
                    pred_motion = _infer_motion_label(
                        pred_denorm[sample_i],
                        output_action_format,
                    )
                    correct += int(pred_motion == expected_motion)
                metrics[f"eval/dataset/{short}/accel_decel_correct_rate"] = float(correct / len(indices))
                metrics[f"eval/dataset/{short}/accel_decel_count"] = float(len(indices))

            selected_accel: list[int] = []
            for short, indices in sorted(per_dataset_accel.items()):
                for sample_i in indices[:accel_decel_vis_samples]:
                    if sample_i not in selected_accel:
                        selected_accel.append(sample_i)

            for sample_i in selected_accel:
                dataset_id = int(dataset_ids[sample_i])
                dataset_name = dataset_names[dataset_id]
                expected_motion = _expected_motion_from_dataset(dataset_name)
                assert expected_motion is not None
                pred_motion = _infer_motion_label(
                    pred_denorm[sample_i],
                    output_action_format,
                )
                matches = pred_motion == expected_motion
                short = _short_dataset_name(dataset_name)
                extra_title = (
                    f"Dataset: {short}\n"
                    f"Expected: {expected_motion.upper()} | "
                    f"Policy: {pred_motion.upper()} | "
                    f"Correct: {'TRUE' if matches else 'FALSE'}"
                )
                accel_figs.append(
                    _make_trajectory_figure(
                        gt_wp[sample_i],
                        pred_wp[sample_i],
                        base_images[sample_i],
                        sample_i=sample_i,
                        step=step,
                        group_label="accel/decel",
                        subtask_text=_subtask_for(sample_i),
                        extra_title=extra_title,
                        caption_suffix=f"{expected_motion} correct={'TRUE' if matches else 'FALSE'}",
                    )
                )

            if accel_figs:
                log_dict["eval/trajectories_accel_decel"] = accel_figs

        log_dict.update(metrics)
        wandb.log(log_dict, step=step)
        logging.info(
            f"Eval step {step}: "
            + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        )

        return metrics
    finally:
        multihost_utils.sync_global_devices("steervla_eval_traj")


# ---------------------------------------------------------------------------
# Chain-of-thought visualization
# ---------------------------------------------------------------------------

def _decode_tokens(token_ids: np.ndarray, mask: np.ndarray, tokenizer) -> str:
    """Decode a padded token array back to text using the given SP tokenizer."""
    valid = token_ids[mask.astype(bool)]
    return tokenizer.decode(valid.tolist())


def _decode_cot_segment(
    token_ids: np.ndarray,
    mask: np.ndarray,
    tokenizer,
    *,
    start_id: int,
    end_id: int,
) -> str:
    """Decode a CoT segment, dropping the leading ``<start_of_*>`` delimiter
    and truncating at the first ``<end_of_*>`` delimiter if either is present
    in the valid (mask=True) portion.

    Both segments emitted by ``CoTPaligemmaTokenizer`` are framed as
    ``[<start_of_X>, ...body..., <end_of_X>, (eos)]``; the model's
    ``sample_cot`` also bootstraps each segment with ``<start_of_X>``, so this
    helper strips them out of both ground-truth and predicted strings.
    """
    valid = token_ids[mask.astype(bool)]
    if valid.size > 0 and int(valid[0]) == start_id:
        valid = valid[1:]
    end_positions = np.where(valid == end_id)[0]
    if end_positions.size > 0:
        valid = valid[: end_positions[0]]
    return tokenizer.decode(valid.tolist())


_LOC_RE = re.compile(r"(?:<loc\d+>)+")


def _strip_loc_spans(text: str) -> str:
    """Remove PaliGemma-style ``<locN>`` spans (common spurious LM prior when CoT boundaries are weak)."""
    return _LOC_RE.sub("", text).strip()


def _decode_fast_segment(token_ids: np.ndarray, mask: np.ndarray, tokenizer) -> str:
    """Decode the FAST action token segment (``Action: ... |`` layout)."""
    valid = token_ids[mask.astype(bool)]
    if valid.size == 0:
        return ""
    return tokenizer._tokenizer.decode(valid.tolist())


def _make_gt_vs_fast_trajectory_figure(
    gt_wp: np.ndarray,
    fast_wp: np.ndarray,
    image: np.ndarray,
    *,
    sample_i: int,
    step: int,
    group_label: str,
) -> wandb.Image:
    """Side-by-side camera + trajectory: GT continuous actions vs FAST token reconstruction."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(gt_wp[:, 1], gt_wp[:, 0], "g.-", label="GT actions", linewidth=2)
    ax.plot(fast_wp[:, 1], fast_wp[:, 0], "m.--", label="FAST tokens (recon)", linewidth=2)
    ax.plot(0, 0, "ko", markersize=8)
    ax.set_xlabel("Y (m)")
    ax.set_ylabel("X (m)")
    ax.set_title(f"FAST recon trajectory — {group_label} sample {sample_i}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis("equal")

    ax = axes[1]
    ax.imshow(image)
    ax.set_title("Camera view")
    ax.axis("off")

    fig.tight_layout()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        fig.savefig(tmp.name, format="png", bbox_inches="tight", dpi=100)
        plt.close(fig)
        return wandb.Image(
            tmp.name,
            caption=f"step {step} {group_label} sample {sample_i} — GT vs FAST token trajectory",
        )


def _fast_reconstruction_metrics(
    gt_wp: np.ndarray,
    fast_wp: np.ndarray,
) -> dict[str, float]:
    """ADE/FDE between GT waypoints and FAST-reconstructed waypoints."""
    metrics: dict[str, float] = {}
    for n in [2, 4, 6]:
        if n > gt_wp.shape[0]:
            continue
        dists = np.sqrt(np.sum((fast_wp[:n] - gt_wp[:n]) ** 2, axis=-1))
        metrics[f"eval/fast_recon_ade_wp{n}"] = float(np.mean(dists))
        metrics[f"eval/fast_recon_fde_wp{n}"] = float(dists[-1])
    return metrics


def _cot_aux_metrics_from_eval(
    model,
    rng: at.KeyArrayLike,
    observation: _model.Observation,
    actions: _model.Actions,
) -> dict[str, float]:
    """Teacher-forced CoT / flow aux losses on an eval batch (for wandb)."""
    if not hasattr(model, "compute_loss_with_aux"):
        return {}
    _, aux_metrics = model.compute_loss_with_aux(rng, observation, actions, train=False)
    out: dict[str, float] = {}
    for key, value in aux_metrics.items():
        if key.startswith("cot_") or key == "flow_loss":
            out[f"eval/{key}"] = float(np.mean(jax.device_get(value)))
    return out


def run_cot_visualization(
    state: training_utils.TrainState,
    rng: at.KeyArrayLike,
    batch: tuple[_model.Observation, _model.Actions],
    *,
    step: int,
    vis_samples: int = 5,
    temperature: float = 0.0,
    use_fast_tokens: bool = False,
    action_horizon: int = 10,
    action_dim: int = 32,
    model_action_dim: int | None = None,
    output_action_format: str | None = None,
    dt: float = 0.5,
) -> None:
    """Log CoT visuals: autoregressive ``sample_cot`` vs ground-truth reasoning/subtask.

    Requires ground-truth CoT token fields on the batch. If the model has no
    ``sample_cot`` (e.g. not ``Pi0CoT``), prediction columns show a placeholder.

    When ``use_fast_tokens`` is enabled, logs teacher-forced ``eval/cot_fast_ce`` and
    related aux metrics, adds a FAST action segment column to the wandb table, and
    logs trajectory figures comparing GT actions to actions reconstructed from FAST tokens.

    Runs only on process 0; other hosts synchronize on a barrier (no duplicate work).
    """
    if jax.process_index() != 0:
        multihost_utils.sync_global_devices("steervla_cot_viz")
        return

    try:
        observation = batch[0]
        if observation.tokenized_subtask is None or observation.tokenized_reasoning is None:
            return
        if observation.tokenized_prompt is None or observation.tokenized_prompt_mask is None:
            return

        tokenizer = CoTPaligemmaTokenizer(use_fast_tokens=use_fast_tokens)
        start_of_subtask_id = tokenizer._start_of_subtask()
        end_of_subtask_id = tokenizer._end_of_subtask()
        start_of_reasoning_id = tokenizer._start_of_reasoning()
        end_of_reasoning_id = tokenizer._end_of_reasoning()
        model = nnx.merge(state.model_def, state.params)
        model.eval()

        gt_actions = batch[1]
        cot_metrics = _cot_aux_metrics_from_eval(model, rng, observation, gt_actions)

        if hasattr(model, "sample_cot"):
            cot_rng = jax.random.fold_in(rng, step)
            cot_out = model.sample_cot(cot_rng, observation, temperature=temperature)
            pred_subtask_ids = np.asarray(jax.device_get(cot_out["tokenized_subtask"]))
            pred_subtask_mask = np.asarray(jax.device_get(cot_out["tokenized_subtask_mask"]))
            pred_reasoning_ids = np.asarray(jax.device_get(cot_out["tokenized_reasoning"]))
            pred_reasoning_mask = np.asarray(jax.device_get(cot_out["tokenized_reasoning_mask"]))
        else:
            pred_subtask_ids = pred_subtask_mask = pred_reasoning_ids = pred_reasoning_mask = None

        prompt_ids = np.asarray(jax.device_get(observation.tokenized_prompt))
        prompt_mask = np.asarray(jax.device_get(observation.tokenized_prompt_mask))
        subtask_ids = np.asarray(jax.device_get(observation.tokenized_subtask))
        subtask_mask = np.asarray(jax.device_get(observation.tokenized_subtask_mask))
        reasoning_ids = np.asarray(jax.device_get(observation.tokenized_reasoning))
        reasoning_mask = np.asarray(jax.device_get(observation.tokenized_reasoning_mask))

        has_fast = (
            use_fast_tokens
            and observation.tokenized_fast is not None
            and observation.tokenized_fast_mask is not None
        )
        if has_fast:
            fast_ids = np.asarray(jax.device_get(observation.tokenized_fast))
            fast_mask = np.asarray(jax.device_get(observation.tokenized_fast_mask))

        if observation.action_loss_mask is not None:
            action_loss_mask = np.asarray(jax.device_get(observation.action_loss_mask))
            non_hl_mask = np.any(action_loss_mask, axis=-1)
        else:
            non_hl_mask = np.ones(prompt_ids.shape[0], dtype=bool)

        gt_actions_np = np.asarray(jax.device_get(gt_actions))
        states = np.asarray(jax.device_get(observation.state))
        initial_speeds = states[:, -2] * 20.0 if states.shape[-1] >= 2 else np.zeros(states.shape[0])

        fast_recon_actions = None
        fast_recon_wp = None
        gt_wp_all = None
        decode_action_dim = model_action_dim if model_action_dim is not None else action_dim
        if has_fast and tokenizer.use_fast_tokens:
            fast_recon_actions = np.stack(
                [
                    tokenizer.extract_fast_actions(
                        fast_ids[i],
                        action_horizon,
                        decode_action_dim,
                        mask=fast_mask[i],
                    )
                    for i in range(fast_ids.shape[0])
                ],
                axis=0,
            )
            gt_denorm = denormalize_actions(gt_actions_np, action_dim, output_action_format)
            fast_denorm = denormalize_actions(fast_recon_actions, action_dim, output_action_format)
            gt_wp_all = compute_waypoints(gt_denorm, initial_speeds, dt, output_action_format)
            fast_recon_wp = compute_waypoints(fast_denorm, initial_speeds, dt, output_action_format)

        images_dict = jax.device_get(observation.images)
        base_key = next(iter(images_dict))
        base_images = np.asarray(images_dict[base_key])

        n_vis = min(vis_samples, prompt_ids.shape[0])

        table_columns = [
            "sample",
            "image",
            "prompt",
            "reasoning (GT)",
            "subtask (GT)",
            "reasoning (pred)",
            "subtask (pred)",
        ]
        if has_fast:
            table_columns.append("FAST actions (GT)")
        table = wandb.Table(columns=table_columns)
        figures = []
        fast_traj_figures = []

        for i in range(n_vis):
            prompt_text = _decode_tokens(prompt_ids[i], prompt_mask[i], tokenizer._tokenizer)
            subtask_gt = _decode_cot_segment(
                subtask_ids[i], subtask_mask[i], tokenizer._tokenizer,
                start_id=start_of_subtask_id, end_id=end_of_subtask_id,
            )
            reasoning_gt = _decode_cot_segment(
                reasoning_ids[i], reasoning_mask[i], tokenizer._tokenizer,
                start_id=start_of_reasoning_id, end_id=end_of_reasoning_id,
            )

            if pred_subtask_ids is not None:
                subtask_pred = _decode_cot_segment(
                    pred_subtask_ids[i], pred_subtask_mask[i], tokenizer._tokenizer,
                    start_id=start_of_subtask_id, end_id=end_of_subtask_id,
                )
                reasoning_pred = _decode_cot_segment(
                    pred_reasoning_ids[i], pred_reasoning_mask[i], tokenizer._tokenizer,
                    start_id=start_of_reasoning_id, end_id=end_of_reasoning_id,
                )
            else:
                subtask_pred = "(no sample_cot on this model)"
                reasoning_pred = "(no sample_cot on this model)"

            # Strip spurious ``<locN>`` spans from predictions (LM prior noise).
            subtask_pred = _strip_loc_spans(subtask_pred)
            reasoning_pred = _strip_loc_spans(reasoning_pred)
            subtask_gt = _strip_loc_spans(subtask_gt)
            reasoning_gt = _strip_loc_spans(reasoning_gt)

            fast_gt = ""
            if has_fast:
                fast_gt = _decode_fast_segment(fast_ids[i], fast_mask[i], tokenizer)
                if fast_recon_actions is not None:
                    fast_gt = (
                        f"{fast_gt}\n(recon Δ: "
                        f"{np.round(fast_recon_actions[i, ..., :action_dim], 3).tolist()})"
                    )

            img = base_images[i]
            if np.issubdtype(img.dtype, np.floating):
                img = np.clip(img * 255, 0, 255).astype(np.uint8) if img.max() <= 1.0 else np.clip(img, 0, 255).astype(np.uint8)

            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.imshow(img)
            ax.set_title("Camera", fontsize=10)
            ax.axis("off")

            text_block = (
                f"Prompt:\n{prompt_text}\n\n"
                f"Reasoning (GT):\n{reasoning_gt}\n\n"
                f"Reasoning (pred):\n{reasoning_pred}\n\n"
                f"Subtask (GT):\n{subtask_gt}\n\n"
                f"Subtask (pred):\n{subtask_pred}"
            )
            if has_fast:
                text_block += f"\n\nFAST actions (GT):\n{fast_gt}"
            fig.text(
                0.5, -0.02, text_block,
                ha="center", va="top", fontsize=8,
                family="monospace", wrap=True,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9),
            )
            fig.tight_layout()

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                fig.savefig(tmp.name, format="png", bbox_inches="tight", dpi=100)
                plt.close(fig)
                figures.append(wandb.Image(tmp.name, caption=f"step {step} sample {i}"))

            wb_img = wandb.Image(img)
            row = [
                i,
                wb_img,
                prompt_text,
                reasoning_gt,
                subtask_gt,
                reasoning_pred,
                subtask_pred,
            ]
            if has_fast:
                row.append(fast_gt)
            table.add_data(*row)

            if (
                fast_recon_wp is not None
                and gt_wp_all is not None
                and non_hl_mask[i]
                and np.any(fast_mask[i])
            ):
                group = "non-HL" if non_hl_mask[i] else "HL"
                fast_traj_figures.append(
                    _make_gt_vs_fast_trajectory_figure(
                        gt_wp_all[i],
                        fast_recon_wp[i],
                        img,
                        sample_i=i,
                        step=step,
                        group_label=group,
                    )
                )

        if fast_recon_wp is not None and gt_wp_all is not None:
            recon_indices = [
                i
                for i in range(min(n_vis, fast_ids.shape[0]))
                if non_hl_mask[i] and np.any(fast_mask[i])
            ]
            if recon_indices:
                recon_sums: dict[str, float] = {}
                for i in recon_indices:
                    for k, v in _fast_reconstruction_metrics(gt_wp_all[i], fast_recon_wp[i]).items():
                        recon_sums[k] = recon_sums.get(k, 0.0) + v
                cot_metrics.update({k: v / len(recon_indices) for k, v in recon_sums.items()})

        log_dict: dict = {"eval/cot_figures": figures, "eval/cot_table": table, **cot_metrics}
        if fast_traj_figures:
            log_dict["eval/cot_fast_trajectories"] = fast_traj_figures
        wandb.log(log_dict, step=step)
        if cot_metrics:
            logging.info(
                f"CoT eval step {step}: "
                + ", ".join(f"{k}={v:.4f}" for k, v in cot_metrics.items())
            )
    finally:
        multihost_utils.sync_global_devices("steervla_cot_viz")
