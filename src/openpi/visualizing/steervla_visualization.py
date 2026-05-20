"""Trajectory visualization and evaluation for SteerVLA training.

Adapted from bigvision-palivla-drive/scripts/visualization.py for the openpi framework.
Produces wandb-logged trajectory plots and computes ADE/FDE metrics by running
the model's flow-matching sampler on a held-out batch during training.
"""

import logging
import re
import tempfile

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

    loss = jnp.mean(model.compute_loss(rng, observation, gt_actions, train=False))
    pred_actions = model.sample_actions(rng, observation)

    return jax.device_get({
        "loss": loss,
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

        images_dict = jax.device_get(observation.images)
        base_key = next(iter(images_dict))
        base_images = np.asarray(images_dict[base_key])

        def _make_trajectory_figures(sample_indices: np.ndarray, group_label: str) -> list:
            figs = []
            for sample_i in sample_indices:
                sample_i = int(sample_i)
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                ax = axes[0]
                ax.plot(
                    gt_wp[sample_i, :, 1], gt_wp[sample_i, :, 0],
                    "g.-", label="Ground Truth", linewidth=2,
                )
                ax.plot(
                    pred_wp[sample_i, :, 1], pred_wp[sample_i, :, 0],
                    "b.--", label="Predicted", linewidth=2,
                )
                ax.plot(0, 0, "ko", markersize=8)
                ax.set_xlabel("Y (m)")
                ax.set_ylabel("X (m)")
                ax.set_title(f"Trajectory — {group_label} sample {sample_i}")
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.axis("equal")

                ax = axes[1]
                img = base_images[sample_i]
                if np.issubdtype(img.dtype, np.floating):
                    img = np.clip(img * 255, 0, 255).astype(np.uint8) if img.max() <= 1.0 else np.clip(img, 0, 255).astype(np.uint8)
                ax.imshow(img)
                ax.set_title("Camera view")
                ax.axis("off")

                fig.tight_layout()

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    fig.savefig(tmp.name, format="png", bbox_inches="tight", dpi=100)
                    plt.close(fig)
                    figs.append(wandb.Image(
                        tmp.name,
                        caption=f"step {step} {group_label} sample {sample_i}",
                    ))
            return figs

        non_hl_indices = np.flatnonzero(non_hl_mask)[:vis_samples]
        hl_indices = np.flatnonzero(~non_hl_mask)[:vis_samples]

        log_dict: dict = {**metrics}
        if non_hl_indices.size > 0:
            log_dict["eval/trajectories_non_hl"] = _make_trajectory_figures(non_hl_indices, "non-HL")
        if hl_indices.size > 0:
            log_dict["eval/trajectories_hl"] = _make_trajectory_figures(hl_indices, "HL")

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


def run_cot_visualization(
    state: training_utils.TrainState,
    rng: at.KeyArrayLike,
    batch: tuple[_model.Observation, _model.Actions],
    *,
    step: int,
    vis_samples: int = 5,
    temperature: float = 0.0,
) -> None:
    """Log CoT visuals: autoregressive ``sample_cot`` vs ground-truth reasoning/subtask.

    Requires ground-truth CoT token fields on the batch. If the model has no
    ``sample_cot`` (e.g. not ``Pi0CoT``), prediction columns show a placeholder.

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

        tokenizer = CoTPaligemmaTokenizer()
        start_of_subtask_id = tokenizer._start_of_subtask()
        end_of_subtask_id = tokenizer._end_of_subtask()
        start_of_reasoning_id = tokenizer._start_of_reasoning()
        end_of_reasoning_id = tokenizer._end_of_reasoning()
        model = nnx.merge(state.model_def, state.params)
        model.eval()

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

        images_dict = jax.device_get(observation.images)
        base_key = next(iter(images_dict))
        base_images = np.asarray(images_dict[base_key])

        n_vis = min(vis_samples, prompt_ids.shape[0])

        table = wandb.Table(
            columns=[
                "sample",
                "image",
                "prompt",
                "reasoning (GT)",
                "subtask (GT)",
                "reasoning (pred)",
                "subtask (pred)",
            ]
        )
        figures = []

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
            table.add_data(
                i,
                wb_img,
                prompt_text,
                reasoning_gt,
                subtask_gt,
                reasoning_pred,
                subtask_pred,
            )

        wandb.log({"eval/cot_figures": figures, "eval/cot_table": table}, step=step)
    finally:
        multihost_utils.sync_global_devices("steervla_cot_viz")
