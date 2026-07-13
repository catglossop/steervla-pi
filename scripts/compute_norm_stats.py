"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}

def _action_supervised_keep_mask(batch: dict) -> np.ndarray | None:
    """Per-batch-row mask for action-supervised samples (exclude HL / CoT-only rows)."""
    mask = batch.get("action_loss_mask")
    if mask is None:
        return None

    mask = np.asarray(mask, dtype=bool)
    if mask.ndim == 1:
        return mask
    return np.any(mask.reshape(mask.shape[0], -1), axis=1)


def _filter_for_norm_stats(batch: dict, key: str) -> np.ndarray | None:
    """Keep only action-supervised samples for norm stats."""
    values = np.asarray(batch[key])
    keep = _action_supervised_keep_mask(batch)
    if keep is None:
        return values
    if not np.any(keep):
        return None
    return values[keep]



def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(config_name: str, max_frames: int | None = None):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    skipped_hl_rows = 0
    total_rows = 0

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        batch_rows = int(np.asarray(batch["state"]).shape[0])
        total_rows += batch_rows

        state = _filter_for_norm_stats(batch, "state")
        actions = _filter_for_norm_stats(batch, "actions")
        if state is None:
            skipped_hl_rows += batch_rows
            continue

        skipped_hl_rows += batch_rows - int(state.shape[0])
        stats["state"].update(state)
        if actions is not None:
            stats["actions"].update(actions)

    if skipped_hl_rows:
        print(
            f"Excluded {skipped_hl_rows}/{total_rows} batch rows from state/action norm stats "
            "(HL / action_loss_mask=False)."
        )

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
