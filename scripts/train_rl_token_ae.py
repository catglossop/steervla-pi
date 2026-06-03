"""Train the RL Token autoencoder on dumped Pi0CoT prefix embeddings."""
import dataclasses
import glob
import json
import logging
import pathlib

import numpy as np
import torch
import torch.utils.data
import tqdm
import tyro
import wandb

from openpi.models.rl_token import RLTokenAEConfig, RLTokenAutoencoder


@dataclasses.dataclass
class Args:
    train_dir: str = "./rl_token_embeddings/traffic_light_v0/train"
    val_dir: str | None = "./rl_token_embeddings/traffic_light_v0/val"
    output_dir: str = "./rl_token_ae/traffic_light_v0"
    batch_size: int = 32
    grad_accum_steps: int = 1
    lr: float = 1e-4
    num_steps: int = 5000
    log_interval: int = 50
    eval_interval: int = 500
    max_train_samples: int | None = None
    seed: int = 0
    d_model: int = 2048
    encoder_layers: int = 4
    decoder_layers: int = 4
    num_heads: int = 8
    weight_decay: float = 0.0
    wandb_enabled: bool = True
    wandb_project: str = "openpi-rlt"
    wandb_name: str | None = None


def _load_shards(shard_dir: str):
    paths = sorted(glob.glob(f"{shard_dir}/shard_*.npz"))
    assert paths, f"no shards found at {shard_dir}"
    arrs = [np.load(p) for p in tqdm.tqdm(paths, desc=f"loading {shard_dir}")]
    z = np.concatenate([a["prefix_out"] for a in arrs])
    m = np.concatenate([a["prefix_mask"] for a in arrs])
    return torch.from_numpy(z), torch.from_numpy(m)


def _loader(z, m, batch_size, shuffle):
    ds = torch.utils.data.TensorDataset(z, m)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                        num_workers=2, pin_memory=True, drop_last=shuffle)


def _masked_l2(pred, target, mask):
    sq = (pred - target).pow(2).sum(-1) * mask.to(pred.dtype)
    return sq.sum() / mask.to(pred.dtype).sum()


def main(args: Args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = pathlib.Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)

    wandb.init(mode="online" if args.wandb_enabled else "disabled", project=args.wandb_project,
               name=args.wandb_name or output_dir.name, config=dataclasses.asdict(args), dir=str(output_dir))

    train_z, train_m = _load_shards(args.train_dir)
    if args.max_train_samples is not None:
        train_z = train_z[: args.max_train_samples]
        train_m = train_m[: args.max_train_samples]
    logging.info("train: %s density=%.3f", tuple(train_z.shape), float(train_m.float().mean()))
    train_loader = _loader(train_z, train_m, args.batch_size, shuffle=True)
    val_loader = None
    if args.val_dir:
        val_z, val_m = _load_shards(args.val_dir)
        logging.info("val: %s density=%.3f", tuple(val_z.shape), float(val_m.float().mean()))
        val_loader = _loader(val_z, val_m, args.batch_size, shuffle=False)

    cfg = RLTokenAEConfig(vla_embed_dim=train_z.shape[-1], d_model=args.d_model,
                          encoder_layers=args.encoder_layers, decoder_layers=args.decoder_layers,
                          num_heads=args.num_heads, max_seq_len=train_z.shape[1])
    model = RLTokenAutoencoder(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logging.info("AE params: %.2fM", n_params / 1e6)
    wandb.config.update({"ae_cfg": dataclasses.asdict(cfg), "ae_params": n_params})

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    log_f = (output_dir / "metrics.jsonl").open("w")

    def log(rec, step):
        log_f.write(json.dumps(rec) + "\n"); log_f.flush()
        wandb.log(rec, step=step)

    def _cycle(loader):
        while True: yield from loader

    model.train()
    data_iter = _cycle(train_loader)
    pbar = tqdm.tqdm(range(args.num_steps), desc="train")
    for step in pbar:
        opt.zero_grad(set_to_none=True)
        losses = []
        for _ in range(args.grad_accum_steps):
            z, m = next(data_iter)
            z, m = z.to(device, non_blocking=True).float(), m.to(device, non_blocking=True)
            out = model(z, m)
            (out["loss"] / args.grad_accum_steps).backward()
            losses.append(float(out["loss"]))
        opt.step()
        train_loss = float(np.mean(losses))

        if step % args.log_interval == 0:
            with torch.no_grad():
                shuf = float(_masked_l2(out["z_hat"], z[torch.randperm(z.shape[0], device=device)], m))
            log({"step": step, "train_loss": train_loss, "shuffle_baseline": shuf, "lr": args.lr,
                 "effective_batch_size": args.batch_size * args.grad_accum_steps}, step)
            pbar.set_postfix(loss=f"{train_loss:.4f}", shuf=f"{shuf:.4f}")

        if val_loader is not None and args.eval_interval > 0 and step > 0 and step % args.eval_interval == 0:
            model.eval()
            with torch.no_grad():
                vl = float(np.mean([float(model(vz.to(device).float(), vm.to(device))["loss"])
                                    for vz, vm in val_loader]))
            model.train()
            log({"step": step, "val_loss": vl}, step)
            logging.info("step %d val_loss=%.4f", step, vl)

    log_f.close()
    torch.save({"model": model.state_dict(), "cfg": dataclasses.asdict(cfg), "args": dataclasses.asdict(args)},
               output_dir / "ae_ckpt.pt")
    logging.info("Done. Checkpoint at %s/ae_ckpt.pt", output_dir)
    wandb.finish()


if __name__ == "__main__":
    main(tyro.cli(Args))
