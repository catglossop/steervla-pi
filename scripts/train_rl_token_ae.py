"""Train the RL Token autoencoder on dumped Pi0CoT prefix embeddings."""
import dataclasses
import glob
import json
import logging
import math
import pathlib

import numpy as np
import torch
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
    lr: float = 1e-4
    cosine_lr: bool = False
    warmup_steps: int = 500
    min_lr: float = 1e-5
    num_steps: int = 5000
    log_interval: int = 50
    eval_interval: int = 500
    resume_from: str | None = None
    start_step: int = 0
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
    wandb_id: str | None = None


class ShardStream:
    """Streams (z, m) batches from .npz shards with bounded RAM (shard-order + buffer shuffle)."""

    def __init__(self, shard_dir, batch_size, *, shuffle, seed=0, buffer_frames=2048, max_frames=None):
        self.paths = sorted(glob.glob(f"{shard_dir}/shard_*.npz"))
        assert self.paths, f"no shards found at {shard_dir}"
        self.batch_size, self.shuffle = batch_size, shuffle
        self.buffer_frames, self.seed, self.max_frames = buffer_frames, seed, max_frames
        self._epoch = 0
        head = np.load(self.paths[0])["prefix_out"]
        self.seq_len, self.embed_dim = int(head.shape[1]), int(head.shape[2])

    def _frames(self, rng):
        order = list(self.paths)
        if self.shuffle:
            rng.shuffle(order)
        buf_z, buf_m, n = [], [], 0
        for p in order:
            a = np.load(p)
            z, m = a["prefix_out"], a["prefix_mask"]
            for i in range(z.shape[0]):
                if self.shuffle:
                    buf_z.append(z[i]); buf_m.append(m[i])
                    if len(buf_z) >= self.buffer_frames:
                        j = int(rng.integers(len(buf_z)))
                        yield buf_z[j], buf_m[j]; n += 1
                        buf_z[j], buf_m[j] = buf_z[-1], buf_m[-1]
                        buf_z.pop(); buf_m.pop()
                else:
                    yield z[i], m[i]; n += 1
                if self.max_frames and n >= self.max_frames:
                    return
        if self.shuffle:
            for j in rng.permutation(len(buf_z)):
                if self.max_frames and n >= self.max_frames:
                    return
                yield buf_z[j], buf_m[j]; n += 1

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch); self._epoch += 1
        bz, bm = [], []
        for z, m in self._frames(rng):
            bz.append(z); bm.append(m)
            if len(bz) == self.batch_size:
                yield torch.from_numpy(np.stack(bz)), torch.from_numpy(np.stack(bm)); bz, bm = [], []
        if bz and not self.shuffle:
            yield torch.from_numpy(np.stack(bz)), torch.from_numpy(np.stack(bm))


def _masked_l2(pred, target, mask):
    sq = (pred - target).pow(2).sum(-1) * mask.to(pred.dtype)
    return sq.sum() / mask.to(pred.dtype).sum()


def _lr_at(step, warmup, total, peak, min_lr):
    if step < warmup:
        return peak * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (peak - min_lr) * (1 + math.cos(math.pi * progress))


def main(args: Args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = pathlib.Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)

    wandb.init(mode="online" if args.wandb_enabled else "disabled", project=args.wandb_project,
               name=args.wandb_name or output_dir.name, config=dataclasses.asdict(args), dir=str(output_dir),
               id=args.wandb_id, resume="allow" if args.wandb_id else None)

    train_loader = ShardStream(args.train_dir, args.batch_size, shuffle=True, seed=args.seed,
                               max_frames=args.max_train_samples)
    logging.info("train: %d shards, seq_len=%d, embed_dim=%d",
                 len(train_loader.paths), train_loader.seq_len, train_loader.embed_dim)
    val_loader = None
    if args.val_dir:
        val_loader = ShardStream(args.val_dir, args.batch_size, shuffle=False, seed=args.seed)
        assert val_loader.seq_len == train_loader.seq_len, "train/val seq_len mismatch"
        logging.info("val: %d shards", len(val_loader.paths))

    cfg = RLTokenAEConfig(vla_embed_dim=train_loader.embed_dim, d_model=args.d_model,
                          encoder_layers=args.encoder_layers, decoder_layers=args.decoder_layers,
                          num_heads=args.num_heads, max_seq_len=train_loader.seq_len)
    model = RLTokenAutoencoder(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logging.info("AE params: %.2fM", n_params / 1e6)
    wandb.config.update({"ae_cfg": dataclasses.asdict(cfg), "ae_params": n_params})

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.resume_from is not None:
        ckpt = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
            logging.info("Restored optimizer state (Adam moments + step)")
        else:
            logging.warning("No optimizer state in %s; Adam moments start from scratch", args.resume_from)
        logging.info("Resumed weights from %s; training %d further steps", args.resume_from, args.num_steps)

    log_f = (output_dir / "metrics.jsonl").open("w")

    def log(rec, step):
        log_f.write(json.dumps(rec) + "\n"); log_f.flush()
        wandb.log(rec, step=step)

    model.train()
    pbar = tqdm.tqdm(total=args.num_steps, desc="train")
    step = 0
    while step < args.num_steps:
        for z, m in train_loader:
            if step >= args.num_steps: break
            gstep = step + args.start_step
            z, m = z.to(device, non_blocking=True).float(), m.to(device, non_blocking=True)
            lr_now = _lr_at(step, args.warmup_steps, args.num_steps, args.lr, args.min_lr) if args.cosine_lr else args.lr
            for g in opt.param_groups: g["lr"] = lr_now
            out = model(z, m)
            opt.zero_grad(set_to_none=True); out["loss"].backward(); opt.step()

            if step % args.log_interval == 0:
                with torch.no_grad():
                    shuf = float(_masked_l2(out["z_hat"], z[torch.randperm(z.shape[0], device=device)], m))
                log({"step": gstep, "train_loss": float(out["loss"]), "shuffle_baseline": shuf, "lr": lr_now}, gstep)
                pbar.set_postfix(loss=f"{out['loss']:.4f}", shuf=f"{shuf:.4f}")

            if val_loader is not None and args.eval_interval > 0 and step > 0 and step % args.eval_interval == 0:
                model.eval()
                with torch.no_grad():
                    vl = float(np.mean([float(model(vz.to(device).float(), vm.to(device))["loss"])
                                        for vz, vm in val_loader]))
                model.train()
                log({"step": gstep, "val_loss": vl}, gstep)
                logging.info("step %d val_loss=%.4f", gstep, vl)

            step += 1; pbar.update(1)

    pbar.close(); log_f.close()
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                "cfg": dataclasses.asdict(cfg), "args": dataclasses.asdict(args)},
               output_dir / "ae_ckpt.pt")
    logging.info("Done. Checkpoint at %s/ae_ckpt.pt", output_dir)
    wandb.finish()


if __name__ == "__main__":
    main(tyro.cli(Args))
