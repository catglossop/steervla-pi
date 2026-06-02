"""RL Token autoencoder (Physical Intelligence's RLT, https://www.pi.website/research/rlt).

Encoder: bidirectional transformer over [z_{1:M}, e_rl]; RL token z_rl is the
encoder output at the appended e_rl position. Decoder: causal transformer with
z_rl prepended; at position i, predicts z̄_{i+1} from [z_rl, z̄_{1:i}].
Linear head projects back to the VLA embedding dim. L2 loss on stop-grad targets.
"""
import dataclasses

import torch
import torch.nn as nn


@dataclasses.dataclass
class RLTokenAEConfig:
    vla_embed_dim: int = 2048   # SigLIP+Gemma-2B prefix-token width
    d_model: int = 512
    encoder_layers: int = 4
    decoder_layers: int = 4
    num_heads: int = 8
    ffn_mult: int = 4
    max_seq_len: int = 648      # matches compacted dump: 256 vision + 200 prompt + 64*3 cot/fast
    dropout: float = 0.0


def _make_block(c: RLTokenAEConfig) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(
        d_model=c.d_model, nhead=c.num_heads, dim_feedforward=c.ffn_mult * c.d_model,
        dropout=c.dropout, activation="gelu", batch_first=True, norm_first=True,
    )


class RLTokenAutoencoder(nn.Module):
    def __init__(self, cfg: RLTokenAEConfig = RLTokenAEConfig()):
        super().__init__()
        self.cfg = cfg
        self.enc_in = nn.Linear(cfg.vla_embed_dim, cfg.d_model)
        self.dec_in = nn.Linear(cfg.vla_embed_dim, cfg.d_model)
        self.rl_query = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
        self.enc_pos = nn.Parameter(torch.randn(1, cfg.max_seq_len + 1, cfg.d_model) * 0.02)
        self.dec_pos = nn.Parameter(torch.randn(1, cfg.max_seq_len + 1, cfg.d_model) * 0.02)
        self.encoder = nn.TransformerEncoder(_make_block(cfg), cfg.encoder_layers, nn.LayerNorm(cfg.d_model))
        self.decoder = nn.TransformerEncoder(_make_block(cfg), cfg.decoder_layers, nn.LayerNorm(cfg.d_model))
        self.out_proj = nn.Linear(cfg.d_model, cfg.vla_embed_dim)

    def encode(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # z: [B, M, D_vla]; mask: [B, M] (True = real). Returns z_rl: [B, d_model].
        B, M, _ = z.shape
        x = self.enc_in(z)
        rl = self.rl_query.expand(B, -1, -1)
        seq = torch.cat([x, rl], dim=1) + self.enc_pos[:, : M + 1]
        rl_m = torch.ones(B, 1, dtype=torch.bool, device=z.device)
        out = self.encoder(seq, src_key_padding_mask=~torch.cat([mask, rl_m], dim=1))
        return out[:, -1, :]

    def decode(self, z_rl: torch.Tensor, z_targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Teacher-forced AR: input = [z_rl, proj(z̄_1), ..., proj(z̄_{M-1})], length M.
        # Causal mask -> position i predicts z̄_{i+1}; M outputs align with z_targets.
        B, M, _ = z_targets.shape
        ctx = self.dec_in(z_targets[:, :-1, :])
        seq = torch.cat([z_rl.unsqueeze(1), ctx], dim=1) + self.dec_pos[:, :M]
        causal = torch.triu(torch.full((M, M), float("-inf"), device=z_rl.device), diagonal=1)
        rl_m = torch.ones(B, 1, dtype=torch.bool, device=z_rl.device)
        kp = ~torch.cat([rl_m, mask[:, :-1]], dim=1)
        out = self.decoder(seq, mask=causal, src_key_padding_mask=kp)
        return self.out_proj(out)

    def forward(self, z: torch.Tensor, mask: torch.Tensor):
        z_targets = z.detach()  # stop-grad on reconstruction targets (paper: z̄_i = sg(z_i))
        z_rl = self.encode(z, mask)
        z_hat = self.decode(z_rl, z_targets, mask)
        per_token = (z_hat - z_targets).pow(2).sum(dim=-1)
        mf = mask.to(per_token.dtype)
        loss = (per_token * mf).sum() / mf.sum()
        return {"loss": loss, "z_rl": z_rl, "z_hat": z_hat, "per_token_l2": per_token.detach()}
