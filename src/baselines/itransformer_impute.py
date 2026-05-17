"""iTransformer-style imputation baseline without ERA5 conditioning."""

from __future__ import annotations

import torch
from torch import nn

from src.models.ecbit import VariateTokenEncoder


class ITransformerImputer(nn.Module):
    """Variate-token transformer that reconstructs sparse AWS windows only."""

    def __init__(
        self,
        seq_len: int = 168,
        n_vars: int = 5,
        n_time: int = 4,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.n_vars = n_vars
        self.n_time = n_time
        self.encoder = VariateTokenEncoder(
            seq_len=seq_len,
            n_vars=n_vars,
            per_step_dim=2 + n_time,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(d_model)
        self.recon_head = nn.Linear(d_model, seq_len)

    def forward(self, x_obs: torch.Tensor, missing_mask: torch.Tensor, time_enc: torch.Tensor) -> torch.Tensor:
        if tuple(x_obs.shape) != (x_obs.shape[0], self.seq_len, self.n_vars):
            raise ValueError(f"x_obs must be (B, {self.seq_len}, {self.n_vars}), got {tuple(x_obs.shape)}")
        if missing_mask.shape != x_obs.shape:
            raise ValueError(f"missing_mask shape {tuple(missing_mask.shape)} != x_obs shape {tuple(x_obs.shape)}")
        if tuple(time_enc.shape) != (x_obs.shape[0], self.seq_len, self.n_time):
            raise ValueError(f"time_enc must be (B, {self.seq_len}, {self.n_time}), got {tuple(time_enc.shape)}")

        obs_mask = 1.0 - missing_mask.to(dtype=x_obs.dtype)
        x_var = x_obs.transpose(1, 2).unsqueeze(-1)
        mask_var = obs_mask.transpose(1, 2).unsqueeze(-1)
        time_var = time_enc.unsqueeze(1).expand(-1, self.n_vars, -1, -1)
        z = self.encoder(torch.cat([x_var, mask_var, time_var], dim=-1))
        return self.recon_head(self.norm(z)).transpose(1, 2)


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, label_mask: torch.Tensor) -> torch.Tensor:
    """MSE over artificially hidden observed positions only."""
    if pred.shape != target.shape or pred.shape != label_mask.shape:
        raise ValueError(
            f"pred, target, label_mask must share shape, got {tuple(pred.shape)}, "
            f"{tuple(target.shape)}, {tuple(label_mask.shape)}"
        )
    denom = label_mask.sum().clamp_min(1.0)
    return (((pred - target) ** 2) * label_mask).sum() / denom
