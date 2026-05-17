"""ERA5-Conditioned Block Imputation Transformer."""

from __future__ import annotations

import torch
from torch import nn


class VariateTokenEncoder(nn.Module):
    """Encode one full time series per meteorological variable as a token."""

    def __init__(
        self,
        seq_len: int,
        n_vars: int,
        per_step_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.n_vars = n_vars
        self.per_step_dim = per_step_dim
        self.input_proj = nn.Linear(seq_len * per_step_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, n_vars, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x_var: torch.Tensor) -> torch.Tensor:
        """Encode flattened per-variable inputs.

        Args:
            x_var: `(B, C, T, F)` tensor.

        Returns:
            `(B, C, d_model)` encoded variable tokens.
        """
        if x_var.ndim != 4:
            raise ValueError(f"x_var must be (B, C, T, F), got {tuple(x_var.shape)}")
        bsz, n_vars, seq_len, per_step_dim = x_var.shape
        if n_vars != self.n_vars or seq_len != self.seq_len or per_step_dim != self.per_step_dim:
            raise ValueError(
                "Unexpected variate input shape: "
                f"got {(bsz, n_vars, seq_len, per_step_dim)}, "
                f"expected (*, {self.n_vars}, {self.seq_len}, {self.per_step_dim})"
            )
        z = self.input_proj(x_var.reshape(bsz, n_vars, seq_len * per_step_dim))
        z = z + self.pos_emb
        return self.encoder(z)


class ConditionalCrossAttention(nn.Module):
    """Cross-attend AWS variable tokens to ERA5 tokens with variable-level gating."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, z_obs: torch.Tensor, z_era5: torch.Tensor, missing_vars: torch.Tensor) -> torch.Tensor:
        if z_obs.shape != z_era5.shape:
            raise ValueError(f"z_obs shape {tuple(z_obs.shape)} != z_era5 shape {tuple(z_era5.shape)}")
        if missing_vars.shape != z_obs.shape[:2]:
            raise ValueError(f"missing_vars shape {tuple(missing_vars.shape)} incompatible with {tuple(z_obs.shape)}")
        cross_out, _ = self.cross_attn(query=z_obs, key=z_era5, value=z_era5, need_weights=False)
        gate = missing_vars.to(dtype=z_obs.dtype).unsqueeze(-1)
        return self.norm(z_obs + gate * self.dropout(cross_out))


class ECBIT(nn.Module):
    """ERA5-conditioned variate-token transformer for sparse block imputation."""

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
        use_era5: bool = True,
        use_cross: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.n_vars = n_vars
        self.n_time = n_time
        self.use_era5 = use_era5
        self.use_cross = use_cross

        self.obs_encoder = VariateTokenEncoder(
            seq_len=seq_len,
            n_vars=n_vars,
            per_step_dim=2 + n_time,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
        )
        if use_era5:
            self.era5_encoder = VariateTokenEncoder(
                seq_len=seq_len,
                n_vars=n_vars,
                per_step_dim=1 + n_time,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                d_ff=d_ff,
                dropout=dropout,
            )
            if use_cross:
                self.cond_attn = ConditionalCrossAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
            else:
                self.fuse = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model))

        self.out_norm = nn.LayerNorm(d_model)
        self.recon_head = nn.Linear(d_model, seq_len)

    def _check_inputs(
        self,
        x_obs: torch.Tensor,
        missing_mask: torch.Tensor,
        era5: torch.Tensor | None,
        time_enc: torch.Tensor,
    ) -> None:
        expected_x = (x_obs.shape[0], self.seq_len, self.n_vars)
        if tuple(x_obs.shape) != expected_x:
            raise ValueError(f"x_obs must be (B, {self.seq_len}, {self.n_vars}), got {tuple(x_obs.shape)}")
        if tuple(missing_mask.shape) != expected_x:
            raise ValueError(f"missing_mask shape {tuple(missing_mask.shape)} != x_obs shape {tuple(x_obs.shape)}")
        if tuple(time_enc.shape) != (x_obs.shape[0], self.seq_len, self.n_time):
            raise ValueError(f"time_enc must be (B, {self.seq_len}, {self.n_time}), got {tuple(time_enc.shape)}")
        if self.use_era5 and (era5 is None or tuple(era5.shape) != expected_x):
            got = None if era5 is None else tuple(era5.shape)
            raise ValueError(f"era5 must be (B, {self.seq_len}, {self.n_vars}), got {got}")

    def _obs_tokens(self, x_obs: torch.Tensor, missing_mask: torch.Tensor, time_enc: torch.Tensor) -> torch.Tensor:
        obs_mask = 1.0 - missing_mask.to(dtype=x_obs.dtype)
        x_var = x_obs.transpose(1, 2).unsqueeze(-1)
        mask_var = obs_mask.transpose(1, 2).unsqueeze(-1)
        time_var = time_enc.unsqueeze(1).expand(-1, self.n_vars, -1, -1)
        return torch.cat([x_var, mask_var, time_var], dim=-1)

    def _era5_tokens(self, era5: torch.Tensor, time_enc: torch.Tensor) -> torch.Tensor:
        era5_var = era5.transpose(1, 2).unsqueeze(-1)
        time_var = time_enc.unsqueeze(1).expand(-1, self.n_vars, -1, -1)
        return torch.cat([era5_var, time_var], dim=-1)

    def forward(
        self,
        x_obs: torch.Tensor,
        missing_mask: torch.Tensor,
        era5: torch.Tensor | None,
        time_enc: torch.Tensor,
    ) -> torch.Tensor:
        """Return reconstructed normalized variables `(B, T, C)`.

        `missing_mask` should include every unavailable model input position:
        original real missing values plus artificial block labels used for the
        current training/evaluation batch.
        """
        self._check_inputs(x_obs, missing_mask, era5, time_enc)
        z_obs = self.obs_encoder(self._obs_tokens(x_obs, missing_mask, time_enc))

        if self.use_era5:
            assert era5 is not None
            z_era5 = self.era5_encoder(self._era5_tokens(era5, time_enc))
            if self.use_cross:
                missing_vars = (missing_mask.bool().sum(dim=1) > 0).float()
                z = self.cond_attn(z_obs, z_era5, missing_vars)
            else:
                z = self.fuse(torch.cat([z_obs, z_era5], dim=-1))
        else:
            z = z_obs

        x_hat = self.recon_head(self.out_norm(z)).transpose(1, 2)
        return x_hat
