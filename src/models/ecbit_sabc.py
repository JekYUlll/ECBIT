"""Station-adaptive ECBIT variant for exploratory ERA5 bias correction."""

from __future__ import annotations

import torch
from torch import nn

from src.models.ecbit import ECBIT


class StationAdaptiveBiasCorrection(nn.Module):
    """Apply a small station-conditioned residual correction to ERA5 tokens."""

    def __init__(
        self,
        d_model: int,
        n_vars: int,
        n_station_features: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        residual_scale_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_vars = n_vars
        self.n_station_features = n_station_features
        self.station_proj = nn.Sequential(
            nn.Linear(n_station_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )
        self.variable_emb = nn.Parameter(torch.randn(1, n_vars, d_model) * 0.02)
        self.delta = nn.Sequential(
            nn.Linear(d_model * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))

    def forward(self, z_era5: torch.Tensor, station_features: torch.Tensor) -> torch.Tensor:
        if z_era5.ndim != 3:
            raise ValueError(f"z_era5 must be (B, C, D), got {tuple(z_era5.shape)}")
        bsz, n_vars, d_model = z_era5.shape
        if n_vars != self.n_vars or d_model != self.d_model:
            raise ValueError(
                f"z_era5 must be (B, {self.n_vars}, {self.d_model}), got {tuple(z_era5.shape)}"
            )
        if station_features.shape != (bsz, self.n_station_features):
            raise ValueError(
                "station_features must be "
                f"(B, {self.n_station_features}), got {tuple(station_features.shape)}"
            )

        station_token = self.station_proj(station_features.to(dtype=z_era5.dtype)).unsqueeze(1)
        station_token = station_token.expand(-1, n_vars, -1)
        variable_token = self.variable_emb.expand(bsz, -1, -1)
        delta = self.delta(torch.cat([z_era5, station_token, variable_token], dim=-1))
        scale = torch.tanh(self.residual_scale)
        return z_era5 + scale * delta


class ECBITSABC(ECBIT):
    """ECBIT with an exploratory station-adaptive correction before fusion."""

    uses_station_features = True

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
        fusion_type: str = "gated",
        n_station_features: int = 9,
        sabc_hidden_dim: int = 64,
        sabc_dropout: float | None = None,
        sabc_residual_scale_init: float = 0.1,
    ) -> None:
        super().__init__(
            seq_len=seq_len,
            n_vars=n_vars,
            n_time=n_time,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            use_era5=True,
            use_cross=fusion_type == "cross_attn",
            fusion_type=fusion_type,
        )
        self.sabc = StationAdaptiveBiasCorrection(
            d_model=d_model,
            n_vars=n_vars,
            n_station_features=n_station_features,
            hidden_dim=sabc_hidden_dim,
            dropout=dropout if sabc_dropout is None else sabc_dropout,
            residual_scale_init=sabc_residual_scale_init,
        )

    def forward(
        self,
        x_obs: torch.Tensor,
        missing_mask: torch.Tensor,
        era5: torch.Tensor | None,
        time_enc: torch.Tensor,
        station_features: torch.Tensor,
    ) -> torch.Tensor:
        self._check_inputs(x_obs, missing_mask, era5, time_enc)
        z_obs = self.obs_encoder(self._obs_tokens(x_obs, missing_mask, time_enc))

        assert era5 is not None
        z_era5 = self.era5_encoder(self._era5_tokens(era5, time_enc))
        z_era5 = self.sabc(z_era5, station_features)
        missing_vars = (missing_mask.bool().sum(dim=1) > 0).float()
        if self.fusion_type == "cross_attn":
            z = self.cond_attn(z_obs, z_era5, missing_vars)
        elif self.fusion_type == "gated":
            z = self.gated_inject(z_obs, z_era5, missing_vars)
        elif self.fusion_type == "concat":
            z = self.fuse(torch.cat([z_obs, z_era5], dim=-1))
        else:
            z = z_obs

        x_hat = self.recon_head(self.out_norm(z)).transpose(1, 2)
        return x_hat
