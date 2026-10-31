from __future__ import annotations

import torch
import torch.nn as nn

from model.noise_predictor import timestep_embedding


class OperationPresenceClassifier(nn.Module):
    """Predict which solid-generation operations occur in a command sequence."""

    def __init__(self, z_dim: int = 256, h_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.z_proj = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
        )
        self.t_proj = nn.Sequential(
            nn.Linear(h_dim, h_dim * 4),
            nn.SiLU(),
            nn.Linear(h_dim * 4, h_dim),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(h_dim),
            nn.Dropout(dropout),
            nn.Linear(h_dim, h_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim, 3),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor):
        if x_t.dim() == 3:
            x_t = x_t[:, 0, :] if x_t.size(1) == 1 else x_t.mean(dim=1)
        if x_t.dim() != 2:
            raise ValueError(f"expected (B,Z) or (B,1,Z), got {tuple(x_t.shape)}")
        width = self.z_proj[0].out_features
        return self.head(
            self.z_proj(x_t) + self.t_proj(timestep_embedding(t, width))
        )
