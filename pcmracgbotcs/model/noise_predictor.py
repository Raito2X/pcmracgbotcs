from __future__ import annotations

import math

import torch
import torch.nn as nn


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000):
    """Create sinusoidal timestep embeddings."""
    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    angles = timesteps[:, None].float() * frequencies[None]
    embedding = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class LayerNormWrapper(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)

    def forward(self, tensor):
        return self.norm(tensor)


class TransformerEncoder(nn.Module):
    """Process latents shaped as [batch, sequence length, hidden width]."""

    def __init__(self, width, ff_width, layers, dropout, heads):
        super().__init__()
        self.mhsa = nn.ModuleList(
            [
                nn.MultiheadAttention(width, heads, dropout, batch_first=False)
                for _ in range(layers)
            ]
        )
        self.norm1 = nn.ModuleList(
            [LayerNormWrapper(width) for _ in range(layers)]
        )
        self.norm2 = nn.ModuleList(
            [LayerNormWrapper(width) for _ in range(layers)]
        )
        self.ffn = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(width, ff_width),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(ff_width, width),
                )
                for _ in range(layers)
            ]
        )

    def forward(self, hidden):
        hidden = hidden.transpose(0, 1)
        for index in range(len(self.mhsa)):
            attention = self.mhsa[index](
                hidden,
                hidden,
                hidden,
                need_weights=False,
            )[0]
            hidden = self.norm1[index](attention + hidden)
            hidden = self.norm2[index](self.ffn[index](hidden) + hidden)
        return hidden.transpose(0, 1)


class LatentNoisePredictor(nn.Module):
    """Fuse timestep and latent information, then predict the added noise."""

    def __init__(self, cfg):
        super().__init__()
        self.input_up_proj = nn.Sequential(
            nn.Linear(cfg.z_dim, cfg.h_dim),
            nn.Tanh(),
            nn.Linear(cfg.h_dim, cfg.h_dim),
        )
        self.time_emb = nn.Sequential(
            nn.Linear(cfg.h_dim, cfg.h_dim * 4),
            nn.SiLU(),
            nn.Linear(cfg.h_dim * 4, cfg.h_dim),
        )
        self.output_down_proj = nn.Sequential(
            nn.Linear(cfg.h_dim, cfg.h_dim),
            nn.Tanh(),
            nn.Linear(cfg.h_dim, cfg.z_dim),
        )
        self.dropout = nn.Dropout(cfg.dropout_rate)
        self.layernorm = LayerNormWrapper(cfg.h_dim)
        self.encoder = TransformerEncoder(
            cfg.h_dim,
            cfg.ff_dim,
            cfg.num_enc_layers,
            cfg.dropout_rate,
            cfg.num_heads,
        )

    def forward(self, sequence: torch.Tensor, timesteps: torch.Tensor):
        context = self.time_emb(
            timestep_embedding(timesteps, self.time_emb[0].in_features)
        )
        hidden = self.input_up_proj(sequence)

        # Broadcast the timestep vector to every latent token.
        hidden = self.dropout(self.layernorm(hidden + context[:, None, :]))
        return self.output_down_proj(self.encoder(hidden))
