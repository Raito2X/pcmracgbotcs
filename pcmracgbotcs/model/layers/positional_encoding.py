import torch
import torch.nn as nn


class PositionalEncodingLUT(nn.Module):
    def __init__(self, width: int, dropout: float = 0.1, max_len: int = 250):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len, dtype=torch.long).unsqueeze(1)
        self.register_buffer("position", position)
        self.pos_embed = nn.Embedding(max_len, width)
        nn.init.kaiming_normal_(self.pos_embed.weight, mode="fan_in")

    def forward(self, tensor):
        positions = self.position[: tensor.size(0)]
        return self.dropout(tensor + self.pos_embed(positions))
