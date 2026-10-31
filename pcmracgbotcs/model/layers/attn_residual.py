import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root-mean-square normalization over the last dimension."""

    def __init__(self, width: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, tensor):
        scale = tensor.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return tensor * scale * self.weight


class FullAttnRes(nn.Module):
    """Apply attention over history depth and return the shape of one history tensor."""

    def __init__(self, width: int):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(width))
        self.key_norm = RMSNorm(width)

    def forward(self, sources):
        if not sources:
            raise ValueError("At least one source tensor is required.")
        shapes = {source.shape for source in sources}
        if len(shapes) != 1:
            raise ValueError(f"Source shapes differ: {sorted(shapes)}")

        values = torch.stack(sources, dim=0)
        keys = self.key_norm(values)

        # M is history depth, S is sequence length, and N is batch size.
        logits = torch.einsum("d,msnd->msn", self.query, keys)
        weights = torch.softmax(logits, dim=0)
        return torch.einsum("msn,msnd->snd", weights, values)
