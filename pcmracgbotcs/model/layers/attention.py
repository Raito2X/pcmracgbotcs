import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        nn.init.xavier_uniform_(self.in_proj_weight)

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
    ):
        target_length, batch_size, _ = query.shape
        q, k, v = F.linear(
            query,
            self.in_proj_weight,
            self.in_proj_bias,
        ).chunk(3, dim=-1)
        if key is not query or value is not query:
            k = F.linear(key, self.in_proj_weight[self.embed_dim:2 * self.embed_dim])
            v = F.linear(value, self.in_proj_weight[2 * self.embed_dim:])

        q = q * (self.head_dim ** -0.5)
        q = q.contiguous().view(
            target_length,
            batch_size * self.num_heads,
            self.head_dim,
        ).transpose(0, 1)
        k = k.contiguous().view(
            -1,
            batch_size * self.num_heads,
            self.head_dim,
        ).transpose(0, 1)
        v = v.contiguous().view(
            -1,
            batch_size * self.num_heads,
            self.head_dim,
        ).transpose(0, 1)
        scores = torch.bmm(q, k.transpose(1, 2))

        if attn_mask is not None:
            scores = scores + (
                attn_mask.unsqueeze(0) if attn_mask.dim() == 2 else attn_mask
            )
        if key_padding_mask is not None:
            scores = scores.view(
                batch_size,
                self.num_heads,
                target_length,
                -1,
            ).masked_fill(
                key_padding_mask[:, None, None, :],
                float("-inf"),
            ).view(
                batch_size * self.num_heads,
                target_length,
                -1,
            )

        weights = F.softmax(scores, dim=-1)
        weights = F.dropout(weights, self.dropout, training=self.training)
        output = torch.bmm(weights, v).transpose(0, 1).contiguous().view(
            target_length,
            batch_size,
            self.embed_dim,
        )
        output = self.out_proj(output)
        if need_weights:
            weights = weights.view(
                batch_size,
                self.num_heads,
                target_length,
                -1,
            ).mean(dim=1)
        else:
            weights = None
        return output, weights
