import copy

import torch.nn.functional as F
from torch import nn


class TransformerEncoder(nn.Module):
    """Collect encoder outputs and aggregate all history layers with attentional residual."""

    def __init__(self, layer, num_layers, norm=None, final_attn_res=None):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(num_layers)]
        )
        self.norm = norm
        self.final_attn_res = final_attn_res

    def forward(self, source, mask=None, src_key_padding_mask=None):
        history = [source]
        for layer in self.layers:
            history = layer(
                history,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
            )
        output = self.final_attn_res(history)
        return self.norm(output) if self.norm is not None else output


class TransformerDecoder(nn.Module):
    """Update fixed-position queries with the global latent and aggregate decoder history."""

    def __init__(self, layer, num_layers, norm=None, final_attn_res=None):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(num_layers)]
        )
        self.norm = norm
        self.final_attn_res = final_attn_res

    def forward(
        self,
        target,
        memory,
        tgt_mask=None,
        tgt_key_padding_mask=None,
    ):
        history = [target]
        for layer in self.layers:
            history = layer(
                history,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )
        output = self.final_attn_res(history)
        return self.norm(output) if self.norm is not None else output


def _get_activation_fn(name):
    if name == "relu":
        return F.relu
    if name == "gelu":
        return F.gelu
    raise ValueError(f"unsupported activation: {name}")
