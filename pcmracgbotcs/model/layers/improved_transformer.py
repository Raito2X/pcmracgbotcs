from torch import nn

from .attention import MultiheadAttention
from .attn_residual import FullAttnRes
from .transformer import _get_activation_fn


def _expand_to_sequence(tensor, sequence_length):
    if tensor.size(0) == sequence_length:
        return tensor
    if tensor.size(0) == 1:
        return tensor.expand(sequence_length, -1, -1)
    raise ValueError(
        f"cannot expand shape {tuple(tensor.shape)} to length {sequence_length}"
    )


class AttnResEncoderLayer(nn.Module):
    """Encoder layer whose history stores all earlier sublayer outputs."""

    def __init__(self, width, heads, ff_width, dropout, activation="relu"):
        super().__init__()
        self.self_attn = MultiheadAttention(width, heads, dropout=dropout)
        self.linear1 = nn.Linear(width, ff_width)
        self.linear2 = nn.Linear(ff_width, width)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.res_attn_1 = FullAttnRes(width)
        self.res_attn_2 = FullAttnRes(width)

    def forward(self, history, src_mask=None, src_key_padding_mask=None):
        # Aggregate all previous representations before self-attention.
        hidden = self.norm1(self.res_attn_1(history))
        attention = self.self_attn(
            hidden,
            hidden,
            hidden,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )[0]
        history.append(self.dropout1(attention))

        # Aggregate the updated history again before the feed-forward network.
        hidden = self.norm2(self.res_attn_2(history))
        feed_forward = self.linear2(
            self.dropout(self.activation(self.linear1(hidden)))
        )
        history.append(self.dropout2(feed_forward))
        return history


class AttnResDecoderLayer(nn.Module):
    """Decoder layer that adds the global latent as an extra history source."""

    def __init__(self, width, latent_dim, heads, ff_width, dropout, activation="relu"):
        super().__init__()
        self.self_attn = MultiheadAttention(width, heads, dropout=dropout)
        self.linear_global = nn.Linear(latent_dim, width)
        self.linear1 = nn.Linear(width, ff_width)
        self.linear2 = nn.Linear(ff_width, width)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.res_attn_1 = FullAttnRes(width)
        self.res_attn_2 = FullAttnRes(width)
        self.res_attn_3 = FullAttnRes(width)

    def forward(self, history, memory, tgt_mask=None, tgt_key_padding_mask=None):
        sequence_length = history[0].size(0)
        hidden = self.norm1(self.res_attn_1(history))
        attention = self.self_attn(
            hidden,
            hidden,
            hidden,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        history.append(self.dropout1(attention))

        # Weight the current history before adding the global latent.
        hidden = self.res_attn_2(history)
        latent = self.linear_global(memory)
        conditioned = hidden + _expand_to_sequence(latent, sequence_length)
        history.append(self.dropout2(conditioned))

        hidden = self.norm2(self.res_attn_3(history))
        feed_forward = self.linear2(
            self.dropout(self.activation(self.linear1(hidden)))
        )
        history.append(self.dropout3(feed_forward))
        return history
