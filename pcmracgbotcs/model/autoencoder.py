import torch.nn as nn

from model.layers.attn_residual import FullAttnRes
from model.layers.improved_transformer import (
    AttnResDecoderLayer,
    AttnResEncoderLayer,
)
from model.layers.positional_encoding import PositionalEncodingLUT
from model.layers.transformer import TransformerDecoder, TransformerEncoder
from model.model_utils import (
    key_padding_mask,
    padding_mask,
    to_batch_first,
    to_sequence_first,
)


class CADEmbedding(nn.Module):
    """Embed commands, 17 parameter slots, and positions; add 1 to map -1 padding."""

    def __init__(self, cfg):
        super().__init__()
        self.command_embed = nn.Embedding(cfg.n_commands, cfg.d_model)
        self.arg_embed = nn.Embedding(cfg.args_dim + 1, 64, padding_idx=0)
        self.embed_fcn = nn.Linear(64 * cfg.n_args, cfg.d_model)
        self.pos_encoding = PositionalEncodingLUT(
            cfg.d_model,
            max_len=cfg.max_total_len + 2,
        )

    def forward(self, commands, args):
        sequence_length, batch_size = commands.shape
        embedded = self.command_embed(commands.long())
        embedded += self.embed_fcn(
            self.arg_embed((args + 1).long()).view(
                sequence_length,
                batch_size,
                -1,
            )
        )
        return self.pos_encoding(embedded)


class ConstEmbedding(nn.Module):
    """Provide fixed-position queries that do not depend on command inputs."""

    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg.d_model
        self.seq_len = cfg.max_total_len
        self.PE = PositionalEncodingLUT(cfg.d_model, max_len=self.seq_len)

    def forward(self, latent):
        batch_size = latent.size(1)
        source = latent.new_zeros(
            self.seq_len,
            batch_size,
            self.d_model,
        )
        return self.PE(source)


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embedding = CADEmbedding(cfg)
        layer = AttnResEncoderLayer(
            cfg.d_model,
            cfg.n_heads,
            cfg.dim_feedforward,
            cfg.dropout,
        )
        self.encoder = TransformerEncoder(
            layer,
            cfg.n_layers,
            nn.LayerNorm(cfg.d_model),
            final_attn_res=FullAttnRes(cfg.d_model),
        )

    def forward(self, commands, args):
        # Pool only real command positions; EOS and padding stay out of the latent.
        valid = padding_mask(commands)
        masked = key_padding_mask(commands)
        memory = self.encoder(
            self.embedding(commands, args),
            src_key_padding_mask=masked,
        )
        return (memory * valid).sum(dim=0, keepdim=True) / valid.sum(
            dim=0,
            keepdim=True,
        )


class OutputHead(nn.Module):
    def __init__(self, width, n_commands, n_args, args_dim):
        super().__init__()
        self.n_args = n_args
        self.args_dim = args_dim
        self.command_fcn = nn.Linear(width, n_commands)
        self.args_fcn = nn.Linear(width, n_args * args_dim)

    def forward(self, hidden):
        sequence_length, batch_size, _ = hidden.shape
        command_logits = self.command_fcn(hidden)
        args_logits = self.args_fcn(hidden).reshape(
            sequence_length,
            batch_size,
            self.n_args,
            self.args_dim,
        )
        return command_logits, args_logits


class Decoder(nn.Module):
    """Decode a global latent into queries for every output position."""

    def __init__(self, cfg):
        super().__init__()
        self.embedding = ConstEmbedding(cfg)
        layer = AttnResDecoderLayer(
            cfg.d_model,
            cfg.dim_z,
            cfg.n_heads,
            cfg.dim_feedforward,
            cfg.dropout,
        )
        self.decoder = TransformerDecoder(
            layer,
            cfg.n_layers_decode,
            nn.LayerNorm(cfg.d_model),
            final_attn_res=FullAttnRes(cfg.d_model),
        )
        self.fcn = OutputHead(
            cfg.d_model,
            cfg.n_commands,
            cfg.n_args,
            cfg.args_dim + 1,
        )

    def forward(self, latent):
        decoded = self.decoder(self.embedding(latent), latent)
        return self.fcn(decoded)


class Bottleneck(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.dim_z),
            nn.Tanh(),
        )

    def forward(self, latent):
        return self.bottleneck(latent)


class CADTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = Encoder(cfg)
        self.bottleneck = Bottleneck(cfg)
        self.decoder = Decoder(cfg)

    def forward(
        self,
        commands,
        args,
        latent=None,
        return_tgt=True,
        encode_mode=False,
    ):
        target_commands = commands
        target_args = args
        if latent is None:
            source_commands, source_args = to_sequence_first(commands, args)
            latent = self.bottleneck(self.encoder(source_commands, source_args))
        else:
            # Use a supplied latent directly and skip the encoder.
            latent = to_sequence_first(latent)

        if encode_mode:
            return to_batch_first(latent)

        commands_logits, args_logits = to_batch_first(*self.decoder(latent))
        output = {
            "command_logits": commands_logits,
            "args_logits": args_logits,
        }
        if return_tgt:
            output["tgt_commands"] = target_commands
            output["tgt_args"] = target_args
        return output
