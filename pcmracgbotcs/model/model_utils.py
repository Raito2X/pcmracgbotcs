from cadlib.macro import EOS_IDX


def to_sequence_first(*args):
    converted = []
    for tensor in args:
        converted.append(
            tensor.permute(1, 0, *range(2, tensor.dim()))
            if tensor is not None
            else None
        )
    return converted if len(converted) > 1 else converted[0]


def to_batch_first(*args):
    converted = []
    for tensor in args:
        converted.append(
            tensor.permute(1, 0, *range(2, tensor.dim()))
            if tensor is not None
            else None
        )
    return converted if len(converted) > 1 else converted[0]


def padding_mask(commands, sequence_dim=0):
    """Mark positions before the first EOS; the EOS itself is not pooled."""
    end = (commands == EOS_IDX).cumsum(dim=sequence_dim)
    valid = end == 0
    return valid.float().unsqueeze(-1)


def key_padding_mask(commands, sequence_dim=0):
    """Mask the first EOS and all following padding positions in self-attention."""
    if sequence_dim != 0:
        raise ValueError("key padding masks are built for sequence-first tensors")
    masked = (commands == EOS_IDX).cumsum(dim=0) > 0
    return masked.transpose(0, 1)
