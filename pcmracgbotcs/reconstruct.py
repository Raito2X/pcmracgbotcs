from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import torch
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cadlib.macro import (
    CMD_ARGS_MASK,
    EOS_IDX,
    EOS_VEC,
    N_ARGS,
)
from config.config_ae import AutoencoderConfig
from model.autoencoder import CADTransformer
from trainer.base import load_torch_checkpoint


def load_sequence(path: Path):
    with h5py.File(path, "r") as fp:
        attrs = {key: fp.attrs[key] for key in fp.attrs}
        if "commands" in fp and "args" in fp:
            commands = fp["commands"][:].astype(np.int64)
            args = fp["args"][:].astype(np.int64)
        else:
            raise KeyError(f"{path} must contain 'commands' and 'args'")
    if args.shape[1] < N_ARGS:
        args = np.pad(args, ((0, 0), (0, N_ARGS - args.shape[1])), constant_values=-1)
    return commands, args[:, :N_ARGS], attrs


def pad_sequence(commands, args, length=100):
    commands = commands[:length]
    args = args[:length]
    if len(commands) < length:
        commands = np.concatenate(
            [commands, np.repeat(EOS_VEC[0][None], length - len(commands), axis=0)]
        )
        args = np.concatenate(
            [args, np.repeat(EOS_VEC[None, 1:], length - len(args), axis=0)]
        )
    return commands, args


def canonicalize(commands, args):
    """Keep the first EOS and replace all later positions with padding values."""
    commands = commands.copy()
    args = args.copy()
    ends = np.where(commands == EOS_IDX)[0]
    if not len(ends):
        commands[-1] = EOS_IDX
        args[-1] = -1
        return commands, args
    end = int(ends[0])
    commands[end + 1:] = EOS_IDX
    args[end + 1:] = -1
    args[end] = -1
    return commands, args


def save_prediction(path: Path, commands, args, attrs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as fp:
        fp.create_dataset("commands", data=commands.astype(np.int32), compression="gzip")
        fp.create_dataset("args", data=args.astype(np.int16), compression="gzip")
        for key, value in attrs.items():
            try:
                fp.attrs[key] = value
            except TypeError:
                pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--split", default="data/split.json")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--phase", default="test", choices=["train", "validation", "val", "test"])
    parser.add_argument("--out_root", default="recon")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    cfg = SimpleNamespace(**AutoencoderConfig.model_config())
    model = CADTransformer(cfg).to(device)
    checkpoint = load_torch_checkpoint(Path(args.ckpt), device)
    state = checkpoint.get("model_state_dict", checkpoint)
    if all(key.startswith("module.") for key in state):
        state = {key[len("module."):]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()

    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    split_key = args.phase
    if split_key not in split and args.phase == "validation":
        split_key = "val"
    ids = split[split_key]
    vector_root = Path(args.data_root) / "cad_vec"
    output_root = Path(args.out_root) / args.phase

    for start in tqdm(range(0, len(ids), args.batch_size), desc="reconstruct"):
        batch_ids = ids[start:start + args.batch_size]
        raw = [load_sequence(vector_root / f"{data_id.replace('/', os.sep)}.h5") for data_id in batch_ids]
        padded = [pad_sequence(commands, sequence_args) for commands, sequence_args, _ in raw]
        commands = np.stack([item[0] for item in padded])
        sequence_args = np.stack([item[1] for item in padded])

        with torch.no_grad():
            output = model(
                torch.from_numpy(commands).to(device),
                torch.from_numpy(sequence_args).to(device),
                return_tgt=False,
            )
        predicted_commands = output["command_logits"].argmax(dim=-1).cpu().numpy()
        predicted_args = output["args_logits"].argmax(dim=-1).sub(1).cpu().numpy()

        # Clear slots that are not used by each predicted command before saving.
        argument_mask = CMD_ARGS_MASK[predicted_commands] == 1
        predicted_args[~argument_mask] = -1

        for data_id, command, sequence_arg, attributes in zip(
            batch_ids,
            predicted_commands,
            predicted_args,
            [item[2] for item in raw],
        ):
            command, sequence_arg = canonicalize(command, sequence_arg)
            save_prediction(
                output_root / f"{data_id}.h5",
                command,
                sequence_arg,
                attributes,
            )

    print(f"saved reconstructed sequences: {len(ids)}")


if __name__ == "__main__":
    main()
