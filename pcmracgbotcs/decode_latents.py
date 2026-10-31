from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from config.config_ae import AutoencoderConfig
from model.autoencoder import CADTransformer
from trainer.base import load_torch_checkpoint
from cadlib.macro import CMD_ARGS_MASK, EOS_IDX


def load_autoencoder(path: Path, device):
    cfg = SimpleNamespace(**AutoencoderConfig.model_config())
    model = CADTransformer(cfg).to(device)
    checkpoint = load_torch_checkpoint(path, device)
    state = checkpoint.get("model_state_dict", checkpoint)
    if all(key.startswith("module.") for key in state):
        state = {key[len("module."):]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def decode(model, latents, device):
    output = model(None, None, latent=latents[:, None].to(device), return_tgt=False)
    commands = output["command_logits"].argmax(dim=-1)
    args = output["args_logits"].argmax(dim=-1) - 1
    args[~torch.tensor(CMD_ARGS_MASK, device=device, dtype=torch.bool)[commands]] = -1
    return commands.cpu(), args.cpu()


def save_samples(commands, args, output_dir: Path, start_index: int, keep_padding: bool):
    output_dir.mkdir(parents=True, exist_ok=True)
    for offset, (command, argument) in enumerate(zip(commands, args)):
        if not keep_padding:
            ends = torch.where(command == EOS_IDX)[0]
            end = int(ends[0]) + 1 if len(ends) else len(command)
            command = command[:end]
            argument = argument[:end]
        with h5py.File(output_dir / f"sample_{start_index + offset:05d}.h5", "w") as fp:
            fp.create_dataset("commands", data=command.numpy().astype(np.int64))
            fp.create_dataset("args", data=argument.numpy().astype(np.int64))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_path", required=True)
    parser.add_argument("--ae_ckpt_path", required=True)
    parser.add_argument("--output_dir")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gpu_ids", default="0")
    parser.add_argument("--keep_padding", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu_ids != "cpu" else "cpu")
    model = load_autoencoder(Path(args.ae_ckpt_path), device)
    latent_path = Path(args.latent_path)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else latent_path.with_name(latent_path.stem + "_decoded")
    )

    with h5py.File(latent_path, "r") as fp:
        latents = torch.from_numpy(fp["zs"][:].astype(np.float32))

    generated = 0
    for start in range(0, len(latents), args.batch_size):
        commands, decoded_args = decode(
            model,
            latents[start:start + args.batch_size],
            device,
        )
        save_samples(
            commands,
            decoded_args,
            output_dir,
            generated,
            args.keep_padding,
        )
        generated += len(commands)
        print(f"decoded {generated}/{len(latents)}")


if __name__ == "__main__":
    main()
