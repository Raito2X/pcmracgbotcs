from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from config.config_ae import AutoencoderConfig
from dataset.cad_dataset import CADDataset
from model.autoencoder import CADTransformer
from trainer.base import load_torch_checkpoint


def infer_experiment(path: Path):
    model_dir = path.resolve().parent
    exp_dir = model_dir.parent
    proj_dir = exp_dir.parent
    return proj_dir, exp_dir.name, exp_dir, path.stem


def load_model(path: Path, cfg, device):
    model = CADTransformer(cfg).to(device)
    checkpoint = load_torch_checkpoint(path, device)
    state = checkpoint.get("model_state_dict", checkpoint)
    if all(key.startswith("module.") for key in state):
        state = {key[len("module."):]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def encode_loader(model, loader, device):
    chunks = []
    for data in loader:
        latent = model(
            data["command"].to(device),
            data["args"].to(device),
            encode_mode=True,
        )
        chunks.append(latent[:, 0].cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ae_ckpt_path", required=True)
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--split_path", default="data/split.json")
    parser.add_argument("--output_h5")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--gpu_ids", default="0")
    args = parser.parse_args()

    checkpoint_path = Path(args.ae_ckpt_path)
    _, _, exp_dir, checkpoint_tag = infer_experiment(checkpoint_path)
    output_path = Path(args.output_h5 or exp_dir / "results" / f"all_zs_ckpt{checkpoint_tag}.h5")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu_ids != "cpu" else "cpu")
    cfg = SimpleNamespace(
        **AutoencoderConfig.model_config(),
        data_root=args.data_root,
        split_path=args.split_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model = load_model(checkpoint_path, cfg, device)

    with h5py.File(output_path, "w") as fp:
        for phase, key in (("train", "train_zs"), ("validation", "validation_zs"), ("test", "test_zs")):
            dataset = CADDataset(phase, cfg)
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                drop_last=False,
            )
            latents = encode_loader(model, loader, device)
            fp.create_dataset(key, data=latents)
            print(f"encoded {phase}: {latents.shape}")

    print(f"saved latents: {output_path}")


if __name__ == "__main__":
    main()
