from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from cadlib.macro import EOS_IDX, PRISM_IDX, REVOL_IDX, THRUSECTIONS_IDX


OP_NAME_TO_ID = {"prism": 0, "revol": 1, "thrusections": 2}
OP_ID_TO_NAME = {value: key for key, value in OP_NAME_TO_ID.items()}
OP_BIT_NAMES = ("has_prism", "has_revol", "has_thrusections")
OP_COMMANDS = (PRISM_IDX, REVOL_IDX, THRUSECTIONS_IDX)


def seed_worker(_):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _split_key(split: dict, phase: str) -> str:
    if phase in split:
        return phase
    if phase == "validation" and "val" in split:
        return "val"
    if phase == "val" and "validation" in split:
        return "validation"
    raise KeyError(f"split has no phase={phase}; available keys={list(split)}")


def _z_key(phase: str) -> str:
    return "validation_zs" if phase in {"val", "validation"} else f"{phase}_zs"


def _read_commands(path: Path) -> np.ndarray:
    """Read a command sequence through the first EOS, including that EOS."""
    with h5py.File(path, "r") as fp:
        if "commands" in fp:
            commands = fp["commands"][:].astype(np.int64)
        else:
            raise KeyError(f"{path} must contain 'commands'")
    end = np.where(commands == EOS_IDX)[0]
    return commands[: end[0] + 1] if len(end) else commands


def operation_presence(commands: np.ndarray) -> Tuple[np.ndarray | None, int | None]:
    """Return Prism, Revol, and Thrusections presence bits and first operation."""
    bits = np.asarray([int(np.any(commands == command)) for command in OP_COMMANDS])
    if bits.sum() == 0:
        return None, None
    first_op = next(
        OP_COMMANDS.index(int(command))
        for command in commands
        if int(command) in OP_COMMANDS
    )
    return bits.astype(np.int64), first_op


class CADLatentDataset(Dataset):
    def __init__(self, cfg, phase: str = "train"):
        self.phase = phase
        self.z_path = Path(cfg.z_path)
        self.data_root = Path(cfg.cad_data_root)
        self.cad_vec_root = self.data_root / "cad_vec"
        split_path = getattr(cfg, "split_path", None)
        if not split_path:
            raise ValueError("split_path must be provided explicitly")
        split_path = Path(split_path)

        if not self.z_path.exists():
            raise FileNotFoundError(f"latent file not found: {self.z_path}")
        if not split_path.exists():
            raise FileNotFoundError(f"split file not found: {split_path}")

        with split_path.open("r", encoding="utf-8") as fp:
            split = json.load(fp)
        ids = split[_split_key(split, phase)]

        with h5py.File(self.z_path, "r") as fp:
            key = _z_key(phase)
            if key not in fp or "train_zs" not in fp:
                raise KeyError(f"{self.z_path} must contain '{key}' and 'train_zs'")
            latents = fp[key][:].astype(np.float32)
            train_latents = fp["train_zs"][:].astype(np.float32)

        if len(ids) != len(latents):
            raise ValueError(
                f"split/latent count mismatch for {phase}: {len(ids)} != {len(latents)}"
            )

        self.z_mean = train_latents.mean(axis=0).astype(np.float32)
        self.z_std = train_latents.std(axis=0).astype(np.float32)
        self.z_std[self.z_std < 1e-6] = 1.0

        keep, bits, first_ops = [], [], []
        for index, data_id in enumerate(ids):
            h5_path = self.cad_vec_root / f"{data_id.replace('/', os.sep)}.h5"
            if not h5_path.exists():
                raise FileNotFoundError(f"CAD sequence not found: {h5_path}")
            presence, first_op = operation_presence(_read_commands(h5_path))
            if presence is None:
                continue
            keep.append(index)
            bits.append(presence)
            first_ops.append(first_op)

        if not keep:
            raise RuntimeError(f"no operation-labelled samples in {phase}")

        keep_array = np.asarray(keep, dtype=np.int64)
        self.ids = [ids[index] for index in keep]
        self.zs = latents[keep_array]
        self.presence = np.stack(bits).astype(np.int64)
        self.first_ops = np.asarray(first_ops, dtype=np.int64)
        self._print_summary(len(ids))

    def _print_summary(self, raw_count: int):
        print(
            f"{self.phase}: {len(self.ids)}/{raw_count} samples, "
            f"latent shape={self.zs.shape}"
        )
        for index, name in enumerate(OP_BIT_NAMES):
            count = int(self.presence[:, index].sum())
            print(f"  {name}: {count} ({100.0 * count / len(self):.2f}%)")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        z = (self.zs[index] - self.z_mean) / self.z_std
        return (
            torch.tensor(z, dtype=torch.float32),
            torch.tensor(self.presence[index], dtype=torch.long),
        )

    def get_latent_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.z_mean.copy(), self.z_std.copy()


def get_dataloader(cfg, phase: str = "train", shuffle=None):
    dataset = CADLatentDataset(cfg, phase)
    shuffle = (phase == "train") if shuffle is None else bool(shuffle)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        drop_last=phase == "train",
        worker_init_fn=seed_worker,
    )
    return loader, dataset
