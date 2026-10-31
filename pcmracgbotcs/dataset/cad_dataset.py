import json
import os
import random

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from cadlib.macro import EOS_VEC, MAX_TOTAL_LEN


def seed_worker(_):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def get_dataloader(phase, cfg, shuffle=None):
    dataset = CADDataset(phase, cfg)
    shuffle = phase == "train" if shuffle is None else bool(shuffle)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        drop_last=phase == "train",
        worker_init_fn=seed_worker,
    )


class CADDataset(Dataset):
    def __init__(self, phase, cfg):
        self.data_root = cfg.data_root
        self.max_total_len = getattr(cfg, "max_total_len", MAX_TOTAL_LEN)
        if not getattr(cfg, "split_path", None):
            raise ValueError("split_path must be provided explicitly")
        split_path = cfg.split_path
        with open(split_path, "r", encoding="utf-8") as fp:
            split = json.load(fp)
        if phase not in split and phase == "validation":
            phase = "val"
        elif phase not in split and phase == "val":
            phase = "validation"
        if phase not in split:
            raise KeyError(f"split has no phase={phase}")
        self.ids = split[phase]
        self.vector_root = os.path.join(self.data_root, "cad_vec")

    def __len__(self):
        return len(self.ids)

    def _read(self, path):
        with h5py.File(path, "r") as fp:
            if "commands" in fp and "args" in fp:
                commands = fp["commands"][:].astype(np.int64)
                args = fp["args"][:].astype(np.int64)
            else:
                raise KeyError(f"{path} must contain 'commands' and 'args'")
        return np.column_stack([commands, args])

    def __getitem__(self, index):
        data_id = self.ids[index]
        vector = self._read(
            os.path.join(self.vector_root, data_id.replace("/", os.sep) + ".h5")
        )
        vector = vector[: self.max_total_len]
        if len(vector) < self.max_total_len:
            vector = np.vstack(
                [
                    vector,
                    np.repeat(EOS_VEC[None], self.max_total_len - len(vector), axis=0),
                ]
            )
        return {
            "command": torch.from_numpy(vector[:, 0]),
            "args": torch.from_numpy(vector[:, 1:]),
            "id": data_id,
        }
