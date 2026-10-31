from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

from cadlib.macro import ALL_COMMANDS, ARGS_DIM, MAX_TOTAL_LEN, N_ARGS


class AutoencoderConfig:
    def __init__(self):
        self.set_configuration()
        _, args = self.parse()
        for key, value in vars(args).items():
            setattr(self, key, value)

        self.exp_dir = os.path.join(self.proj_dir, self.exp_name)
        self.log_dir = os.path.join(self.exp_dir, "log")
        self.model_dir = os.path.join(self.exp_dir, "model")

        if not self.cont and os.path.exists(self.exp_dir):
            answer = input(f"Experiment exists: {self.exp_dir}. Overwrite? (y/n) ")
            if answer.lower() != "y":
                raise SystemExit
            shutil.rmtree(self.exp_dir)

        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        if self.gpu_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_ids)
        self.device = "cuda" if self.gpu_ids != "cpu" else "cpu"

        serializable = {
            key: value
            for key, value in self.__dict__.items()
            if isinstance(value, (str, int, float, bool, list, dict, type(None)))
        }
        with open(os.path.join(self.exp_dir, "config.json"), "w", encoding="utf-8") as fp:
            json.dump(serializable, fp, ensure_ascii=False, indent=2)

    @staticmethod
    def model_config() -> dict:
        return {
            "args_dim": ARGS_DIM,
            "n_args": N_ARGS,
            "n_commands": len(ALL_COMMANDS),
            "n_layers": 4,
            "n_layers_decode": 4,
            "n_heads": 8,
            "dim_feedforward": 512,
            "d_model": 256,
            "dropout": 0.1,
            "dim_z": 256,
            "max_total_len": MAX_TOTAL_LEN,
            "loss_weights": {"loss_cmd_weight": 1.0, "loss_args_weight": 1.0},
        }

    def set_configuration(self):
        for key, value in self.model_config().items():
            setattr(self, key, value)

    def parse(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--proj_dir", default="runs")
        parser.add_argument("--data_root", default="data")
        parser.add_argument("--split_path", default="data/split.json")
        parser.add_argument("--exp_name", default="mcl")
        parser.add_argument("--gpu_ids", default="0")
        parser.add_argument("--batch_size", type=int, default=300)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--nr_epochs", type=int, default=1000)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--grad_clip", type=float, default=1.0)
        parser.add_argument("--warmup_step", type=int, default=2000)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--continue", dest="cont", action="store_true")
        parser.add_argument("--ckpt", default="latest")
        parser.add_argument("--save_frequency", type=int, default=50)
        parser.add_argument("--val_frequency", type=int, default=10)
        args = parser.parse_args()
        return parser, args
