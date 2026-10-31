from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil


class DiffusionConfig:
    def __init__(self, mode: str = "train"):
        if mode not in {"train", "classifier", "sample"}:
            raise ValueError(f"unknown config mode: {mode}")
        self.mode = mode
        self.set_configuration()
        _, args = self.parse()
        for key, value in vars(args).items():
            setattr(self, key, value)

        if self.z_path is None:
            self.z_path = os.path.join(
                self.proj_dir,
                self.exp_name,
                "results",
                f"all_zs_ckpt{self.ae_ckpt}.h5",
            )

        self.exp_dir = os.path.join(
            self.proj_dir,
            self.exp_name,
            f"latent_diffusion_{self.ae_ckpt}",
        )
        self.log_dir = os.path.join(self.exp_dir, "log")
        self.model_dir = os.path.join(self.exp_dir, "model")
        self.result_dir = os.path.join(self.exp_dir, "results")
        self.classifier_model_dir = os.path.join(self.exp_dir, "classifier_model")

        if mode == "train" and not self.cont and os.path.exists(self.exp_dir):
            answer = input(f"Experiment exists: {self.exp_dir}. Overwrite? (y/n) ")
            if answer.lower() != "y":
                raise SystemExit
            shutil.rmtree(self.exp_dir)

        for path in (
            self.log_dir,
            self.model_dir,
            self.result_dir,
            self.classifier_model_dir,
        ):
            Path(path).mkdir(parents=True, exist_ok=True)

        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_ids)
        self.device = "cuda" if self.gpu_ids != "cpu" else "cpu"

        if mode == "train":
            serializable = {
                key: value
                for key, value in self.__dict__.items()
                if isinstance(value, (str, int, float, bool, list, dict, type(None)))
            }
            with open(os.path.join(self.exp_dir, "config.json"), "w", encoding="utf-8") as fp:
                json.dump(serializable, fp, ensure_ascii=False, indent=2)

    def set_configuration(self):
        self.z_dim = 256
        self.seq_len = 1
        self.h_dim = 512
        self.ff_dim = 1024
        self.num_enc_layers = 4
        self.num_heads = 8
        self.dropout_rate = 0.1
        self.timesteps = 1000
        self.batch_size = 256
        self.num_workers = 4
        self.n_iters = 100000
        self.lr = 2e-4
        self.beta1 = 0.9
        self.grad_clip = 1.0
        self.save_frequency = 5000
        self.log_frequency = 100
        self.classifier_n_iters = 10000
        self.classifier_lr = 2e-4
        self.classifier_save_frequency = 5000
        self.classifier_log_frequency = 100
        self.classifier_ckpt = None
        self.guidance_scale = 1.0
        self.guidance_start_t = 500
        self.seed = 0

    def parse(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--proj_dir", default="runs")
        parser.add_argument("--exp_name", default="mcl")
        parser.add_argument("--ae_ckpt", required=True)
        parser.add_argument("--z_path", default=None)
        parser.add_argument("--cad_data_root", default="data")
        parser.add_argument("--split_path", default="data/split.json")
        parser.add_argument("--batch_size", type=int, default=self.batch_size)
        parser.add_argument("--num_workers", type=int, default=self.num_workers)
        parser.add_argument("--timesteps", type=int, default=self.timesteps)
        parser.add_argument("--grad_clip", type=float, default=self.grad_clip)
        parser.add_argument("--seed", type=int, default=self.seed)
        parser.add_argument("-g", "--gpu_ids", default="0")

        if self.mode == "train":
            parser.add_argument("--n_iters", type=int, default=self.n_iters)
            parser.add_argument("--lr", type=float, default=self.lr)
            parser.add_argument("--save_frequency", type=int, default=self.save_frequency)
            parser.add_argument("--log_frequency", type=int, default=self.log_frequency)
            parser.add_argument("--continue", dest="cont", action="store_true")

        if self.mode == "classifier":
            parser.add_argument("--classifier_n_iters", type=int, default=self.classifier_n_iters)
            parser.add_argument("--classifier_lr", type=float, default=self.classifier_lr)
            parser.add_argument("--classifier_save_frequency", type=int, default=self.classifier_save_frequency)
            parser.add_argument("--classifier_log_frequency", type=int, default=self.classifier_log_frequency)

        if self.mode == "sample":
            parser.add_argument("--n_samples", type=int, default=1000)
            parser.add_argument("--op_type", choices=["prism", "revol", "thrusections"])
            parser.add_argument("--out", default=None)
            parser.add_argument("--classifier_ckpt", default=self.classifier_ckpt)
            parser.add_argument("--guidance_scale", type=float, default=self.guidance_scale)
            parser.add_argument("--guidance_start_t", type=int, default=self.guidance_start_t)

        if self.mode in {"train", "sample"}:
            parser.add_argument("--ckpt", default="latest")
        parsed = parser.parse_args()
        if self.mode == "sample" and not 0 <= parsed.guidance_start_t < parsed.timesteps:
            parser.error("--guidance_start_t must be within [0, timesteps)")
        return parser, parsed
