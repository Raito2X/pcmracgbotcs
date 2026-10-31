from __future__ import annotations

import os

import h5py
import numpy as np
import torch
from torch import optim
from tqdm import tqdm

from dataset.latent_dataset import OP_ID_TO_NAME, OP_NAME_TO_ID
from model.diffusion import GaussianDiffusion1D
from model.noise_predictor import LatentNoisePredictor
from model.operation_classifier import OperationPresenceClassifier
from trainer.base import BaseTrainer, load_torch_checkpoint
from utils import cycle


class LatentDiffusionTrainer(BaseTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.result_dir = cfg.result_dir
        os.makedirs(self.result_dir, exist_ok=True)
        self.z_mean = None
        self.z_std = None

    def build_net(self, cfg):
        self.net = GaussianDiffusion1D(
            LatentNoisePredictor(cfg),
            cfg,
        ).to(self.device)

    def set_optimizer(self, cfg):
        self.optimizer = optim.Adam(
            self.net.parameters(),
            lr=cfg.lr,
            betas=(cfg.beta1, 0.9),
        )
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda _: 1.0,
        )

    def set_latent_stats(self, mean, std):
        self.z_mean = torch.as_tensor(mean, device=self.device, dtype=torch.float32)
        self.z_std = torch.as_tensor(std, device=self.device, dtype=torch.float32)
        self.z_std[self.z_std < 1e-6] = 1.0

    def save_ckpt(self, name=None):
        path = os.path.join(
            self.model_dir,
            f"{name or 'ckpt_epoch' + str(self.clock.step)}.pth",
        )
        checkpoint = {
            "clock": self.clock.state_dict(),
            "net_state_dict": self.net.cpu().state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "op_names": [OP_ID_TO_NAME[index] for index in range(3)],
        }
        if self.z_mean is not None:
            checkpoint["z_mean"] = self.z_mean.cpu()
            checkpoint["z_std"] = self.z_std.cpu()
        torch.save(checkpoint, path)
        self.net.to(self.device)
        print(f"saved checkpoint: {path}")

    def load_ckpt(self, name="latest", restore_optimizer=False):
        path = os.path.join(
            self.model_dir,
            f"{name if name == 'latest' else 'ckpt_epoch' + name}.pth",
        )
        checkpoint = load_torch_checkpoint(path, self.device)
        state = checkpoint["net_state_dict"]
        expected = self.net.state_dict()
        compatible = {key: value for key, value in state.items() if key in expected}
        missing = set(expected) - set(compatible)
        if missing:
            raise RuntimeError(f"checkpoint is missing model weights: {sorted(missing)}")
        self.net.load_state_dict(compatible, strict=True)
        if restore_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "clock" in checkpoint:
            self.clock.load_state_dict(checkpoint["clock"])
        if "z_mean" in checkpoint and "z_std" in checkpoint:
            self.z_mean = checkpoint["z_mean"].to(self.device)
            self.z_std = checkpoint["z_std"].to(self.device)
        self.net.to(self.device)
        print(f"loaded checkpoint: {path}")

    def load_operation_classifier(self, path):
        classifier = OperationPresenceClassifier(
            z_dim=self.cfg.z_dim,
            h_dim=self.cfg.h_dim,
            dropout=self.cfg.dropout_rate,
        ).to(self.device)
        checkpoint = load_torch_checkpoint(path, self.device)
        state = checkpoint.get("model_state_dict", checkpoint)
        classifier.load_state_dict(state, strict=True)
        classifier.eval()
        for parameter in classifier.parameters():
            parameter.requires_grad_(False)
        print(f"loaded operation classifier: {path}")
        return classifier

    def train(self, loader):
        self.net.train()
        batches = cycle(loader)
        progress = tqdm(range(self.clock.step, self.cfg.n_iters), desc="latent diffusion")
        for _ in progress:
            latents, _ = next(batches)
            loss = self.net(latents.to(self.device))
            self.optimizer.zero_grad()
            loss.backward()
            if self.cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip)
            self.optimizer.step()
            progress.set_postfix(loss=f"{loss.item():.6f}")

            if self.clock.step % self.cfg.log_frequency == 0:
                self.train_tb.add_scalar("loss", loss.item(), self.clock.step)
            self.clock.tick()
            if self.clock.step % self.cfg.save_frequency == 0:
                self.save_ckpt()
        self.save_ckpt("latest")

    def generate(self, n_samples, op_type=None, classifier=None, guidance_scale=0.0):
        self.net.eval()
        generated = []
        target_id = None if op_type is None else OP_NAME_TO_ID[op_type]
        remaining = n_samples
        while remaining:
            batch_size = min(self.batch_size, remaining)
            latents = self.net.sample(
                batch_size,
                classifier=classifier,
                target=target_id,
                guidance_scale=guidance_scale,
            )
            if self.z_mean is None or self.z_std is None:
                raise RuntimeError("latent normalization statistics are not loaded")
            latents = latents.squeeze(1) * self.z_std + self.z_mean
            generated.append(latents.cpu().numpy())
            remaining -= batch_size
        return np.concatenate(generated, axis=0)

    def save_samples(
        self,
        n_samples,
        save_path,
        op_type=None,
        classifier=None,
        guidance_scale=0.0,
    ):
        latents = self.generate(n_samples, op_type, classifier, guidance_scale)
        if save_path is None:
            label = op_type or "unconditioned"
            save_path = os.path.join(
                self.result_dir,
                f"{label}_scale{guidance_scale}_n{n_samples}.h5",
            )
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with h5py.File(save_path, "w") as fp:
            fp.create_dataset("zs", data=latents.astype(np.float32))
            fp.attrs["op_type_mapping"] = "0:prism,1:revol,2:thrusections"
            fp.attrs["guidance_scale"] = float(guidance_scale)
            fp.attrs["guidance_start_t"] = int(self.cfg.guidance_start_t)
            if op_type is not None:
                fp.attrs["target_op_type"] = OP_NAME_TO_ID[op_type]
                fp.attrs["target_op_name"] = op_type
        print(f"saved samples: {save_path}")
        return save_path
