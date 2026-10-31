from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch import optim
from tensorboardX import SummaryWriter


class TrainClock:
    def __init__(self):
        self.epoch = 1
        self.minibatch = 0
        self.step = 0

    def tick(self):
        self.minibatch += 1
        self.step += 1

    def tock(self):
        self.epoch += 1
        self.minibatch = 0

    def state_dict(self):
        return {
            "epoch": self.epoch,
            "minibatch": self.minibatch,
            "step": self.step,
        }

    def load_state_dict(self, state):
        self.epoch = state["epoch"]
        self.minibatch = state["minibatch"]
        self.step = state["step"]


def load_torch_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


class BaseTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.log_dir = cfg.log_dir
        self.model_dir = cfg.model_dir
        self.batch_size = cfg.batch_size
        self.clock = TrainClock()
        self.device = torch.device(cfg.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")

        self.build_net(cfg)
        self.set_loss_function()
        self.set_optimizer(cfg)
        self.train_tb = SummaryWriter(str(Path(self.log_dir) / "train.events"))
        self.val_tb = SummaryWriter(str(Path(self.log_dir) / "val.events"))

    def build_net(self, cfg):
        raise NotImplementedError

    def set_loss_function(self):
        pass

    def set_optimizer(self, cfg):
        self.optimizer = optim.Adam(self.net.parameters(), lr=cfg.lr)

    def checkpoint_state(self):
        net = self.net.module if isinstance(self.net, nn.DataParallel) else self.net
        return net.state_dict()

    def save_ckpt(self, name=None):
        tag = f"ckpt_epoch{self.clock.epoch}" if name is None else name
        path = Path(self.model_dir) / f"{tag}.pth"
        torch.save(
            {
                "clock": self.clock.state_dict(),
                "model_state_dict": self.checkpoint_state(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
            },
            path,
        )
        print(f"saved checkpoint: {path}")

    def load_ckpt(self, name="latest"):
        tag = "latest" if name == "latest" else f"ckpt_epoch{name}"
        path = Path(self.model_dir) / f"{tag}.pth"
        if not path.exists():
            raise FileNotFoundError(path)
        checkpoint = load_torch_checkpoint(path, self.device)
        target = self.net.module if isinstance(self.net, nn.DataParallel) else self.net
        target.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "clock" in checkpoint:
            self.clock.load_state_dict(checkpoint["clock"])
        self.net.to(self.device)
        print(f"loaded checkpoint: {path}")

    def update_network(self, losses):
        loss = sum(losses.values())
        self.optimizer.zero_grad()
        loss.backward()
        if self.cfg.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        return loss

    def update_learning_rate(self):
        self.train_tb.add_scalar(
            "learning_rate",
            self.optimizer.param_groups[0]["lr"],
            self.clock.step,
        )
        self.scheduler.step()

    def record_losses(self, losses, split):
        writer = self.train_tb if split == "train" else self.val_tb
        for name, value in losses.items():
            writer.add_scalar(name, value, self.clock.step)

    def train_func(self, data):
        self.net.train()
        outputs, losses = self.forward(data)
        self.update_network(losses)
        if self.clock.step % 10 == 0:
            self.record_losses(losses, "train")
        return outputs, losses

    def val_func(self, data):
        self.net.eval()
        with torch.no_grad():
            outputs, losses = self.forward(data)
        self.record_losses(losses, "validation")
        return outputs, losses
