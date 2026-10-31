import torch
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from cadlib.macro import (
    CMD_ARGS_MASK,
    EOS_IDX,
)
from model.autoencoder import CADTransformer
from trainer.base import BaseTrainer
from trainer.loss import CADLoss

class AutoencoderTrainer(BaseTrainer):
    def build_net(self, cfg):
        self.net = CADTransformer(cfg).to(self.device)

    def set_loss_function(self):
        self.loss_func = CADLoss(self.cfg).to(self.device)

    def set_optimizer(self, cfg):
        self.optimizer = optim.Adam(self.net.parameters(), lr=cfg.lr)
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(1.0, (step + 1) / cfg.warmup_step),
        )

    def forward(self, data):
        commands = data["command"].to(self.device)
        args = data["args"].to(self.device)
        output = self.net(commands, args)
        return output, self.loss_func(output)

    def evaluate(self, loader):
        self.net.eval()
        command_correct = 0
        command_total = 0
        parameter_correct = 0
        parameter_total = 0

        for data in tqdm(loader, desc=f"evaluate epoch {self.clock.epoch}"):
            commands = data["command"].to(self.device)
            args = data["args"].to(self.device)
            with torch.no_grad():
                output = self.net(commands, args)
            predicted = output["command_logits"].argmax(dim=-1)
            predicted_args = output["args_logits"].argmax(dim=-1) - 1

            end = commands == EOS_IDX
            counts = end.cumsum(dim=1)
            valid = (counts == 0) | ((counts == 1) & end)
            command_correct += int((predicted[valid] == commands[valid]).sum())
            command_total += int(valid.sum())

            parameter_mask = (
                valid[:, :, None]
                & (torch.tensor(
                    CMD_ARGS_MASK,
                    dtype=torch.bool,
                    device=self.device,
                )[commands] == 1)
                & (args != -1)
            )
            parameter_correct += int(
                (predicted_args[parameter_mask] == args[parameter_mask]).sum()
            )
            parameter_total += int(parameter_mask.sum())

        metrics = {
            "ACC_cmd": command_correct / max(command_total, 1),
            "ACC_param": parameter_correct / max(parameter_total, 1),
        }
        self.val_tb.add_scalars("metrics", metrics, self.clock.epoch)
        print(metrics)
        self.net.train()
