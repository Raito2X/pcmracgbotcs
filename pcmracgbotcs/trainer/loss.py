import torch
import torch.nn as nn
import torch.nn.functional as F

from cadlib.macro import CMD_ARGS_MASK, EOS_IDX


class CADLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_commands = cfg.n_commands
        self.args_dim = cfg.args_dim + 1
        self.weights = cfg.loss_weights
        self.register_buffer(
            "cmd_args_mask",
            torch.tensor(CMD_ARGS_MASK, dtype=torch.bool),
        )

    @staticmethod
    def valid_positions(commands):
        """Keep commands through the first EOS; repeated EOS padding is excluded."""
        end = commands == EOS_IDX
        counts = end.cumsum(dim=1)
        return (counts == 0) | ((counts == 1) & end)

    def forward(self, output):
        commands = output["tgt_commands"]
        args = output["tgt_args"]
        command_logits = output["command_logits"]
        args_logits = output["args_logits"]
        valid_positions = self.valid_positions(commands)

        loss_command = F.cross_entropy(
            command_logits[valid_positions],
            commands[valid_positions].long(),
        )

        # Parameter loss uses command-specific slots and excludes -1 padding.
        valid_args = self.cmd_args_mask[commands.long()]
        valid_args &= valid_positions.unsqueeze(-1)
        valid_args &= args != -1
        loss_args = F.cross_entropy(
            args_logits[valid_args],
            args[valid_args].long() + 1,
        )

        return {
            "loss_cmd": self.weights["loss_cmd_weight"] * loss_command,
            "loss_args": self.weights["loss_args_weight"] * loss_args,
        }
