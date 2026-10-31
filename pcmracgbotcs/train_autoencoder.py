import random

import numpy as np
import torch
from tqdm import tqdm

from config.config_ae import AutoencoderConfig
from dataset.cad_dataset import get_dataloader
from trainer.autoencoder import AutoencoderTrainer
from utils import cycle


def main():
    cfg = AutoencoderConfig()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    trainer = AutoencoderTrainer(cfg)
    if cfg.cont:
        trainer.load_ckpt(cfg.ckpt)

    train_loader = get_dataloader("train", cfg)
    validation_loader = get_dataloader("validation", cfg)
    validation_iterator = cycle(validation_loader)

    for epoch in range(trainer.clock.epoch, cfg.nr_epochs + 1):
        progress = tqdm(train_loader, desc=f"epoch {epoch}")
        for data in progress:
            _, losses = trainer.train_func(data)
            progress.set_postfix({name: f"{value.item():.4f}" for name, value in losses.items()})
            if trainer.clock.step % cfg.val_frequency == 0:
                trainer.val_func(next(validation_iterator))
            trainer.clock.tick()
            trainer.update_learning_rate()

        if trainer.clock.epoch % 5 == 0:
            trainer.evaluate(validation_loader)
        if trainer.clock.epoch % cfg.save_frequency == 0:
            trainer.save_ckpt()
        trainer.save_ckpt("latest")
        trainer.clock.tock()


if __name__ == "__main__":
    main()
