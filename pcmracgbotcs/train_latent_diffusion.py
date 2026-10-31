import random

import numpy as np
import torch

from config.config_diffusion import DiffusionConfig
from dataset.latent_dataset import get_dataloader
from trainer.latent_diffusion import LatentDiffusionTrainer


def main():
    cfg = DiffusionConfig("train")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    loader, dataset = get_dataloader(cfg, "train", shuffle=True)
    trainer = LatentDiffusionTrainer(cfg)
    trainer.set_latent_stats(*dataset.get_latent_stats())
    if cfg.cont:
        trainer.load_ckpt(cfg.ckpt, restore_optimizer=True)
    trainer.train(loader)


if __name__ == "__main__":
    main()
