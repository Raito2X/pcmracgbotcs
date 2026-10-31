import os
import random

import numpy as np
import torch

from config.config_diffusion import DiffusionConfig
from trainer.latent_diffusion import LatentDiffusionTrainer


def main():
    cfg = DiffusionConfig("sample")
    if cfg.op_type is None:
        raise SystemExit("--op_type is required")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    trainer = LatentDiffusionTrainer(cfg)
    trainer.load_ckpt(cfg.ckpt)
    classifier_path = cfg.classifier_ckpt or os.path.join(
        cfg.classifier_model_dir,
        "op_classifier_latest.pth",
    )
    if not os.path.exists(classifier_path):
        raise FileNotFoundError(classifier_path)
    classifier = trainer.load_operation_classifier(classifier_path)
    trainer.save_samples(
        n_samples=cfg.n_samples,
        save_path=cfg.out,
        op_type=cfg.op_type,
        classifier=classifier,
        guidance_scale=cfg.guidance_scale,
    )


if __name__ == "__main__":
    main()
