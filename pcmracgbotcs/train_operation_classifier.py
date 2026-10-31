import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config.config_diffusion import DiffusionConfig
from dataset.latent_dataset import OP_BIT_NAMES, get_dataloader
from model.operation_classifier import OperationPresenceClassifier
from trainer.latent_diffusion import LatentDiffusionTrainer


def evaluate(classifier, diffusion, loader, device):
    classifier.eval()
    total_loss = 0.0
    total_values = 0
    tp = np.zeros(3, dtype=np.int64)
    fp = np.zeros(3, dtype=np.int64)
    fn = np.zeros(3, dtype=np.int64)
    tn = np.zeros(3, dtype=np.int64)

    with torch.no_grad():
        for latents, labels in loader:
            latents = latents.to(device)
            labels = labels.float().to(device)
            timesteps = torch.randint(
                0,
                diffusion.num_timesteps,
                (latents.shape[0],),
                device=device,
            )
            noisy = diffusion.q_sample(latents[:, None], timesteps)
            logits = classifier(noisy, timesteps)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            total_loss += loss.item() * labels.numel()
            total_values += labels.numel()

            predictions = (torch.sigmoid(logits) >= 0.5).cpu().numpy()
            expected = labels.cpu().numpy().astype(np.int64)
            tp += ((predictions == 1) & (expected == 1)).sum(axis=0)
            fp += ((predictions == 1) & (expected == 0)).sum(axis=0)
            fn += ((predictions == 0) & (expected == 1)).sum(axis=0)
            tn += ((predictions == 0) & (expected == 0)).sum(axis=0)

    return {
        "loss": total_loss / total_values,
        "accuracy": float((tp.sum() + tn.sum()) / total_values),
        "recall": tp / np.maximum(tp + fn, 1),
        "precision": tp / np.maximum(tp + fp, 1),
    }


def main():
    cfg = DiffusionConfig("classifier")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    train_loader, _ = get_dataloader(cfg, "train", shuffle=True)
    validation_loader, _ = get_dataloader(cfg, "validation", shuffle=False)
    trainer = LatentDiffusionTrainer(cfg)
    diffusion = trainer.net
    diffusion.eval()
    for parameter in diffusion.parameters():
        parameter.requires_grad_(False)

    classifier = OperationPresenceClassifier(
        cfg.z_dim,
        cfg.h_dim,
        cfg.dropout_rate,
    ).to(trainer.device)
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=cfg.classifier_lr,
        weight_decay=1e-4,
    )

    batches = iter(train_loader)
    best_accuracy = -1.0
    progress = tqdm(range(cfg.classifier_n_iters), desc="operation classifier")
    for step in progress:
        classifier.train()
        try:
            latents, labels = next(batches)
        except StopIteration:
            batches = iter(train_loader)
            latents, labels = next(batches)
        latents = latents.to(trainer.device)
        labels = labels.float().to(trainer.device)
        timesteps = torch.randint(
            0,
            diffusion.num_timesteps,
            (latents.shape[0],),
            device=trainer.device,
        )

        # Use the diffusion noise schedule to train on matching noisy latents.
        noisy = diffusion.q_sample(latents[:, None], timesteps)
        logits = classifier(noisy, timesteps)
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), cfg.grad_clip)
        optimizer.step()
        if step % cfg.classifier_log_frequency == 0:
            progress.set_postfix(loss=f"{loss.item():.6f}")

        if step and step % cfg.classifier_save_frequency == 0:
            metrics = evaluate(classifier, diffusion, validation_loader, trainer.device)
            print(f"step {step}: {metrics}")
            checkpoint = {
                "model_state_dict": classifier.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": step,
                "z_dim": cfg.z_dim,
                "h_dim": cfg.h_dim,
                "bit_names": list(OP_BIT_NAMES),
            }
            torch.save(
                checkpoint,
                cfg.classifier_model_dir + f"/op_classifier_step{step}.pth",
            )
            if metrics["accuracy"] > best_accuracy:
                best_accuracy = metrics["accuracy"]
                torch.save(
                    checkpoint,
                    cfg.classifier_model_dir + "/op_classifier_latest.pth",
                )

    final_metrics = evaluate(classifier, diffusion, validation_loader, trainer.device)
    final_checkpoint = {
        "model_state_dict": classifier.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": cfg.classifier_n_iters,
        "z_dim": cfg.z_dim,
        "h_dim": cfg.h_dim,
        "bit_names": list(OP_BIT_NAMES),
    }
    torch.save(final_checkpoint, cfg.classifier_model_dir + "/op_classifier_final.pth")
    torch.save(
        final_checkpoint,
        cfg.classifier_model_dir
        + f"/op_classifier_step{cfg.classifier_n_iters}.pth",
    )
    if final_metrics["accuracy"] > best_accuracy:
        torch.save(final_checkpoint, cfg.classifier_model_dir + "/op_classifier_latest.pth")
    print(f"final metrics: {final_metrics}")


if __name__ == "__main__":
    main()
