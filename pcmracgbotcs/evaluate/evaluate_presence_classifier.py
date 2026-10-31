"""Evaluate operation-presence predictions across a dataset split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from dataset.latent_dataset import (
    OP_BIT_NAMES,
    CADLatentDataset,
)
from model.diffusion import cosine_beta_schedule
from model.operation_classifier import OperationPresenceClassifier


BIT_LABELS = ("Prism", "Revol", "Thrusections")


def binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    rank_sum = float(ranks[labels == 1].sum())
    return (
        rank_sum - len(positive) * (len(positive) + 1) / 2.0
    ) / (len(positive) * len(negative))


def binary_metrics(probabilities: np.ndarray, labels: np.ndarray, threshold: float):
    predictions = (probabilities >= threshold).astype(np.int64)
    tp = int(((predictions == 1) & (labels == 1)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {
        "counts": [[tn, fp], [fn, tp]],
        "support_negative": int(tn + fp),
        "support_positive": int(fn + tp),
        "recall": float(recall),
        "precision": float(precision),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "auc": binary_auc(probabilities, labels),
    }


def load_checkpoint(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_dataset(args):
    cfg = SimpleNamespace(
        z_path=args.z_path,
        cad_data_root=args.cad_data_root,
        split_path=args.split_path,
        normalize_latent=True,
        batch_size=256,
        num_workers=0,
    )
    return CADLatentDataset(cfg, phase=args.phase)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--z_path", required=True, help="HDF5 file from encode_latents.py")
    parser.add_argument("--classifier_ckpt", required=True)
    parser.add_argument("--cad_data_root", default="data")
    parser.add_argument("--split_path", default="data/split.json")
    parser.add_argument("--phase", default="validation", choices=["train", "validation", "val", "test"])
    parser.add_argument("--output_dir", default="runs/evaluation/presence_classifier")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset = build_dataset(args)
    normalized_latents = (
        dataset.zs.astype(np.float32) - dataset.z_mean[None, :]
    ) / dataset.z_std[None, :]
    latents = torch.from_numpy(np.ascontiguousarray(normalized_latents))
    labels = dataset.presence.astype(np.int64)

    checkpoint_path = Path(args.classifier_ckpt).resolve()
    checkpoint = load_checkpoint(checkpoint_path, device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Expected a classifier checkpoint with key 'model_state_dict'.")
    classifier = OperationPresenceClassifier(
        z_dim=checkpoint.get("z_dim", 256),
        h_dim=checkpoint.get("h_dim", 512),
        dropout=0.0,
    ).to(device)
    classifier.load_state_dict(checkpoint["model_state_dict"], strict=True)
    classifier.eval()

    betas = cosine_beta_schedule(args.timesteps).float()
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    generator = torch.Generator().manual_seed(args.seed)
    random_t = torch.randint(
        0,
        args.timesteps,
        (latents.shape[0],),
        generator=generator,
    )
    noise_generator = torch.Generator().manual_seed(args.seed + 1)
    noise = torch.randn(latents.shape, generator=noise_generator)
    alpha_t = alphas_cumprod[random_t].unsqueeze(1)
    noisy_z = torch.sqrt(alpha_t) * latents + torch.sqrt(1.0 - alpha_t) * noise

    with torch.no_grad():
        logits = classifier(noisy_z.to(device), random_t.to(device))
        probabilities = torch.sigmoid(logits).cpu().numpy()

    metrics = {}
    for index, (bit_name, display_name) in enumerate(zip(OP_BIT_NAMES, BIT_LABELS)):
        metrics[bit_name] = {
            "display_name": display_name,
            **binary_metrics(
                probabilities[:, index],
                labels[:, index],
                threshold=args.threshold,
            ),
        }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": checkpoint.get("step"),
        "phase": args.phase,
        "samples": int(len(labels)),
        "seed": args.seed,
        "timesteps": args.timesteps,
        "threshold": args.threshold,
        "metrics": metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            ["operation", "precision", "recall", "f1", "accuracy", "auc", "positive_support"]
        )
        for bit_name, display_name in zip(OP_BIT_NAMES, BIT_LABELS):
            result = metrics[bit_name]
            writer.writerow(
                [
                    display_name,
                    result["precision"],
                    result["recall"],
                    result["f1"],
                    result["accuracy"],
                    result["auc"],
                    result["support_positive"],
                ]
            )
    with (output_dir / "confusion_counts.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as fp:
        writer = csv.writer(fp)
        writer.writerow(["operation", "tn", "fp", "fn", "tp"])
        for bit_name, display_name in zip(OP_BIT_NAMES, BIT_LABELS):
            tn, fp = metrics[bit_name]["counts"][0]
            fn, tp = metrics[bit_name]["counts"][1]
            writer.writerow([display_name, tn, fp, fn, tp])

    for bit_name, display_name in zip(OP_BIT_NAMES, BIT_LABELS):
        result = metrics[bit_name]
        print(
            f"{display_name:12s} precision={result['precision']:.4f} "
            f"recall={result['recall']:.4f} f1={result['f1']:.4f} "
            f"auc={result['auc']:.4f}"
        )
    print(f"Saved report to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
