"""Evaluate reconstructed command and parameter accuracy."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cadlib.macro import CMD_ARGS_MASK, EOS_IDX, N_ARGS


def load_sequence(path: Path):
    with h5py.File(path, "r") as fp:
        if "commands" not in fp or "args" not in fp:
            raise KeyError(f"{path} has no commands/args")
        commands = fp["commands"][:].astype(np.int64)
        args = fp["args"][:].astype(np.int64)
    if args.shape[1] < N_ARGS:
        args = np.pad(
            args,
            ((0, 0), (0, N_ARGS - args.shape[1])),
            constant_values=-1,
        )
    return commands, args[:, :N_ARGS]


def valid_positions(commands: np.ndarray) -> np.ndarray:
    """Return positions through the first EOS; repeated EOS padding is excluded."""
    end = commands == EOS_IDX
    counts = np.cumsum(end)
    return (counts == 0) | ((counts == 1) & end)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_root", default="recon/test")
    parser.add_argument("--gt_root", default="data/cad_vec")
    parser.add_argument(
        "--output_dir",
        default="runs/evaluation/reconstruction",
    )
    args = parser.parse_args()

    pred_root = Path(args.pred_root)
    gt_root = Path(args.gt_root)
    pred_files = sorted(pred_root.rglob("*.h5"))
    if not pred_files:
        raise SystemExit(f"no prediction files under {pred_root}")

    command_correct = 0
    command_total = 0
    parameter_correct = 0
    parameter_total = 0
    used_files = 0

    for pred_path in pred_files:
        gt_path = gt_root / pred_path.relative_to(pred_root)
        if not gt_path.exists():
            print(f"[SKIP] missing ground truth: {pred_path}")
            continue

        pred_commands, pred_args = load_sequence(pred_path)
        gt_commands, gt_args = load_sequence(gt_path)
        if pred_commands.shape != gt_commands.shape or pred_args.shape != gt_args.shape:
            print(f"[SKIP] shape mismatch: {pred_path}")
            continue

        # Command accuracy includes the first EOS but excludes repeated EOS padding.
        command_mask = valid_positions(gt_commands)
        command_correct += int(
            (pred_commands[command_mask] == gt_commands[command_mask]).sum()
        )
        command_total += int(command_mask.sum())

        # Parameter accuracy uses only command-specific non-padding slots.
        parameter_mask = (
            command_mask[:, None]
            & (CMD_ARGS_MASK[gt_commands] == 1)
            & (gt_args != -1)
        )
        parameter_correct += int(
            (pred_args[parameter_mask] == gt_args[parameter_mask]).sum()
        )
        parameter_total += int(parameter_mask.sum())
        used_files += 1

    if used_files == 0 or command_total == 0 or parameter_total == 0:
        raise RuntimeError("no matched command or parameter positions")

    metrics = {
        "ACC_cmd": command_correct / command_total,
        "ACC_param": parameter_correct / parameter_total,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "metrics.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as fp:
        writer = csv.writer(fp)
        writer.writerow(["ACC_cmd", "ACC_param"])
        writer.writerow([metrics["ACC_cmd"], metrics["ACC_param"]])

    print(f"ACC_cmd: {metrics['ACC_cmd']:.6f}")
    print(f"ACC_param: {metrics['ACC_param']:.6f}")


if __name__ == "__main__":
    main()
