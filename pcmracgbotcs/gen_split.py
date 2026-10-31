import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cad_vec_dir", default="data/cad_vec")
    parser.add_argument("--out", default="data/split.json")
    parser.add_argument("--train", type=float, default=0.9)
    parser.add_argument("--val", type=float, default=0.05)
    parser.add_argument("--test", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if abs(args.train + args.val + args.test - 1.0) > 1e-8:
        raise ValueError("split ratios must sum to 1")

    root = Path(args.cad_vec_dir)
    files = sorted(root.rglob("*.h5"))
    if not files:
        raise RuntimeError(f"no HDF5 files under {root}")
    ids = [
        str(path.relative_to(root).with_suffix("")).replace("\\", "/")
        for path in files
    ]
    random.Random(args.seed).shuffle(ids)

    train_count = int(len(ids) * args.train)
    validation_count = int(len(ids) * args.val)
    split = {
        "train": ids[:train_count],
        "val": ids[train_count:train_count + validation_count],
        "test": ids[train_count + validation_count:],
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(split, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print({name: len(items) for name, items in split.items()})


if __name__ == "__main__":
    main()
