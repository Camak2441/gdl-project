"""
Print the question type distribution for each split in a dataset directory.

Usage:
    python dataset_type_stats.py [--source <dataset_dir>]
"""

import argparse
from collections import Counter
from pathlib import Path

import torch

from consts import DATA_DIR, Q_TYPES

SPLITS = ("train", "val", "test")
QTYPE_NAMES = {v: k for k, v in Q_TYPES.items()}


def split_stats(split_dir: Path) -> Counter:
    questions_dir = split_dir / "questions"
    if not questions_dir.exists():
        return Counter()

    counts: Counter = Counter()
    for path in questions_dir.glob("*.pth"):
        q = torch.load(path, weights_only=False)
        for qtype_val in q["qtype"].tolist():
            counts[int(qtype_val)] += 1
    return counts


def print_stats(split_name: str, counts: Counter):
    total = sum(counts.values())
    if total == 0:
        print(f"{split_name}: no questions found.\n")
        return

    print(f"{split_name} ({total} total):")
    print(f"  {'Type':<12} {'Count':>8}  {'Proportion':>12}")
    print(f"  {'-' * 34}")
    for qtype_int, count in sorted(counts.items(), key=lambda x: -x[1]):
        name = QTYPE_NAMES.get(qtype_int, str(qtype_int))
        print(f"  {name:<12} {count:>8}  {count / total:>11.1%}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Show question type distribution per split."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DATA_DIR / "dataset",
        help="Dataset root directory (default: DATA_DIR/dataset)",
    )
    args = parser.parse_args()

    # Handle both layouts:
    # Case A: source/<split>/       (splits directly under source)
    # Case B: source/<encoder>/<split>/
    source_splits = [args.source / s for s in SPLITS]
    if any(p.exists() for p in source_splits):
        encoder_dirs = [("", args.source)]
    else:
        encoder_dirs = [
            (d.name, d) for d in sorted(args.source.iterdir()) if d.is_dir()
        ]

    for encoder_name, encoder_src in encoder_dirs:
        if encoder_name:
            print(f"=== Encoder: {encoder_name} ===\n")
        for split in SPLITS:
            counts = split_stats(encoder_src / split)
            print_stats(split, counts)


if __name__ == "__main__":
    main()
