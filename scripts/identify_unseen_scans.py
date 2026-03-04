"""
Identify which scene graphs in test/val splits are unseen (not in the training set).

A scene graph is "unseen" if its scan_id does not appear in the training split.
Scenes that appear in both train and test/val are "seen" — only some questions
from those scenes were held out to test/val.

Usage:
    python scripts/identify_unseen_scans.py
    python scripts/identify_unseen_scans.py --dataset_dir path/to/dataset
    python scripts/identify_unseen_scans.py --output unseen_scans.json
"""

import argparse
import json
from pathlib import Path

from consts import DATA_DIR

EDGE_ENCODER = "all_minilm_l6v2"
NODE_ENCODER = "all_minilm_l6v2"
QUERY_ENCODER = "all_minilm_l6v2"

DEFAULT_DATASET_DIR = (
    DATA_DIR
    / "dataset_balanced"
    / ";".join([EDGE_ENCODER, NODE_ENCODER, QUERY_ENCODER])
)


def get_scan_ids(questions_dir: Path) -> set[str]:
    if not questions_dir.exists():
        return set()
    return {p.stem for p in questions_dir.glob("*.pth")}


def identify_unseen(dataset_dir: Path) -> dict:
    """Return a dict describing seen/unseen scan_ids for each non-train split."""
    train_scan_ids = get_scan_ids(dataset_dir / "train" / "questions")

    result = {"dataset_dir": str(dataset_dir), "train_scans": sorted(train_scan_ids)}

    splits = [p.name for p in dataset_dir.iterdir() if p.is_dir() and p.name != "train"]
    for split in sorted(splits):
        split_scan_ids = get_scan_ids(dataset_dir / split / "questions")
        seen = sorted(split_scan_ids & train_scan_ids)
        unseen = sorted(split_scan_ids - train_scan_ids)
        result[split] = {"seen": seen, "unseen": unseen}

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Identify unseen scene graphs in test/val splits."
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Root of the dataset (must contain train/, test/, val/ subdirs).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save results as JSON.",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    if not dataset_dir.exists():
        print(f"Dataset directory not found: {dataset_dir}")
        return

    info = identify_unseen(dataset_dir)

    print(f"Dataset: {dataset_dir}")
    print(f"\nTraining scenes: {len(info['train_scans'])}")

    for split, data in info.items():
        if split in ("dataset_dir", "train_scans"):
            continue
        total = len(data["seen"]) + len(data["unseen"])
        print(f"\n{split} split:")
        print(f"  Total scenes:   {total}")
        print(f"  Seen in train:  {len(data['seen'])}")
        print(f"  Unseen (new):   {len(data['unseen'])}")
        if data["unseen"]:
            print(f"  Unseen scan IDs:")
            for sid in data["unseen"]:
                print(f"    {sid}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(info, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
