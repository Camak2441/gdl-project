"""
Produce a balanced dataset with an equal number of each question type per split,
maximising the total size (i.e. sample min-count-across-types from each type).

The source dataset is expected to have the structure:
    <source>/train/questions/*.pth
    <source>/train/scene_graphs/*.pth
    <source>/test/...
    <source>/val/...

Each questions file contains:
    query   - tensor [N, D]
    y       - tensor [N, num_nodes]
    qtype   - tensor [N]  (integer codes from Q_TYPES)
    scanId  - str

Usage:
    python balance_dataset.py --source <src_dir> --dest <dst_dir>
"""

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

import torch

from consts import DATA_DIR, Q_TYPES

SPLITS = ("train", "test", "val")
QTYPE_NAMES = {v: k for k, v in Q_TYPES.items()}


def balance_split(split_dir: Path, output_dir: Path, seed: int):
    """Sample an equal number of each question type, maximising total size."""
    questions_in = split_dir / "questions"
    scene_graphs_in = split_dir / "scene_graphs"

    if not questions_in.exists():
        print(f"  Skipping {split_dir.name}: no questions directory found.")
        return

    # Index all questions by type
    # type_index[qtype_int] = list of (scan_id, local_idx)
    type_index: dict[int, list[tuple[str, int]]] = defaultdict(list)
    scan_data: dict[str, dict] = {}

    for path in sorted(questions_in.glob("*.pth")):
        q = torch.load(path, weights_only=False)
        scan_id = q["scanId"]
        scan_data[scan_id] = q
        for local_idx, qtype_val in enumerate(q["qtype"].tolist()):
            type_index[int(qtype_val)].append((scan_id, local_idx))

    if not type_index:
        print(f"  Skipping {split_dir.name}: no questions found.")
        return

    # The maximum balanced count is constrained by the rarest type
    per_type = min(len(candidates) for candidates in type_index.values())
    print(
        f"  [{split_dir.name}] Balancing to {per_type} questions per type "
        f"({per_type * len(type_index)} total across {len(type_index)} types)."
    )

    rng = random.Random(seed)

    # Sample per_type from each type
    selected: dict[int, list[tuple[str, int]]] = {
        qtype_int: rng.sample(candidates, per_type)
        for qtype_int, candidates in type_index.items()
    }

    # Group selected questions by scan_id
    scan_selected: dict[str, list[int]] = defaultdict(list)
    for indices in selected.values():
        for scan_id, local_idx in indices:
            scan_selected[scan_id].append(local_idx)

    # Write output
    questions_out = output_dir / "questions"
    scene_graphs_out = output_dir / "scene_graphs"
    questions_out.mkdir(parents=True, exist_ok=True)
    scene_graphs_out.mkdir(parents=True, exist_ok=True)

    total_written = 0
    for scan_id, local_indices in scan_selected.items():
        local_indices_t = sorted(set(local_indices))
        idx_tensor = torch.tensor(local_indices_t, dtype=torch.long)
        q = scan_data[scan_id]

        filtered = {
            "query": q["query"][idx_tensor],
            "y": q["y"][idx_tensor],
            "qtype": q["qtype"][idx_tensor],
            "scanId": scan_id,
            **(
                {"question": [q["question"][i] for i in local_indices_t]}
                if "question" in q
                else {}
            ),
        }
        torch.save(filtered, questions_out / f"{scan_id}.pth")
        total_written += len(local_indices_t)

        src_sg = scene_graphs_in / f"{scan_id}.pth"
        dst_sg = scene_graphs_out / f"{scan_id}.pth"
        if src_sg.exists() and not dst_sg.exists():
            shutil.copy2(src_sg, dst_sg)

    print(
        f"  [{split_dir.name}] Wrote {total_written} questions across "
        f"{len(scan_selected)} scenes."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Balance a dataset by question type, maximising total size."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DATA_DIR / "dataset",
        help="Source dataset root (default: DATA_DIR/dataset)",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DATA_DIR / "dataset_balanced",
        help="Output dataset root (default: DATA_DIR/dataset_balanced)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default: 0)",
    )
    args = parser.parse_args()

    # Handle both layouts:
    # Case A: source/<split>/  (splits directly under source)
    # Case B: source/<encoder>/<split>/  (encoder sub-dirs under source)
    source_splits = [args.source / s for s in SPLITS]
    if any(p.exists() for p in source_splits):
        encoder_dirs = [("", args.source)]
    else:
        encoder_dirs = [
            (d.name, d) for d in sorted(args.source.iterdir()) if d.is_dir()
        ]

    for encoder_name, encoder_src in encoder_dirs:
        encoder_dst = args.dest / encoder_name if encoder_name else args.dest
        label = f"encoder: {encoder_name}" if encoder_name else "dataset"
        print(f"\nProcessing {label}")

        for split in SPLITS:
            balance_split(encoder_src / split, encoder_dst / split, args.seed)

    print("\nDone.")


if __name__ == "__main__":
    main()
