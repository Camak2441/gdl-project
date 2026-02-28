"""
Print statistics on scene graph sizes (nodes and relations) in a dataset.

Reports max, min, mean, median, and std for both node and edge counts,
broken down by split and in aggregate.

Usage:
    python sg_size_stats.py [--source <dataset_dir>]
"""

import argparse
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from consts import DATA_DIR

SPLITS = ("train", "val", "test")


def collect_sizes_for_dir(sg_dir: Path) -> tuple[list[int], list[int]]:
    node_counts: list[int] = []
    edge_counts: list[int] = []
    for path in sorted(sg_dir.glob("*.pth")):
        sg = torch.load(path, weights_only=False)
        node_counts.append(sg.x.shape[0])
        edge_counts.append(sg.edge_index.shape[1])
    return node_counts, edge_counts


def print_stats(label: str, counts: list[int], unit: str) -> None:
    if not counts:
        print(f"  {label} ({unit}): no data")
        return
    print(
        f"  {label} ({unit}): "
        f"min={min(counts)}, "
        f"max={max(counts)}, "
        f"mean={statistics.mean(counts):.1f}, "
        f"median={statistics.median(counts):.1f}, "
        f"std={statistics.stdev(counts):.1f}"
        if len(counts) > 1
        else f"  {label} ({unit}): "
        f"min={min(counts)}, "
        f"max={max(counts)}, "
        f"mean={statistics.mean(counts):.1f}, "
        f"median={statistics.median(counts):.1f}"
    )


def process_source(source: Path) -> None:
    # Detect layout:
    #   Case A: source/<split>/scene_graphs/
    #   Case B: source/<encoder>/<split>/scene_graphs/
    split_dirs: dict[str, Path] = {}
    for split in SPLITS:
        candidate = source / split / "scene_graphs"
        if candidate.is_dir():
            split_dirs[split] = candidate

    if not split_dirs:
        for encoder_dir in sorted(source.iterdir()):
            if not encoder_dir.is_dir():
                continue
            for split in SPLITS:
                candidate = encoder_dir / split / "scene_graphs"
                if candidate.is_dir() and split not in split_dirs:
                    split_dirs[split] = candidate

    if not split_dirs:
        print("No scene_graphs directories found under the source.")
        return

    all_nodes: list[int] = []
    all_edges: list[int] = []
    seen_stems: set[str] = set()

    for split, sg_dir in sorted(split_dirs.items()):
        nodes, edges = collect_sizes_for_dir(sg_dir)
        print(f"\n[{split}] ({len(nodes)} scene graphs)")
        print_stats("nodes", nodes, "nodes")
        print_stats("edges", edges, "relations")

        for path in sorted(sg_dir.glob("*.pth")):
            if path.stem not in seen_stems:
                seen_stems.add(path.stem)
                sg = torch.load(path, weights_only=False)
                all_nodes.append(sg.x.shape[0])
                all_edges.append(sg.edge_index.shape[1])

    print(f"\n[overall] ({len(all_nodes)} unique scene graphs)")
    print_stats("nodes", all_nodes, "nodes")
    print_stats("edges", all_edges, "relations")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scene graph size statistics (nodes and relations)."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DATA_DIR / "dataset",
        help="Dataset root directory (default: DATA_DIR/dataset)",
    )
    args = parser.parse_args()

    print(f"Source: {args.source}")
    process_source(args.source)
    print()


if __name__ == "__main__":
    main()
