"""
Generate histograms of scene graph sizes (nodes and relations) across a dataset.

Usage:
    python sg_size_histogram.py [--source <dataset_dir>] [--output <output_path>]
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from consts import DATA_DIR

SPLITS = ("train", "val", "test")


def collect_sizes(source: Path) -> tuple[list[int], list[int]]:
    """Collect node and edge counts from all scene graph .pth files under source."""
    node_counts: list[int] = []
    edge_counts: list[int] = []

    # Discover all scene_graph directories: handle both layout styles:
    #   Case A: source/<split>/scene_graphs/
    #   Case B: source/<encoder>/<split>/scene_graphs/
    sg_dirs: list[Path] = []

    for split in SPLITS:
        direct = source / split / "scene_graphs"
        if direct.is_dir():
            sg_dirs.append(direct)

    if not sg_dirs:
        for encoder_dir in sorted(source.iterdir()):
            if not encoder_dir.is_dir():
                continue
            for split in SPLITS:
                sg_dir = encoder_dir / split / "scene_graphs"
                if sg_dir.is_dir():
                    sg_dirs.append(sg_dir)

    # De-duplicate (same scene graph file can appear in multiple encoder dirs)
    seen_stems: set[str] = set()
    for sg_dir in sg_dirs:
        for path in sorted(sg_dir.glob("*.pth")):
            if path.stem in seen_stems:
                continue
            seen_stems.add(path.stem)
            sg = torch.load(path, weights_only=False)
            node_counts.append(sg.x.shape[0])
            edge_counts.append(sg.edge_index.shape[1])

    return node_counts, edge_counts


def plot_histograms(
    node_counts: list[int],
    edge_counts: list[int],
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(node_counts, bins=30, color="steelblue", edgecolor="white")
    axes[0].set_title("Scene Graph Size: Nodes")
    axes[0].set_xlabel("Number of Nodes")
    axes[0].set_ylabel("Count")

    axes[1].hist(edge_counts, bins=30, color="darkorange", edgecolor="white")
    axes[1].set_title("Scene Graph Size: Relations (Edges)")
    axes[1].set_xlabel("Number of Relations")
    axes[1].set_ylabel("Count")

    fig.suptitle(f"Scene Graph Size Distribution (n={len(node_counts)} graphs)")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Histogram saved to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Histogram of scene graph sizes (nodes and relations)."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DATA_DIR / "dataset",
        help="Dataset root directory (default: DATA_DIR/dataset)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sg_size_histogram.png"),
        help="Output image path (default: sg_size_histogram.png)",
    )
    args = parser.parse_args()

    print(f"Scanning {args.source} ...")
    node_counts, edge_counts = collect_sizes(args.source)

    if not node_counts:
        print("No scene graph files found.")
        return

    print(f"Found {len(node_counts)} unique scene graphs.")
    plot_histograms(node_counts, edge_counts, args.output)


if __name__ == "__main__":
    main()
