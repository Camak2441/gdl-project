"""
Calculate metrics from a saved model output file produced by eval_model.py.

Usage:
    python scripts/calc_model_metrics.py <model_name>
    python scripts/calc_model_metrics.py <model_name> --split val
"""

import argparse
import json
import math
from pathlib import Path

import torch

from consts import OUTPUT_DIR, Q_TYPES, DATA_DIR
from models import get_model_name, get_most_epochs_file

THRESHOLD = 0.5

Q_TYPE_NAMES = {v: k for k, v in Q_TYPES.items()}


EDGE_ENCODER = "all_minilm_l6v2"
NODE_ENCODER = "all_minilm_l6v2"
QUERY_ENCODER = "all_minilm_l6v2"


def precision_recall_accuracy_f1(out: torch.Tensor, y: torch.Tensor):
    pred = out >= THRESHOLD
    tp = (pred & y.bool()).sum().item()
    fp = (pred & ~y.bool()).sum().item()
    fn = (~pred & y.bool()).sum().item()
    tn = (~pred & ~y.bool()).sum().item()
    total = len(y)
    precision = tp / (tp + fp) if (tp + fp) > 0 else math.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else math.nan
    accuracy = (tp + tn) / total if total > 0 else math.nan
    f1 = (
        2 * precision * recall / (precision + recall)
        if not math.isnan(precision)
        and not math.isnan(recall)
        and precision + recall > 0
        else math.nan
    )
    return (precision, recall, accuracy, f1)


def top_n_recall(out: torch.Tensor, y: torch.Tensor, num_nodes: torch.Tensor):
    """For each query, take the top-n predictions where n = number of positive nodes."""
    recalls = []
    node_offset = 0
    for n_nodes in num_nodes.tolist():
        out_q = out[node_offset : node_offset + n_nodes]
        y_q = y[node_offset : node_offset + n_nodes]
        n_positive = int(y_q.sum().item())
        if n_positive > 0:
            top_n_idx = out_q.topk(min(n_positive, n_nodes)).indices
            recalls.append(y_q[top_n_idx].sum().item() / n_positive)
        node_offset += n_nodes
    return sum(recalls) / len(recalls) if recalls else 0.0


def per_q_precision_recall_accuracy_f1(
    out: torch.Tensor, y: torch.Tensor, num_nodes: torch.Tensor
):
    """For each query, take the top-n predictions where n = number of positive nodes."""
    recalls = []
    precisions = []
    accuracies = []
    f1s = []
    node_offset = 0
    for n_nodes in num_nodes.tolist():
        pred_q = out[node_offset : node_offset + n_nodes] >= THRESHOLD
        y_q = y[node_offset : node_offset + n_nodes]
        tp = (pred_q & y_q.bool()).sum().item()
        fp = (pred_q & ~y_q.bool()).sum().item()
        fn = (~pred_q & y_q.bool()).sum().item()
        tn = (~pred_q & ~y_q.bool()).sum().item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        accuracy = (tp + tn) / n_nodes if n_nodes > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0
        )
        recalls.append(recall)
        precisions.append(precision)
        accuracies.append(accuracy)
        f1s.append(f1)
        node_offset += n_nodes
    assert len(precisions) == len(num_nodes)
    assert len(recalls) == len(num_nodes)
    assert len(accuracies) == len(num_nodes)
    assert len(f1s) == len(num_nodes)
    return (
        sum(precisions) / len(precisions) if precisions else 0.0,
        sum(recalls) / len(recalls) if recalls else 0.0,
        sum(accuracies) / len(accuracies) if accuracies else 0.0,
        sum(f1s) / len(f1s) if f1s else 0.0,
    )


def recall_n(n: int, out: torch.Tensor, y: torch.Tensor, num_nodes: torch.Tensor):
    recalls = []
    node_offset = 0
    for n_nodes in num_nodes.tolist():
        out_q = out[node_offset : node_offset + n_nodes]
        y_q = y[node_offset : node_offset + n_nodes]
        if torch.sum(y_q[torch.argsort(out_q, descending=True)[:n]]) >= 1:
            recalls.append(1)
        else:
            recalls.append(0)
        node_offset += n_nodes
    return sum(recalls) / len(recalls) if recalls else 0.0


def compute_metrics(
    out: torch.Tensor, y: torch.Tensor, num_nodes: torch.Tensor, multi: bool = False
):
    precision, recall, accuracy, f1 = precision_recall_accuracy_f1(out, y)
    per_q_precision, per_q_recall, per_q_accuracy, per_q_f1 = (
        per_q_precision_recall_accuracy_f1(out, y, num_nodes)
    )
    metrics = {
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1": f1,
        "top_n_recall": top_n_recall(out, y, num_nodes),
        "mae": (out - y).abs().mean().item(),
        "mse": ((out - y) ** 2).mean().item(),
        "qs": num_nodes.shape[0],
        "per_q_precision": per_q_precision,
        "per_q_recall": per_q_recall,
        "per_q_accuracy": per_q_accuracy,
        "per_q_f1": per_q_f1,
    }
    if not multi:
        for i in [1, 3, 5, 10]:
            metrics["recall_" + str(i)] = recall_n(i, out, y, num_nodes)
    return metrics


def print_metrics(metrics: dict, indent: int = 0):
    pad = " " * indent
    for key, value in metrics.items():
        print(f"{pad}{key}: {value:.4f}")


def split_results_by_seen_unseen(
    out: torch.Tensor,
    y: torch.Tensor,
    qtype: torch.Tensor,
    num_nodes: torch.Tensor,
    split_dir: Path,
) -> tuple[dict, dict]:
    """Split flattened results tensors into seen and unseen subsets.

    "Seen" means the scan appeared in the training split.
    "Unseen" means the scan only appears in test/val, never in training.

    Returns:
        (seen, unseen) where each is a dict with keys: out, y, qtype, num_nodes
    """
    dataset_dir = split_dir.parent
    train_questions_dir = dataset_dir / "train" / "questions"
    train_scan_ids = (
        {p.stem for p in train_questions_dir.glob("*.pth")}
        if train_questions_dir.exists()
        else set()
    )

    # Read question files in sorted order (matches dataset/eval ordering)
    questions_dir = split_dir / "questions"
    scan_info = []  # [(scan_id, n_queries), ...]
    for qf in sorted(questions_dir.glob("*.pth")):
        q = torch.load(qf, weights_only=False)
        n_queries = q["y"].shape[0]
        scan_info.append((q["scanId"], n_queries))

    # Build per-query and per-node boolean masks
    seen_query_mask = []
    unseen_query_mask = []
    seen_node_mask = []
    unseen_node_mask = []

    query_offset = 0
    for scan_id, n_queries in scan_info:
        is_seen = scan_id in train_scan_ids
        for _ in range(n_queries):
            n_nodes = int(num_nodes[query_offset].item())
            seen_query_mask.append(is_seen)
            unseen_query_mask.append(not is_seen)
            seen_node_mask.extend([is_seen] * n_nodes)
            unseen_node_mask.extend([not is_seen] * n_nodes)
            query_offset += 1

    seen_qm = torch.tensor(seen_query_mask)
    unseen_qm = torch.tensor(unseen_query_mask)
    seen_nm = torch.tensor(seen_node_mask)
    unseen_nm = torch.tensor(unseen_node_mask)

    seen = {
        "out": out[seen_nm],
        "y": y[seen_nm],
        "qtype": qtype[seen_qm],
        "num_nodes": num_nodes[seen_qm],
    }
    unseen = {
        "out": out[unseen_nm],
        "y": y[unseen_nm],
        "qtype": qtype[unseen_qm],
        "num_nodes": num_nodes[unseen_qm],
    }
    return seen, unseen


def compute_and_print_by_qtype(out, y, qtype, num_nodes, multi, indent=2):
    qtype_per_node = qtype.repeat_interleave(num_nodes)
    qtype_vals = qtype.unique().tolist()
    by_qtype = {}
    for qt in sorted(qtype_vals):
        qt = int(qt)
        name = Q_TYPE_NAMES.get(qt, str(qt))
        node_mask = qtype_per_node == qt
        query_mask = qtype == qt
        print(f"\n{' ' * indent}[{name}]")
        metrics_qt = compute_metrics(
            out[node_mask], y[node_mask], num_nodes[query_mask], multi
        )
        print_metrics(metrics_qt, indent=indent + 4)
        by_qtype[name] = metrics_qt
    return by_qtype


def main():
    parser = argparse.ArgumentParser(
        description="Calculate metrics from a saved model output file."
    )
    parser.add_argument(
        "model_name", type=str, help="The model which produced the .pth results file."
    )
    parser.add_argument(
        "--multi",
        action="store_true",
        help="Whether the model was trained on a multi-answer dataset.",
    )
    parser.add_argument(
        "--split",
        "-s",
        type=str,
        default="test",
        help="Which split to calculate stats for.",
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=None,
        help=(
            "Root of the dataset (must contain train/, test/, val/ subdirs). "
            "Defaults to the balanced dataset directory for the configured encoders."
        ),
    )
    args = parser.parse_args()

    model_name = get_model_name(
        args.model_name,
        edge_encoder=EDGE_ENCODER,
        node_encoder=NODE_ENCODER,
        query_encoder=QUERY_ENCODER,
        multi=args.multi,
    )

    model_results_path = OUTPUT_DIR / "results" / model_name
    if args.split != "":
        prefix = args.split + "_"
    else:
        prefix = ""
    most_epochs = get_most_epochs_file(model_results_path, prefix=prefix)

    results = torch.load(
        model_results_path / (prefix + str(most_epochs) + ".pth"),
        map_location="cpu",
        weights_only=False,
    )
    out = results["out"]
    y = results["y"]
    qtype = results["qtype"]
    num_nodes = results["num_nodes"]

    print("=== Overall ===")
    overall = compute_metrics(out, y, num_nodes, multi=args.multi)
    print_metrics(overall)

    print("\n=== By question type ===")
    by_qtype = compute_and_print_by_qtype(out, y, qtype, num_nodes, args.multi)

    output = {"overall": overall, "by_qtype": by_qtype}

    # Seen / unseen breakdown
    if args.split:
        dataset_dir = args.dataset_dir
        if dataset_dir is None:
            if args.multi:
                dataset_dir = (
                    DATA_DIR
                    / "dataset_balanced"
                    / ";".join([EDGE_ENCODER, NODE_ENCODER, QUERY_ENCODER])
                )
            else:
                dataset_dir = (
                    DATA_DIR
                    / "dataset_single_balanced"
                    / ";".join([EDGE_ENCODER, NODE_ENCODER, QUERY_ENCODER])
                )
        split_dir = dataset_dir / args.split

        if split_dir.exists():
            seen, unseen = split_results_by_seen_unseen(
                out, y, qtype, num_nodes, split_dir
            )

            n_seen_q = int(seen["qtype"].shape[0])
            n_unseen_q = int(unseen["qtype"].shape[0])

            print(f"\n=== Seen scenes ({n_seen_q} queries) ===")
            seen_overall = compute_metrics(
                seen["out"], seen["y"], seen["num_nodes"], args.multi
            )
            print_metrics(seen_overall)
            print("\n  By question type:")
            seen_by_qtype = compute_and_print_by_qtype(
                seen["out"],
                seen["y"],
                seen["qtype"],
                seen["num_nodes"],
                args.multi,
                indent=4,
            )

            print(f"\n=== Unseen scenes ({n_unseen_q} queries) ===")
            unseen_overall = compute_metrics(
                unseen["out"], unseen["y"], unseen["num_nodes"], args.multi
            )
            print_metrics(unseen_overall)
            print("\n  By question type:")
            unseen_by_qtype = compute_and_print_by_qtype(
                unseen["out"],
                unseen["y"],
                unseen["qtype"],
                unseen["num_nodes"],
                args.multi,
                indent=4,
            )

            output["seen"] = {"overall": seen_overall, "by_qtype": seen_by_qtype}
            output["unseen"] = {"overall": unseen_overall, "by_qtype": unseen_by_qtype}
        else:
            print(f"\nWarning: dataset split directory not found: {split_dir}")
            print("Skipping seen/unseen breakdown.")

    out_path = model_results_path / (prefix + str(most_epochs) + "_metrics.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved metrics to {out_path}")


if __name__ == "__main__":
    main()
