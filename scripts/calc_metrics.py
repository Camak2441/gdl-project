"""
Calculate metrics from a saved model output file produced by eval_model.py.

Usage:
    python scripts/calc_metrics.py <results_file.pth>
"""

import argparse
import json
import math

import torch

from consts import Q_TYPES

THRESHOLD = 0.5

Q_TYPE_NAMES = {v: k for k, v in Q_TYPES.items()}


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


def recall_n(n: int, out: torch.Tensor, y: torch.Tensor, num_nodes: torch.Tensor):
    recalls = []
    node_offset = 0
    for n_nodes in num_nodes.tolist():
        out_q = out[node_offset : node_offset + n_nodes]
        y_q = y[node_offset : node_offset + n_nodes]
        ranking = torch.sum((torch.argsort(out_q, descending=True) + 1) * y_q).item()
        if ranking > 0 and ranking <= n:
            recalls.append(1)
        else:
            recalls.append(0)
        node_offset += n_nodes
    return sum(recalls) / len(recalls) if recalls else 0.0


def compute_metrics(
    out: torch.Tensor, y: torch.Tensor, num_nodes: torch.Tensor, multi: bool = True
):
    precision, recall, accuracy, f1 = precision_recall_accuracy_f1(out, y)
    metrics = {
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1": f1,
        "top_n_recall": top_n_recall(out, y, num_nodes),
        "mae": (out - y).abs().mean().item(),
        "mse": ((out - y) ** 2).mean().item(),
    }
    if not multi:
        for i in [1, 3, 5, 10]:
            metrics["recall_" + str(i)] = recall_n(i, out, y, num_nodes)
    return metrics


def print_metrics(metrics: dict, indent: int = 0):
    pad = " " * indent
    for key, value in metrics.items():
        print(f"{pad}{key}: {value:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate metrics from a saved model output file."
    )
    parser.add_argument("results_file", type=str, help="Path to the .pth results file.")
    args = parser.parse_args()

    results = torch.load(args.results_file, map_location="cpu", weights_only=False)
    out = results["out"]
    y = results["y"]
    qtype = results["qtype"]
    num_nodes = results["num_nodes"]

    print("=== Overall ===")
    overall = compute_metrics(out, y, num_nodes)
    print_metrics(overall)

    print("\n=== By question type ===")
    # qtype is per-query; expand to per-node for masking
    qtype_per_node = qtype.repeat_interleave(num_nodes)
    qtype_vals = qtype.unique().tolist()
    by_qtype = {}
    for qt in sorted(qtype_vals):
        qt = int(qt)
        name = Q_TYPE_NAMES.get(qt, str(qt))
        node_mask = qtype_per_node == qt
        query_mask = qtype == qt
        out_qt = out[node_mask]
        y_qt = y[node_mask]
        num_nodes_qt = num_nodes[query_mask]
        print(f"\n  [{name}]")
        metrics_qt = compute_metrics(out_qt, y_qt, num_nodes_qt)
        print_metrics(metrics_qt, indent=4)
        by_qtype[name] = metrics_qt

    out_path = args.results_file.replace(".pth", "_metrics.json")
    with open(out_path, "w") as f:
        json.dump({"overall": overall, "by_qtype": by_qtype}, f, indent=2)
    print(f"\nSaved metrics to {out_path}")


if __name__ == "__main__":
    main()
