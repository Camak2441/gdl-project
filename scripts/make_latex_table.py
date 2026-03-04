"""
Build a LaTeX table from saved metric results.

Usage (run from src/):
    python ../scripts/make_latex_table.py MODEL [MODEL ...] [options]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from consts import OUTPUT_DIR
from models import get_model_name, get_most_epochs_file

RESULTS_DIR = OUTPUT_DIR / "results"

DEFAULT_ENCODER = "all_minilm_l6v2"

ALL_METRICS_SINGLE = [
    "precision",
    "recall",
    "accuracy",
    "f1",
    "top_n_recall",
    "recall_1",
    "recall_3",
    "recall_5",
    "recall_10",
    "mae",
    "mse",
]
ALL_METRICS_MULTI = [
    "precision",
    "recall",
    "accuracy",
    "f1",
    "top_n_recall",
    "mae",
    "mse",
]

METRIC_DISPLAY = {
    "precision": "Prec.",
    "recall": "Recall",
    "accuracy": "Acc.",
    "f1": "F1",
    "top_n_recall": "Top-N Recall",
    "recall_1": "R@1",
    "recall_3": "R@3",
    "recall_5": "R@5",
    "recall_10": "R@10",
    "mae": "MAE",
    "mse": "MSE",
}

QTYPE_DISPLAY = {
    "overall": "Overall",
    "semantic": "Semantic",
    "spatial": "Spatial",
    "support": "Support",
    "compound": "Compound",
    "complex": "Complex",
}


def fmt(value, sig_figs: int) -> str:
    if value is None or value != value:  # nan check
        return "---"
    return f"{value:.{sig_figs}f}"


def find_metrics_file(model_dir: Path, split: str) -> Path | None:
    prefix = split + "_" if split else ""
    most_epochs = get_most_epochs_file(model_dir, prefix=prefix)
    if most_epochs is not None:
        return model_dir / f"{prefix}{most_epochs}_metrics.json"
    # Fallback: plain <split>_metrics.json (for llm/rag results)
    candidate = model_dir / f"{prefix}metrics.json"
    if candidate.exists():
        return candidate
    return None


def load_metrics(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def latex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def make_table(
    row_labels: list[str],
    metrics_data: list[dict],  # list of {"overall": {...}, "by_qtype": {...}}
    selected_metrics: list[str],
    by_qtype: bool,
    sig_figs: int,
    caption: str = "",
    label: str = "tab:results",
    scene_split: str | None = None,
    qtype_filter: str | None = None,
) -> str:
    # qtype_filter selects a single question type; by_qtype shows all types.
    # qtype_filter takes precedence (they are mutually exclusive at the CLI level).
    if qtype_filter is not None:
        qtypes = [qtype_filter]
    elif by_qtype:
        qtypes = ["overall"] + list(QTYPE_DISPLAY.keys())[1:]
    else:
        qtypes = ["overall"]

    # Show a group-header row when there are multiple groups or when a single
    # qtype is selected (to make the filter context visible in the table).
    show_group_header = len(qtypes) > 1 or qtype_filter is not None

    # Build column spec
    # Model | [qtype group: metric cols]*
    n_metric_cols = len(selected_metrics) * len(qtypes)
    col_spec = "l" + "r" * n_metric_cols

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    if caption:
        lines.append(rf"\caption{{{latex_escape(caption)}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    if show_group_header:
        # Header row 1: qtype groups
        qtype_header_parts = [""]
        for qt in qtypes:
            span = len(selected_metrics)
            qtype_header_parts.append(
                rf"\multicolumn{{{span}}}{{c}}{{{QTYPE_DISPLAY[qt]}}}"
            )
        lines.append(" & ".join(qtype_header_parts) + r" \\")
        # Cmidrule under each group
        cmidrules = []
        for i in range(len(qtypes)):
            start = 2 + i * len(selected_metrics)
            end = start + len(selected_metrics) - 1
            cmidrules.append(rf"\cmidrule(lr){{{start}-{end}}}")
        lines.append(" ".join(cmidrules))

    # Header row: metric names
    metric_headers = ["Model"]
    for qt in qtypes:
        for m in selected_metrics:
            metric_headers.append(METRIC_DISPLAY.get(m, latex_escape(m)))
    lines.append(" & ".join(metric_headers) + r" \\")
    lines.append(r"\midrule")

    # Data rows
    for label_str, data in zip(row_labels, metrics_data):
        row = [latex_escape(label_str)]
        # Navigate into seen/unseen sub-dict when scene_split is specified
        base = data.get(scene_split, {}) if scene_split else data
        for qt in qtypes:
            if qt == "overall":
                metrics = base.get("overall", {})
            else:
                metrics = base.get("by_qtype", {}).get(qt, {})
            for m in selected_metrics:
                row.append(fmt(metrics.get(m), sig_figs))
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a LaTeX table from saved model metric results."
    )
    parser.add_argument(
        "models",
        nargs="+",
        type=str,
        help="Model names (shorthands or full canonical names).",
    )
    parser.add_argument(
        "--multi",
        action="store_true",
        help="Use multi-answer mode when resolving model names.",
    )
    parser.add_argument(
        "--split",
        "-s",
        type=str,
        default="test",
        help="Dataset split to load metrics for (default: test).",
    )
    parser.add_argument(
        "--metrics",
        "-m",
        nargs="+",
        type=str,
        default=None,
        help="Metrics to include (default: f1 top_n_recall recall_1 recall_3 recall_5 recall_10 for single; f1 top_n_recall for multi).",
    )
    parser.add_argument(
        "--by-qtype",
        action="store_true",
        help="Break down metrics by question type.",
    )
    parser.add_argument(
        "--qtype",
        choices=list(QTYPE_DISPLAY.keys())[1:],
        default=None,
        metavar="QTYPE",
        help=(
            "Show metrics for a single question type only "
            f"(choices: {', '.join(list(QTYPE_DISPLAY.keys())[1:])}). "
            "Mutually exclusive with --by-qtype."
        ),
    )
    parser.add_argument(
        "--scene-split",
        choices=["seen", "unseen"],
        default=None,
        metavar="SCENE_SPLIT",
        help=(
            "Filter to seen or unseen scenes (seen = scene graph was in training set, "
            "unseen = scene graph was held out). Requires the metrics file to have been "
            "generated with seen/unseen breakdown."
        ),
    )
    parser.add_argument(
        "--sig-figs",
        type=int,
        default=3,
        help="Number of significant figures (default: 3).",
    )
    parser.add_argument(
        "--caption",
        type=str,
        default="",
        help="LaTeX table caption.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="tab:results",
        help="LaTeX table label (default: tab:results).",
    )
    parser.add_argument(
        "--row-labels",
        nargs="+",
        type=str,
        default=None,
        help="Custom display names for each model row (must match number of models).",
    )
    args = parser.parse_args()

    if args.row_labels and len(args.row_labels) != len(args.models):
        parser.error("--row-labels must have the same number of entries as models.")

    if args.by_qtype and args.qtype:
        parser.error("--by-qtype and --qtype are mutually exclusive.")

    selected_metrics = args.metrics
    if selected_metrics is None:
        if args.multi:
            selected_metrics = ["f1", "top_n_recall"]
        else:
            selected_metrics = [
                "f1",
                "top_n_recall",
                "recall_1",
                "recall_3",
                "recall_5",
                "recall_10",
            ]

    row_labels = []
    metrics_data = []

    for i, model_arg in enumerate(args.models):
        display_name = args.row_labels[i] if args.row_labels else model_arg

        resolved = get_model_name(
            model_arg,
            node_encoder=DEFAULT_ENCODER,
            edge_encoder=DEFAULT_ENCODER,
            query_encoder=DEFAULT_ENCODER,
            multi=args.multi,
            allow_llms=True,
        )

        model_dir = RESULTS_DIR / resolved
        if not model_dir.exists():
            print(
                f"[warning] results directory not found: {model_dir}", file=sys.stderr
            )
            metrics_data.append({})
            row_labels.append(display_name)
            continue

        metrics_file = find_metrics_file(model_dir, args.split)
        if metrics_file is None or not metrics_file.exists():
            print(
                f"[warning] no metrics file found for {resolved} (split={args.split})",
                file=sys.stderr,
            )
            metrics_data.append({})
            row_labels.append(display_name)
            continue

        data = load_metrics(metrics_file)
        metrics_data.append(data)
        row_labels.append(display_name)
        print(
            f"[loaded] {display_name} <- {metrics_file.relative_to(OUTPUT_DIR.parent)}",
            file=sys.stderr,
        )

    table = make_table(
        row_labels=row_labels,
        metrics_data=metrics_data,
        selected_metrics=selected_metrics,
        by_qtype=args.by_qtype,
        sig_figs=args.sig_figs,
        caption=args.caption,
        label=args.label,
        scene_split=args.scene_split,
        qtype_filter=args.qtype,
    )
    print(table)


if __name__ == "__main__":
    main()
