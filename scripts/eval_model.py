import argparse
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from consts import DATA_DIR, OUTPUT_DIR
from data import get_dataset
from data.queried_scene_graph_dataset import QueriedSceneGraphDataset
from models import (
    canonical_model_name,
    get_most_epochs_file,
    load_encoders_from_model,
    load_model,
    get_model_name,
)


MODEL_OUT_DIR = OUTPUT_DIR / "models"
RESULTS_OUT_DIR = OUTPUT_DIR / "results"

EDGE_ENCODER = "all_minilm_l6v2"
NODE_ENCODER = "all_minilm_l6v2"
QUERY_ENCODER = "all_minilm_l6v2"

DATASET_DIR = (
    DATA_DIR
    / "dataset_balanced"
    / ";".join([EDGE_ENCODER, NODE_ENCODER, QUERY_ENCODER])
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 256


def select_interactively(options: list[str], prompt: str) -> str:
    print(f"\n{prompt}")
    for i, option in enumerate(options):
        print(f"  [{i}] {option}")
    while True:
        raw = input("Enter number: ").strip()
        if raw.isdigit() and 0 <= int(raw) < len(options):
            return options[int(raw)]
        print(f"Please enter a number between 0 and {len(options) - 1}.")


def get_available_models() -> list[str]:
    if not MODEL_OUT_DIR.exists():
        return []
    return sorted(p.name for p in MODEL_OUT_DIR.iterdir() if p.is_dir())


def get_available_datasets() -> list[str]:
    base = DATA_DIR / "dataset_balanced"
    if not base.exists():
        return []
    return sorted(p.name for p in base.iterdir() if p.is_dir())


def dataset_dir_from_model_name(model_name: str) -> Path | None:
    e_enc, n_enc, q_enc = load_encoders_from_model(model_name)
    if e_enc is None or n_enc is None or q_enc is None:
        return None
    return DATA_DIR / "dataset_balanced" / ";".join([e_enc, n_enc, q_enc])


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on the test set."
    )
    parser.add_argument(
        "model",
        type=str,
        help="Model to train",
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="Dataset to train the model on",
    )
    parser.add_argument(
        "--multi",
        type=bool,
        default=True,
        help=f"Whether the dataset allows multiple answers per question or not",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Interactively select model and dataset.",
    )
    args = parser.parse_args()

    dataset_dir, multi = get_dataset(
        args.dataset,
        node_encoder=NODE_ENCODER,
        edge_encoder=EDGE_ENCODER,
        query_encoder=QUERY_ENCODER,
    )
    if multi is None:
        multi = args.multi
    model_name = get_model_name(
        args.model,
        node_encoder=NODE_ENCODER,
        edge_encoder=EDGE_ENCODER,
        query_encoder=QUERY_ENCODER,
        multi=multi,
    )

    if args.interactive:
        models = get_available_models()
        if not models:
            print(f"No models found in {MODEL_OUT_DIR}")
            return
        model_name = select_interactively(models, "Select a model:")

        inferred = dataset_dir_from_model_name(model_name)
        if inferred is not None and inferred.exists():
            print(f"\nDataset inferred from model name: {inferred.name}")
            use_inferred = input("Use this dataset? [Y/n]: ").strip().lower()
            if use_inferred in ("", "y", "yes"):
                dataset_dir = inferred
            else:
                datasets = get_available_datasets()
                if not datasets:
                    print(f"No datasets found in {DATA_DIR / 'dataset_balanced'}")
                    return
                dataset_enc = select_interactively(datasets, "Select a dataset:")
                dataset_dir = DATA_DIR / "dataset_balanced" / dataset_enc
        else:
            datasets = get_available_datasets()
            if not datasets:
                print(f"No datasets found in {DATA_DIR / 'dataset_balanced'}")
                return
            dataset_enc = select_interactively(datasets, "Select a dataset:")
            dataset_dir = DATA_DIR / "dataset_balanced" / dataset_enc

    splits = [p.name for p in dataset_dir.iterdir() if p.is_dir()]
    if not splits:
        print(f"No splits found in {dataset_dir}")
        return

    model_dir = MODEL_OUT_DIR / model_name
    if not model_dir.exists():
        print(f"Model directory not found: {model_dir}")
        return

    most_epochs = get_most_epochs_file(model_dir)
    if most_epochs is None:
        print(f"No checkpoint files found in {model_dir}")
        return
    model_path = model_dir / f"{most_epochs}.pth"

    print(f"\nModel:    {model_name}")
    print(f"Weights:  {model_path.name} ({most_epochs} epochs)")
    print(f"Dataset:  {dataset_dir.name}")
    print(f"Splits:   {', '.join(splits)}")
    print(f"Device:   {DEVICE}")

    print("\nLoading model...")
    model = load_model(model_name)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    out_dir = RESULTS_OUT_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        print(f"\nRunning model on {split} split...")
        split_set = QueriedSceneGraphDataset(dataset_dir / split)
        split_loader = DataLoader(
            split_set, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True
        )

        all_out = []
        all_y = []
        all_qtype = []
        all_num_nodes = []
        with torch.no_grad():
            for data in split_loader:
                data.to(DEVICE)
                out = model(
                    x=data.x,
                    edge_index=data.edge_index,
                    edge_attr=data.edge_attr,
                    query=data.query,
                    batch=data.batch,
                )
                all_out.append(out.cpu())
                all_y.append(data.y.cpu())
                all_qtype.append(data.qtype.cpu())
                all_num_nodes.append(torch.bincount(data.batch.cpu()))

        out_file = out_dir / f"{split}_{most_epochs}.pth"
        torch.save(
            {
                "out": torch.cat(all_out),
                "y": torch.cat(all_y),
                "qtype": torch.cat(all_qtype),
                "num_nodes": torch.cat(all_num_nodes),
            },
            out_file,
        )
        print(f"  Saved to {out_file}")


if __name__ == "__main__":
    main()
