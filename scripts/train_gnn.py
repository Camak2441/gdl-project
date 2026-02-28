import argparse
from pathlib import Path
import re

from tqdm import trange
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from consts import DATA_DIR, OUTPUT_DIR
from data import get_dataset
from data.queried_scene_graph_dataset import QueriedSceneGraphDataset
from losses import BatchLoss, RecallPrecisionWeightedLoss, WeightedLosses
from stats import generate_training_stats, save_training_stats, update_epoch_stats
from train.utils import (
    EVAL_MULTI_STATS,
    EVAL_STATS,
    eval_model,
    eval_multi_model,
    train,
)
from models import canonical_model_name, get_model_name, load_model


MODEL_OUT_DIR = OUTPUT_DIR / "models"
STATS_OUT_DIR = OUTPUT_DIR / "stats"


EDGE_ENCODER = "all_minilm_l6v2"
NODE_ENCODER = "all_minilm_l6v2"
QUERY_ENCODER = "all_minilm_l6v2"


MULTI_ANSWER_DATASET = True
MULTI_ANSWER_ARG = str(MULTI_ANSWER_DATASET).lower()


EPOCHS = 1000
EPOCHS_FOR_ALL_DATA = 20


CHECKPOINT_EPOCHS = set(range(0, 10000, 100))

MIN_LR = 1e-6

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _cleanup_old_checkpoints(model_name, final_epoch):
    """Delete model/stats checkpoint files with epoch numbers > final_epoch."""
    for base_dir, suffix in [(MODEL_OUT_DIR, ".pth"), (STATS_OUT_DIR, ".h5")]:
        dir_ = base_dir / model_name
        if not dir_.exists():
            continue
        for f in dir_.iterdir():
            if f.suffix == suffix:
                try:
                    if int(f.stem) > final_epoch:
                        f.unlink()
                        print(f"Deleted stale checkpoint: {f}")
                except ValueError:
                    pass


def run_experiment(
    model: torch.nn.Module,
    model_name,
    train_set,
    val_set,
    test_set,
    n_epochs=100,
    batch_size=256,
    epochs_for_all_data=20,
    device=DEVICE,
    checkpoint_epochs=set(),
    multi: bool = False,
    min_lr: float = MIN_LR,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.9, patience=5, min_lr=min_lr
    )

    eval_fn = eval_multi_model if multi else eval_model
    stat_keys = EVAL_MULTI_STATS if multi else EVAL_STATS

    training_stats = generate_training_stats(
        {
            "loss": "epoch",
            **{"val_" + key: "epoch" for key in stat_keys},
        },
        n_epochs=n_epochs,
    )

    criterion = BatchLoss(
        RecallPrecisionWeightedLoss(
            loss=WeightedLosses(
                [
                    torch.nn.L1Loss(reduction="none"),
                    torch.nn.MSELoss(reduction="none"),
                ],
                [0.8, 0.2],
            ),
        )
    )
    criterion.to(device)

    split = [1.0 / epochs_for_all_data] * (epochs_for_all_data - 1)
    split.append(1 - sum(split))

    def make_epoch_sets():
        nonlocal split
        return random_split(train_set, split)

    epoch_sets = make_epoch_sets()

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
    )

    final_epoch = n_epochs

    pbar = trange(n_epochs)
    for epoch in pbar:
        lr = scheduler.optimizer.param_groups[0]["lr"]

        cycle_pos = epoch % epochs_for_all_data
        if cycle_pos == 0 and epoch > 0:
            epoch_sets = make_epoch_sets()

        epoch_loader = DataLoader(
            epoch_sets[cycle_pos],
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
        )

        loss = train(
            model=model,
            optimizer=optimizer,
            dataset=epoch_loader,
            criterion=criterion,
            device=device,
        )
        val_stats = eval_fn(
            model=model, dataset=val_loader, criterion=criterion, device=device
        )

        update_epoch_stats(
            stats=training_stats,
            epoch_stats={
                "loss": loss,
                **{"val_" + key: val_stats[key] for key in val_stats},
            },
            epoch=epoch,
        )

        scheduler.step(val_stats["loss"])
        pbar.set_description(f"loss={loss:.4f}, lr={lr:.6f}")

        if epoch + 1 in checkpoint_epochs:
            this_model_dir = MODEL_OUT_DIR / model_name
            this_model_dir.mkdir(exist_ok=True, parents=True)
            torch.save(model.state_dict(), this_model_dir / (str(epoch + 1) + ".pth"))

            this_model_stats_dir = STATS_OUT_DIR / model_name
            this_model_stats_dir.mkdir(exist_ok=True, parents=True)
            save_training_stats(
                training_stats, this_model_stats_dir / (str(epoch + 1) + ".h5")
            )

        new_lr = scheduler.optimizer.param_groups[0]["lr"]
        if new_lr <= min_lr:
            final_epoch = epoch + 1
            print(
                f"\nEarly stopping: LR hit minimum ({min_lr:.2e}) at epoch {final_epoch}"
            )
            _cleanup_old_checkpoints(model_name, final_epoch)
            break

    return training_stats, final_epoch


def main():
    parser = argparse.ArgumentParser(description="Trains GNN on the dataset")
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
        "--min-lr",
        type=float,
        default=MIN_LR,
        help=f"Minimum learning rate; training stops when LR hits this value (default: {MIN_LR})",
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

    print(f"Loading dataset {dataset_dir}...")

    train_set = QueriedSceneGraphDataset(dataset_dir / "train")
    val_set = QueriedSceneGraphDataset(dataset_dir / "val")
    test_set = QueriedSceneGraphDataset(dataset_dir / "test")

    model_name = get_model_name(
        args.model,
        edge_encoder=EDGE_ENCODER,
        node_encoder=NODE_ENCODER,
        query_encoder=QUERY_ENCODER,
        multi=multi,
    )

    print(f"Loading model {model_name}...")

    model = load_model(model_name)

    print("Running on device", DEVICE)

    model.to(DEVICE)
    training_stats, final_epoch = run_experiment(
        model,
        model_name,
        train_set,
        val_set,
        test_set,
        n_epochs=EPOCHS,
        epochs_for_all_data=EPOCHS_FOR_ALL_DATA,
        device=DEVICE,
        checkpoint_epochs=CHECKPOINT_EPOCHS,
        multi=multi,
        min_lr=args.min_lr,
    )

    this_model_dir = MODEL_OUT_DIR / model_name
    this_model_dir.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), this_model_dir / (str(final_epoch) + ".pth"))

    this_model_stats_dir = STATS_OUT_DIR / model_name
    this_model_stats_dir.mkdir(exist_ok=True, parents=True)
    save_training_stats(
        training_stats, this_model_stats_dir / (str(final_epoch) + ".h5")
    )


if __name__ == "__main__":
    main()
