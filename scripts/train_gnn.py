from tqdm import trange
import torch
import numpy as np
import matplotlib as plt
import h5py

from consts import DATA_DIR, OUTPUT_DIR
from data.utils import get_queried_graph_dataset
from stats import generate_training_stats, save_training_stats, update_epoch_stats
from train.utils import eval_model, train
from models import load_model


MODEL_OUT_DIR = OUTPUT_DIR / "models"
STATS_OUT_DIR = OUTPUT_DIR / "stats"


MODEL_NAME = "query_gat_384,128,128,128"


NODE_DIM = 384
EDGE_DIM = 384
QUERY_DIM = 384
OUT_DIM = 1


EPOCHS = 100


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_experiment(
    model: torch.nn.Module, train_set, test_set, val_set, n_epochs=100, device=DEVICE
):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    pbar = trange(n_epochs)

    training_stats = generate_training_stats(
        {"loss": "epoch", "test_loss": "epoch", "val_loss": "epoch"}, n_epochs=n_epochs
    )

    criterion = torch.nn.L1Loss()

    for epoch in pbar:
        loss = train(
            model=model,
            optimizer=optimizer,
            dataset=train_set,
            criterion=criterion,
            device=device,
        )
        test_loss = eval_model(
            model=model, dataset=test_set, criterion=criterion, device=device
        )
        val_loss = eval_model(
            model=model, dataset=val_set, criterion=criterion, device=device
        )

        update_epoch_stats(
            stats=training_stats,
            epoch_stats={"loss": loss, "test_loss": test_loss, "val_loss": val_loss},
            epoch=epoch,
        )

        pbar.set_description(f"loss = {loss:.4f}")

    return training_stats


def plot_stats(training_stats, figsize=(5, 5), name=""):
    """Create one plot for each metric stored in training_stats"""
    stats_names = [key[6:] for key in training_stats.keys() if key.startswith("train_")]
    f, ax = plt.subplots(len(stats_names), 1, figsize=figsize)
    if len(stats_names) == 1:
        ax = np.array([ax])
    for key, axx in zip(
        stats_names,
        ax.reshape(
            -1,
        ),
    ):
        axx.plot(
            training_stats["epoch"],
            training_stats[f"train_{key}"],
            label=f"Training {key}",
        )
        axx.plot(
            training_stats["epoch"],
            training_stats[f"val_{key}"],
            label=f"Validation {key}",
        )
        axx.set_xlabel("Training epoch")
        axx.set_ylabel(key)
        axx.legend()
    plt.title(name)


def main():

    print("Loading datasets...")

    train_set = get_queried_graph_dataset(
        DATA_DIR / "dataset" / "train",
        batch_size=100,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_set = get_queried_graph_dataset(
        DATA_DIR / "dataset" / "test", batch_size=100, num_workers=4, pin_memory=True
    )
    val_set = get_queried_graph_dataset(
        DATA_DIR / "dataset" / "val", batch_size=100, num_workers=4, pin_memory=True
    )

    print("Loading model...")

    model = load_model(NODE_DIM, QUERY_DIM, EDGE_DIM, OUT_DIM, MODEL_NAME)

    print("Running on device", DEVICE)

    model.to(DEVICE)
    training_stats = run_experiment(
        model, train_set, test_set, val_set, n_epochs=EPOCHS, device=DEVICE
    )

    this_model_dir = MODEL_OUT_DIR / MODEL_NAME
    this_model_dir.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), this_model_dir / (str(EPOCHS) + ".pth"))

    STATS_OUT_DIR.mkdir(exist_ok=True, parents=True)
    this_model_stats_dir = STATS_OUT_DIR / (MODEL_NAME + ".h5")
    save_training_stats(training_stats, this_model_stats_dir)


if __name__ == "__main__":
    main()
