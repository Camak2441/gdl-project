from math import ceil
from tqdm import trange
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import RandomSampler, BatchSampler

from consts import DATA_DIR, OUTPUT_DIR
from data.utils import get_queried_graph_dataset
from stats import generate_training_stats, save_training_stats, update_epoch_stats
from train.utils import EVAL_STATS, eval_model, train
from models import canonical_model_name, load_model


MODEL_OUT_DIR = OUTPUT_DIR / "models"
STATS_OUT_DIR = OUTPUT_DIR / "stats"


EDGE_ENCODER = "all_minilm_l6v2"
NODE_ENCODER = "all_minilm_l6v2"
QUERY_ENCODER = "all_minilm_l6v2"


DATASET_DIR = (
    DATA_DIR / "dataset" / ";".join([EDGE_ENCODER, NODE_ENCODER, QUERY_ENCODER])
)


MODEL_NAME = canonical_model_name(
    f"QueryInGat(e_enc={EDGE_ENCODER},n_enc={NODE_ENCODER},q_enc={QUERY_ENCODER},hidden_dims=[128,128,128,128],out_dim=1,heads=4)"
)


EPOCHS = 1000
EPOCHS_FOR_ALL_DATA = 20


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_experiment(
    model: torch.nn.Module,
    train_set,
    test_set,
    val_set,
    n_epochs=100,
    epochs_for_all_data=20,
    device=DEVICE,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.9, patience=5, min_lr=0.00001
    )

    pbar = trange(n_epochs)

    training_stats = generate_training_stats(
        {
            "loss": "epoch",
            **{"val_" + key: "epoch" for key in EVAL_STATS},
            **{"test_" + key: "epoch" for key in EVAL_STATS},
        },
        n_epochs=n_epochs,
    )

    criterion = torch.nn.L1Loss()

    epoch_batch_size = ceil(len(train_set) / epochs_for_all_data)

    epoch_sets = BatchSampler(RandomSampler(train_set), epoch_batch_size, False)

    for epoch in pbar:
        lr = scheduler.optimizer.param_groups[0]["lr"]

        cycle_pos = epoch % epochs_for_all_data
        if cycle_pos == 0 and epoch > 0:
            epoch_sets = BatchSampler(RandomSampler(train_set), epoch_batch_size, False)

        loss = train(
            model=model,
            optimizer=optimizer,
            dataset=epoch_sets[cycle_pos],
            criterion=criterion,
            device=device,
        )
        val_stats = eval_model(
            model=model, dataset=val_set, criterion=criterion, device=device
        )
        test_stats = eval_model(
            model=model, dataset=test_set, criterion=criterion, device=device
        )

        update_epoch_stats(
            stats=training_stats,
            epoch_stats={
                "loss": loss,
                **{"val_" + key: val_stats[key] for key in val_stats},
                **{"test_" + key: test_stats[key] for key in test_stats},
            },
            epoch=epoch,
        )

        scheduler.step(val_stats["loss"])
        pbar.set_description(f"loss={loss:.4f}, lr={lr:.6f}")

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
        DATASET_DIR / "train",
        batch_size=100,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_set = get_queried_graph_dataset(
        DATASET_DIR / "test", batch_size=100, num_workers=4, pin_memory=True
    )
    val_set = get_queried_graph_dataset(
        DATASET_DIR / "val", batch_size=100, num_workers=4, pin_memory=True
    )

    print(f"Loading model {MODEL_NAME}...")

    model = load_model(MODEL_NAME)

    print("Running on device", DEVICE)

    model.to(DEVICE)
    model = torch.compile(model)
    training_stats = run_experiment(
        model,
        train_set,
        test_set,
        val_set,
        n_epochs=EPOCHS,
        epochs_for_all_data=EPOCHS_FOR_ALL_DATA,
        device=DEVICE,
    )

    this_model_dir = MODEL_OUT_DIR / MODEL_NAME
    this_model_dir.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), this_model_dir / (str(EPOCHS) + ".pth"))

    this_model_stats_dir = STATS_OUT_DIR / MODEL_NAME
    this_model_stats_dir.mkdir(exist_ok=True, parents=True)
    save_training_stats(training_stats, this_model_stats_dir / (str(EPOCHS) + ".h5"))


if __name__ == "__main__":
    main()
