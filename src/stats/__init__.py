import h5py
import numpy as np


def generate_training_stats(stats_schema, n_epochs):
    stats = {}
    for stat in stats_schema:
        match stats_schema[stat]:
            case "epoch":
                stats[stat] = np.zeros(n_epochs)
            case "train":
                stats[stat] = 0
            case _:
                raise KeyError("Unknown stat type " + str(stat))
    return stats


def update_epoch_stats(stats, epoch_stats, epoch, stats_schema=None):
    if stats_schema is not None:
        for stat in stats_schema:
            if stats_schema[stat] == "epoch" and stat not in epoch_stats:
                raise KeyError("Missing stat " + str(stat))
        for stat in epoch_stats:
            if stat not in stats_schema or stats_schema[stat] != "epoch":
                raise KeyError("Additional stat " + str(stat))
    for stat in epoch_stats:
        if stat in stats:
            stats[stat][epoch] = epoch_stats[stat]


def save_training_stats(stats, path):
    with h5py.File(path, "w") as hdf5_file:
        for key in stats:
            hdf5_file.create_dataset(name=key, data=stats[key])
