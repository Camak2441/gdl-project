import torch
from torch_geometric.loader import DataLoader
from pathlib import Path


def get_data_list(dir: Path):
    data = []
    for file_path in dir.iterdir():
        if file_path.suffix == ".pth":
            data.append(torch.load(file_path, weights_only=False))
    return data


def get_queried_graph_dataset(
    dir: Path,
    batch_size: int = 10,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
):
    return DataLoader(
        get_data_list(dir),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
