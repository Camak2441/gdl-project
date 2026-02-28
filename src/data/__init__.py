from pathlib import Path

from consts import DATA_DIR
from data.query_data import QueryData
from data.queried_scene_graph_dataset import QueriedSceneGraphDataset

DATASETS = {
    "balanced": {
        "path": DATA_DIR / "dataset_balanced",
        "multi": True,
    },
    "single": {
        "path": DATA_DIR / "dataset_single_answer",
        "multi": False,
    },
    "single_balanced": {
        "path": DATA_DIR / "dataset_single_balanced",
        "multi": False,
    },
}


def get_dataset(name, edge_encoder, node_encoder, query_encoder):
    if name in DATASETS:
        dataset_info = DATASETS[name]
        return (
            dataset_info["path"]
            / ";".join([edge_encoder, node_encoder, query_encoder]),
            dataset_info["multi"],
        )
    return Path(name), None
