from consts import DATA_DIR
from data.queried_scene_graph_dataset import QueriedSceneGraphDataset


EDGE_ENCODER = "all_minilm_l6v2"
NODE_ENCODER = "all_minilm_l6v2"
QUERY_ENCODER = "all_minilm_l6v2"


DATASET_DIR = (
    DATA_DIR
    / "dataset"
    / ";".join([EDGE_ENCODER, NODE_ENCODER, QUERY_ENCODER])
    / "train"
)
TARGET_INDICES = [0, 1000, 2000, 3000]


def main():
    data = QueriedSceneGraphDataset(DATASET_DIR)
    for i in TARGET_INDICES:
        print(data[i])


if __name__ == "__main__":
    main()
