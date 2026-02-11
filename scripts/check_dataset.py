from consts import DATA_DIR
from data.utils import get_data_list

DATASET_DIR = DATA_DIR / "dataset" / "train"
TARGET_INDICES = [0]


def main():
    data = get_data_list(DATASET_DIR)
    for i in TARGET_INDICES:
        print(data[i])


if __name__ == "__main__":
    main()
