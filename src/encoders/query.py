import torch
from consts import DEVICE
from encoders._common import load_sentence_transformer


def all_minilm_l6v2_encode(queries: list[str]):
    return torch.tensor(
        load_sentence_transformer("all-MiniLM-L6-v2", DEVICE).encode(queries).squeeze()
    )
