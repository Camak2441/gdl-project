import torch
from encoders._common import ALL_MINILM_L6_V2


def all_minilm_l6v2_encode(query):
    return torch.tensor(ALL_MINILM_L6_V2.encode([query]).squeeze())
