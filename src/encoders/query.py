import torch
from encoders._common import ALL_MINILM_L6_V2


def encode_query(query):
    return torch.tensor(ALL_MINILM_L6_V2.encode([query]).squeeze())
