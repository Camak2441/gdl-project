import torch
from encoders._common import ALL_MINILM_L6_V2


def encode_edge(edge):
    edge_desc = ", ".join(edge_d["name"] for edge_d in edge)
    return torch.tensor(ALL_MINILM_L6_V2.encode([edge_desc]).squeeze())
