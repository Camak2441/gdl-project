from typing import Any, Dict, List

import torch
from encoders._common import load_sentence_transformer
from consts import DEVICE


def all_minilm_l6v2_encode(edges: List[List[Dict[str, Any]]]):
    descs = [", ".join(edge_d["name"] for edge_d in edge) for edge in edges]
    return torch.tensor(
        load_sentence_transformer("all-MiniLM-L6-v2", DEVICE).encode(descs)
    )
