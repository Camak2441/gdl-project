from typing import Any, Dict, List

import torch
from encoders._common import ALL_MINILM_L6_V2


def all_minilm_l6v2_encode(edges: List[List[Dict[str, Any]]]):
    descs = [", ".join(edge_d["name"] for edge_d in edge) for edge in edges]
    return torch.tensor(ALL_MINILM_L6_V2.encode(descs))
