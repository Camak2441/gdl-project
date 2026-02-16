import json
from typing import Any, Dict, List

import torch

from encoders._common import ALL_MINILM_L6_V2
from utils import intersect_dict


def all_minilm_l6v2_encode(nodes: List[Dict[str, Any]]):
    descs = []
    for node in nodes:
        node = intersect_dict({"ply_color", "label", "affordances", "attributes"}, node)
        descs.append(json.dumps(node))
    return torch.tensor(ALL_MINILM_L6_V2.encode(descs))
