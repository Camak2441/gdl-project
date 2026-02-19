import json
from typing import Any, Dict, List

import torch

from consts import DEVICE
from encoders._common import load_sentence_transformer
from utils import intersect_dict


def all_minilm_l6v2_encode(nodes: List[Dict[str, Any]]):
    descs = []
    for node in nodes:
        node = intersect_dict({"ply_color", "label", "affordances", "attributes"}, node)
        descs.append(json.dumps(node))
    return torch.tensor(
        load_sentence_transformer("all-MiniLM-L6-v2", DEVICE).encode(descs)
    )
