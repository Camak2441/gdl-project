import json

import torch

from encoders._common import ALL_MINILM_L6_V2
from utils import intersect_dict


def encode_node(node):
    node = intersect_dict({"ply_color", "label", "affordances", "attributes"}, node)
    node_desc = json.dumps(node)
    return torch.tensor(ALL_MINILM_L6_V2.encode([node_desc]).squeeze())
