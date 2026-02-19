import encoders.edge
import encoders.node
import encoders.query

EDGE_ENCODERS = {"all_minilm_l6v2": (encoders.edge.all_minilm_l6v2_encode, 384)}
NODE_ENCODERS = {"all_minilm_l6v2": (encoders.node.all_minilm_l6v2_encode, 384)}
QUERY_ENCODERS = {"all_minilm_l6v2": (encoders.query.all_minilm_l6v2_encode, 384)}


def get_edge_encoder(encoder: str):
    return EDGE_ENCODERS[encoder][0]


def get_encoded_edge_dim(encoder: str):
    return EDGE_ENCODERS[encoder][1]


def get_node_encoder(encoder: str):
    return NODE_ENCODERS[encoder][0]


def get_encoded_node_dim(encoder: str):
    return NODE_ENCODERS[encoder][1]


def get_query_encoder(encoder: str):
    return QUERY_ENCODERS[encoder][0]


def get_encoded_query_dim(encoder: str):
    return QUERY_ENCODERS[encoder][1]
