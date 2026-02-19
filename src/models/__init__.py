import json
from pathlib import Path
import re
from typing import Any, Dict

from encoders import get_encoded_edge_dim, get_encoded_node_dim, get_encoded_query_dim
from models.query_gat import QueryGAT
from models.query_in_gat import QueryInGAT
from models.query_in_gatv2 import QueryInGATv2
from models.multi_query_in_gat import MultiQueryInGAT
from utils import check_keys_are_in_order, quote_json


MODEL_PREFIXES = {
    "QueryGat": QueryGAT,
    "QueryInGat": QueryInGAT,
    "QueryInGatv2": QueryInGATv2,
    "MultiQueryInGat": MultiQueryInGAT,
}


def _load_argstr(argstr: str):
    return json.loads(quote_json("{" + argstr[1:-1].replace("=", ":") + "}"))


def canonical_model_name(model_name: str):
    for prefix in MODEL_PREFIXES:
        if model_name.startswith(prefix):
            argstr = model_name[len(prefix) :]
            if not (argstr.startswith("(") and argstr.endswith(")")):
                continue
            kwargs = _load_argstr(argstr)
            return (
                prefix
                + "("
                + "".join(
                    json.dumps(kwargs, separators=(",", "="), sort_keys=True).split()
                ).replace('"', "")[1:-1]
                + ")"
            )
    raise Exception(f"Unknown model {model_name}")


def load_model(model_name: str):
    for prefix in MODEL_PREFIXES:
        if model_name.startswith(prefix):
            argstr = model_name[len(prefix) :]
            if not (argstr.startswith("(") and argstr.endswith(")")):
                continue
            kwargs = _load_argstr(argstr)
            if not check_keys_are_in_order(kwargs):
                continue
            if "tag" in kwargs:
                kwargs.pop("tag")
            if "e_enc" in kwargs:
                e_encoder = kwargs.pop("e_enc")
                kwargs["edge_dim"] = get_encoded_edge_dim(e_encoder)
            if "n_enc" in kwargs:
                n_encoder = kwargs.pop("n_enc")
                kwargs["in_dim"] = get_encoded_node_dim(n_encoder)
            if "q_enc" in kwargs:
                q_encoder = kwargs.pop("q_enc")
                kwargs["query_dim"] = get_encoded_query_dim(q_encoder)
            return MODEL_PREFIXES[prefix](**kwargs)
    raise Exception(f"Unknown model {model_name}")


def load_encoders_from_model(model_name: str):
    for prefix in MODEL_PREFIXES:
        if model_name.startswith(prefix):
            argstr = model_name[len(prefix) :]
            if not (argstr.startswith("(") and argstr.endswith(")")):
                continue
            kwargs = _load_argstr(argstr)
            if not check_keys_are_in_order(kwargs):
                continue
            return (
                kwargs.get("e_enc", None),
                kwargs.get("n_enc", None),
                kwargs.get("q_enc", None),
            )
    raise Exception(f"Unknown model {model_name}")


def get_most_epochs_file(model_dir: Path):
    most_epochs = None
    for model_path in model_dir.iterdir():
        if re.match(r"0|[1-9][0-9]*", model_path.stem):
            file_epochs = int(model_path.stem)
            if most_epochs is None or file_epochs > most_epochs:
                most_epochs = file_epochs
    return most_epochs
