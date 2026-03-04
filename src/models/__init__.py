import json
from pathlib import Path
import re

from encoders import get_encoded_edge_dim, get_encoded_node_dim, get_encoded_query_dim
from models.gat import Gat
from models.query_gat import QueryGat
from models.query_in_gat import QueryInGat
from models.query_in_gatv2 import QueryInGatv2
from models.virtual_node_wrapper import VirtualNodeWrapper
from models.qs_gat import QsGat
from models.qs_gnn import QsGnn
from models.query_mpn import QueryMpn
from utils import add_dict, check_keys_are_in_order, quote_json


MODEL_PREFIXES = {
    "Gat": Gat,
    "QueryGat": QueryGat,
    "QueryInGat": QueryInGat,
    "QueryInGatv2": QueryInGatv2,
    "VirtualNodeWrapper": VirtualNodeWrapper,
    "QsGat": QsGat,
    "QsGnn": QsGnn,
    "QueryMpn": QueryMpn,
}


GET_SUBMODEL_ARGS = {"VirtualNodeWrapper": {"model": VirtualNodeWrapper.get_args}}


def _dump_argstr(kwargs):
    match kwargs:
        case bool():
            return str(kwargs).lower()
        case int():
            return str(kwargs)
        case float():
            return str(kwargs)
        case str():
            return '"' + kwargs + '"'
        case list():
            s = []
            for item in kwargs:
                s.append(_dump_argstr(item))
            return "[" + ",".join(s) + "]"
        case dict():
            s = []
            for key in kwargs:
                s.append(key + "=" + _dump_argstr(kwargs[key]))
            return "(" + ",".join(s) + ")"


def _load_argstr(argstr: str):
    return json.loads(quote_json("{" + argstr[1:-1].replace("=", ":") + "}"))


def canonical_model_name(model_name: str):
    for prefix in MODEL_PREFIXES:
        if model_name.startswith(prefix):
            argstr = model_name[len(prefix) :]
            if not (argstr.startswith("(") and argstr.endswith(")")):
                continue
            kwargs = _load_argstr(argstr)
            for arg in kwargs:
                if arg.endswith("model"):
                    kwargs[arg] = canonical_model_name(kwargs[arg])
            return prefix + _dump_argstr(kwargs)
    raise Exception(f"Unknown model {model_name}")


def load_model(model_name: str, over_kwargs={}):
    for prefix in MODEL_PREFIXES:
        if model_name.startswith(prefix):
            argstr = model_name[len(prefix) :]
            if not (argstr.startswith("(") and argstr.endswith(")")):
                continue
            kwargs = _load_argstr(argstr)
            kwargs = add_dict(over_kwargs, kwargs)

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
            for arg in kwargs:
                if arg.endswith("model"):
                    sub_model_kwargs = GET_SUBMODEL_ARGS[prefix][arg](**kwargs)
                    kwargs[arg] = load_model(kwargs[arg], sub_model_kwargs)
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


def get_most_epochs_file(model_dir: Path, prefix=""):
    most_epochs = None
    for model_path in model_dir.iterdir():
        stem = model_path.stem
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
            if re.fullmatch(r"0|[1-9][0-9]*", stem):
                file_epochs = int(stem)
                if most_epochs is None or file_epochs > most_epochs:
                    most_epochs = file_epochs
    return most_epochs


MODEL_SHORTHANDS = {
    "qigat": "QueryInGat(\
        e_enc={edge_encoder},\
        n_enc={node_encoder},\
        q_enc={query_encoder},\
        multi={multi_answer},\
        hidden_dims=[64,64,64],\
        out_dim=1,\
        heads=4\
    )",
    "qmpn": "QueryMpn(\
        e_enc={edge_encoder},\
        n_enc={node_encoder},\
        q_enc={query_encoder},\
        multi={multi_answer},\
        hidden_dims=[128,128,128],\
        out_dim=1\
    )",
    "vngnn": 'VirtualNodeWrapper(\
        e_enc={edge_encoder},\
        n_enc={node_encoder},\
        q_enc={query_encoder},\
        multi={multi_answer},\
        model="Gat(\
            hidden_dims=[64,64,64],\
            out_dim=1,\
            heads=4\
        )",\
        query_mlp=false\
    )',
    "vngnn2": 'VirtualNodeWrapper(\
        e_enc={edge_encoder},\
        n_enc={node_encoder},\
        q_enc={query_encoder},\
        multi={multi_answer},\
        model="Gat(\
            hidden_dims=[128,128,128],\
            out_dim=1,\
            heads=4\
        )",\
        query_mlp=true\
    )',
    "qsgat": "QsGat(\
        e_enc={edge_encoder},\
        n_enc={node_encoder},\
        q_enc={query_encoder},\
        multi={multi_answer},\
        hidden_dims=[128,128,128],\
        out_dim=1\
    )",
    "qsgnn": "QsGnn(\
        e_enc={edge_encoder},\
        n_enc={node_encoder},\
        q_enc={query_encoder},\
        multi={multi_answer},\
        hidden_dims=[128,128,128]\
    )",
}


LLM_MODEL_NAMES = {
    "llm_gpt-4.1-nano",
    "llm_gpt-4.1-nano_single",
    "llm_gpt-5-mini",
    "rag_gpt-4.1-nano_k10",
    "rag_gpt-4.1-nano_single_k10",
}


def get_model_name(
    model_name: str, node_encoder, edge_encoder, query_encoder, multi, allow_llms=False
):
    if allow_llms:
        for llm_name in LLM_MODEL_NAMES:
            if model_name == llm_name:
                return model_name
            if model_name.startswith(llm_name + "/"):
                return (
                    llm_name
                    + "/"
                    + get_model_name(
                        model_name[len(llm_name) + 1 :],
                        node_encoder=node_encoder,
                        edge_encoder=edge_encoder,
                        query_encoder=query_encoder,
                        multi=multi,
                        allow_llms=allow_llms,
                    )
                )
    while model_name in MODEL_SHORTHANDS:
        model_name = MODEL_SHORTHANDS[model_name]
    return canonical_model_name(
        model_name.format(
            node_encoder=node_encoder,
            edge_encoder=edge_encoder,
            query_encoder=query_encoder,
            multi_answer=str(multi).lower(),
        )
    )
