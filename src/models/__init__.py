import json
from pathlib import Path
import re
from typing import Any, Dict

from models.query_gat import QueryGAT
from models.query_in_gat import QueryInGAT
from utils import check_keys_are_in_order, quote_json


MODEL_PREFIXES = {
    "QueryGat": QueryGAT,
    "QueryInGat": QueryInGAT,
}


def _load_argstr(argstr: str):
    return json.loads(quote_json("{" + argstr[1:-1].replace("=", ":") + "}"))


def canonical_model_name(model_name: str):
    for prefix in MODEL_PREFIXES:
        if model_name.startswith(prefix):
            argstr = model_name[len(prefix) :]
            if not (argstr.startswith("(") and argstr.endswith(")")):
                return
            kwargs = _load_argstr(argstr)
            return (
                prefix
                + "("
                + "".join(
                    json.dumps(kwargs, separators=(",", "="), sort_keys=True).split()
                ).replace('"', "")[1:-1]
                + ")"
            )


def load_model(model_name: str):
    for prefix in MODEL_PREFIXES:
        if model_name.startswith(prefix):
            argstr = model_name[len(prefix) :]
            if not (argstr.startswith("(") and argstr.endswith(")")):
                return
            kwargs = _load_argstr(argstr)
            if not check_keys_are_in_order(kwargs):
                return
            if "tag" in kwargs:
                kwargs.pop("tag")
            return MODEL_PREFIXES[prefix](**kwargs)


def get_most_epochs_weight_file(model_dir: Path):
    most_epochs = None
    for model_path in model_dir.iterdir():
        if re.match(r"0|[1-9][0-9]*", model_path.stem) and model_path.suffix == ".pth":
            file_epochs = int(model_path.stem)
            if most_epochs is None or file_epochs > most_epochs:
                most_epochs = file_epochs
    return most_epochs
