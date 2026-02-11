import os
from pathlib import Path
import time
from typing import Any, Dict, Set


def filter_dict(pred, d):
    return {key: d[key] for key in d if pred(key)}


def index_by_key(l, key, i):
    for e in l:
        if key in e and e[key] == i:
            return e
    raise IndexError(f"Unable to find index {i} under key {key}")


def intersect_dict(keyset: Set[Any], d: Dict[Any, Any]):
    return {key: d[key] for key in d if key in keyset}


class NotInjectiveError(Exception):
    def __init__(self, d):
        super().__init__("Tried to find inverse of non-injective dictionary")
        self.d = d


def invert_dict(d):
    inverse = {}
    for key in d:
        if d[key] not in inverse:
            inverse[d[key]] = key
        else:
            raise NotInjectiveError(d)
    return inverse


def add_dict(d1, d2):
    dres = d1.copy()
    for k in d2:
        if k not in d1:
            dres[k] = d2[k]
    return dres


def save_completion_prompt(messages, fp):
    for message in messages:
        fp.write(message["role"] + "\n")
        fp.write("-" * len(message["role"]) + "\n")
        fp.write(message["content"] + "\n")


def lock_file(path):
    lockpath = Path(f"{path}.lock")
    wait_time = 0.0001
    max_wait = 2
    while True:
        while os.path.isfile(lockpath):
            time.sleep(wait_time)
            wait_time *= 1.1
            wait_time = min(wait_time, max_wait)
        try:
            open(lockpath, "x").close()
            break
        except FileExistsError as e:
            continue


def unlock_file(path):
    lockpath = Path(f"{path}.lock")
    if os.path.isfile(lockpath):
        os.remove(lockpath)


def multi_get(d, indices, default=None):
    curr = d
    for index in indices:
        if isinstance(curr, dict):
            if index in curr:
                curr = curr[index]
                continue
        return default
    return curr
