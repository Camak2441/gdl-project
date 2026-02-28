import os
from pathlib import Path
import time
from typing import Any, Dict, Set
import re


def filter_dict(pred, d):
    return {key: d[key] for key in d if pred(key)}


def index_by_key(l, key, i):
    for e in l:
        if key in e and e[key] == i:
            return e
    raise IndexError(f"Unable to find index {i} under key {key}")


def intersect_dict(keyset: Set[Any], d: Dict[Any, Any]):
    return {key: d[key] for key in d if key in keyset}


def check_keys_are_in_order(kwargs: Dict[str, Any]):
    prev = ""
    for key in kwargs:
        if prev > key:
            return False
    return True


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


def save_prompt_files(messages, output_dir):
    """Save messages to separate files: system.txt for the system prompt,
    messages.txt for the remaining chat messages."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    system_messages = [m for m in messages if m["role"] == "system"]
    other_messages = [m for m in messages if m["role"] != "system"]

    with open(output_dir / "system.txt", "w") as fp:
        save_completion_prompt(system_messages, fp)

    with open(output_dir / "messages.txt", "w") as fp:
        save_completion_prompt(other_messages, fp)


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


def quote_json_val(s):
    val = s.strip()
    if re.fullmatch(r"0|[1-9][0-9]*|(0|[1-9][0-9]*)\.[0-9]*|true|false", val):
        return val
    if val.startswith('"') and val.endswith('"'):
        return val
    if val == "":
        return val
    return '"' + val + '"'


def quote_json_key(s):
    key = s.strip()
    if key.startswith('"') and key.endswith('"'):
        return key
    return '"' + key + '"'


def quote_json(s):
    bracket_stack = []
    segs = []

    index = 0
    seg = []

    def get_seg():
        nonlocal seg
        seg_s = "".join(seg)
        seg.clear()
        return seg_s

    while index < len(s):
        c = s[index]
        index += 1
        if len(bracket_stack) > 0 and bracket_stack[-1] == '"':
            seg.append(c)
            match c:
                case '"':
                    if not (
                        len(seg) > 0
                        and seg[-1] == "\\"
                        and (len(seg) < 2 or seg[-2] != "\\")
                    ):
                        bracket_stack.pop()
                        seg_s = get_seg()
                        segs.append(seg_s)
        else:
            match c:
                case '"':
                    seg_s = get_seg()
                    segs.append(seg_s)
                    seg.append('"')
                    bracket_stack.append('"')
                case "{":
                    seg_s = get_seg()
                    assert (
                        seg_s.strip() == ""
                    ), f"Unexpected value before {index} in {s}"
                    segs.append("{")
                    bracket_stack.append("{")
                case "}":
                    seg_s = get_seg()
                    assert (
                        len(bracket_stack) > 0
                    ), f"Unmatched bracket at {index} in {s}"
                    if bracket_stack[-1] == ":":
                        quoted = quote_json_val(seg_s)
                        assert (
                            quoted is not None
                        ), f"Missing value before {index} in {s}"
                        segs.append(quoted + "}")
                        bracket_stack.pop()
                    else:
                        assert (
                            seg_s.strip() == ""
                        ), f"Unexpected value before {index} in {s}"
                    assert (
                        bracket_stack[-1] == "{"
                    ), f"Unmatched bracket at {index} in {s}"
                    bracket_stack.pop()
                case "[":
                    seg_s = get_seg()
                    assert (
                        seg_s.strip() == ""
                    ), f"Unexpected value before {index} in {s}"
                    segs.append("[")
                    bracket_stack.append("[")
                case "]":
                    seg_s = get_seg()
                    assert (
                        len(bracket_stack) > 0
                    ), f"Unmatched bracket at {index} in {s}"
                    if bracket_stack[-1] == ",":
                        quoted = quote_json_val(seg_s)
                        assert (
                            quoted is not None
                        ), f"Missing value before {index} in {s}"
                        segs.append(quoted + "]")
                        bracket_stack.pop()
                    else:
                        quoted = quote_json_val(seg_s)
                        if quoted is not None:
                            segs.append(quoted + "]")

                    assert (
                        bracket_stack[-1] == "["
                    ), f"Unmatched bracket at {index} in {s}"
                    bracket_stack.pop()
                case ":":
                    seg_s = get_seg()
                    quoted = quote_json_key(seg_s)
                    assert quoted is not None, f"Missing value before {index} in {s}"
                    assert len(bracket_stack) > 0, f"Unmatched colon at {index} in {s}"
                    segs.append(quoted + ":")
                    if bracket_stack[-1] == ",":
                        bracket_stack.pop()
                    assert (
                        bracket_stack[-1] in "{"
                    ), f"Unmatched colon at {index} in {s}"
                    bracket_stack.append(":")
                case ",":
                    seg_s = get_seg()
                    quoted = quote_json_val(seg_s)
                    assert quoted is not None, f"Missing value before {index} in {s}"
                    segs.append(quoted + ",")
                    assert len(bracket_stack) > 0, f"Unexpected comma at {index} in {s}"
                    assert (
                        bracket_stack[-1] in "[:,"
                    ), f"Unexpected comma at {index} in {s}"
                    if bracket_stack[-1] == ":":
                        bracket_stack.pop()
                    if bracket_stack[-1] == ",":
                        bracket_stack.pop()
                        assert (
                            bracket_stack[-1] in "["
                        ), f"Unexpected comma at {index} in {s}"
                    bracket_stack.append(",")
                case _:
                    seg.append(c)

    seg_s = get_seg()
    if len(segs) != 0:
        assert seg_s.strip() == "", f"Unexpected value before {index} in {s}"
    else:
        quoted = quote_json_val(seg_s)
        assert quoted is not None, f"Missing value before {index} in {s}"
        segs.append(quoted)

    return "".join(segs)
