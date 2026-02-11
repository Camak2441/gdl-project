import json
from pathlib import Path
from json_validator import *


QUESTION_SCHEMA = generate_json_schema(
    {
        "questions": [
            {
                "question": str,
                "type": JSONLit(["semantic", "spatial", "support"]),
                "expectedObjectIds": [str],
            }
        ]
    }
)


def main(dir):
    for filepath in Path(dir).glob("*.json"):
        valid = False
        with open(filepath, mode="r") as fp:
            data = json.load(fp)
            valid = QUESTION_SCHEMA.valid(data)
        if not valid:
            val = input(f"Confirm delete the following json: {data}")
            if val.lower() in ["y", "yes"]:
                filepath.unlink()


if __name__ == "__main__":
    import sys

    assert len(sys.argv) == 2, "Expected exactly one argument"
    main(sys.argv[1])
