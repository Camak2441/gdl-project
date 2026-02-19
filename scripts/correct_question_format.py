import json
from typing import Any, Dict

from consts import DATA_DIR

QUESTIONS_DIR = DATA_DIR / "questions_complex"


def main():
    for path in QUESTIONS_DIR.iterdir():
        if path.suffix == ".json":
            with open(path) as fp:
                data: Dict[str, Any] = json.load(fp)
            if isinstance(data, dict) and "scan_id" in data:
                data["scanId"] = data["scan_id"]
                data.pop("scan_id")
                with open(path, "w") as fp:
                    json.dump(data, fp, indent=4)


if __name__ == "__main__":
    main()
