"""
Filter a questions directory to only retain questions with exactly one answer.

Each JSON file is expected to have the structure:
    {
        "scanId": str,
        "questions": [
            {
                "question": str,
                "answerObjectIds": [str],
                "type": str
            },
            ...
        ]
    }

Files where no questions survive the filter are not written to the output.

Usage:
    python filter_single_answer_qs.py --source <src_dir> --dest <dst_dir>
"""

import argparse
import json
from pathlib import Path

from consts import DATA_DIR


def filter_file(path: Path, dest_dir: Path) -> tuple[int, int]:
    """Filter a single questions JSON file.

    Returns:
        (before, after) question counts.
    """
    with open(path) as fp:
        data = json.load(fp)

    questions = data.get("questions", [])
    filtered = [q for q in questions if len(q.get("answerObjectIds", [])) == 1]

    if not filtered:
        return len(questions), 0

    dest_dir.mkdir(parents=True, exist_ok=True)
    out = {**data, "questions": filtered}
    with open(dest_dir / path.name, "w") as fp:
        json.dump(out, fp, indent=2)

    return len(questions), len(filtered)


def main():
    parser = argparse.ArgumentParser(
        description="Filter questions to those with exactly one answer."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DATA_DIR / "questions",
        help="Source questions directory (default: DATA_DIR/questions)",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DATA_DIR / "questions_single_answer",
        help="Output questions directory (default: DATA_DIR/questions_single_answer)",
    )
    args = parser.parse_args()

    paths = sorted(args.source.glob("*.json"))
    if not paths:
        print(f"No JSON files found in {args.source}")
        return

    total_before = total_after = files_written = 0

    for path in paths:
        before, after = filter_file(path, args.dest)
        total_before += before
        total_after += after
        if after > 0:
            files_written += 1

    print(f"Source:  {args.source}")
    print(f"Output:  {args.dest}")
    print(f"Files written: {files_written}/{len(paths)}")
    print(f"Questions kept: {total_after}/{total_before} "
          f"({100 * total_after / total_before:.1f}% retained)" if total_before else "")


if __name__ == "__main__":
    main()
