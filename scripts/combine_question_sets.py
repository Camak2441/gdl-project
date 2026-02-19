import glob
import json
from pathlib import Path
from typing import List

from consts import DATA_DIR, Q_TYPES
from json_validator import generate_json_schema


QUESTION_SET_SCHEMA = generate_json_schema(
    {
        "questions": [
            {
                "question": str,
                "answerObjectIds": [str],
                "type": ("lit", *(Q_TYPES.keys())),
            }
        ],
        "scanId": str,
    }
)


QUESTION_INPUT_DIRS: List[Path] = [
    DATA_DIR / "questions_programmatic",
    DATA_DIR / "questions_naturalised",
    DATA_DIR / "questions_complex" / "gpt-5-mini",
]
QUESTION_OUTPUT_DIR = DATA_DIR / "questions"
HOLDOUT_SCENES = {"0a4b8ef6-a83a-21f2-8672-dce34dd0d7ca"}


def add_questions(dir: Path):
    question_files = list(glob.glob(dir.as_posix() + "/*.json"))

    print(f"Processing {len(question_files)} files in {dir}.")

    skipped_files = 0
    skipped_questions = 0
    total_questions = 0
    group_size = 100

    for idx, path in enumerate(question_files):

        if idx % group_size == 0:
            print(f"Processed {idx} / {len(question_files)}.")
            print(f"- Skipped {skipped_files} / {idx} of those files.")
            print(f"- Skipped {skipped_questions} / {total_questions} questions.")

        path = Path(path)
        if path.stem in HOLDOUT_SCENES:
            skipped_files += 1
            print(f"Skipping scene {path.stem} since in holdout set.")
            continue
        with open(path) as fp:
            questions = json.load(fp)

        if not QUESTION_SET_SCHEMA.valid(questions):
            skipped_files += 1
            print(f"Questions at {path} in an invalid format. Skipping.")
            continue

        output_path = QUESTION_OUTPUT_DIR / path.name
        if output_path.exists():
            with open(output_path) as fp:
                data = json.load(fp)
            if not QUESTION_SET_SCHEMA.valid(data):
                print(f"Questions at {output_path} in an invalid format. Skipping.")
                skipped_files += 1
                continue
            if data["scanId"] != questions["scanId"]:
                skipped_files += 1
                print(f"Scan id for {path} and {output_path} do not match. Skipping.")
                continue
        else:
            data = {"scanId": questions["scanId"], "questions": []}

        seen = set()

        for q in data["questions"]:
            seen.add(q["question"])

        skipped_count = 0
        for q in questions["questions"]:
            if q["question"] in seen:
                skipped_count += 1
                continue
            data["questions"].append(q)

        skipped_questions += skipped_count
        total_questions += len(questions["questions"])

        with open(output_path, "w") as fp:
            json.dump(data, fp)

    print(f"Processed {len(question_files)} / {len(question_files)}.")
    print(f"- Skipped {skipped_files} / {len(question_files)} of those files.")
    print(f"- Skipped {skipped_questions} / {total_questions} questions.")


def main():
    QUESTION_OUTPUT_DIR.mkdir(exist_ok=True)
    for dir in QUESTION_INPUT_DIRS:
        add_questions(dir)


if __name__ == "__main__":
    main()
