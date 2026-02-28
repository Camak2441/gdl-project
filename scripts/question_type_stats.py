import json
from collections import Counter
from consts import DATA_DIR

QUESTIONS_DIR = DATA_DIR / "questions"


def main():
    type_counts: Counter = Counter()

    for path in QUESTIONS_DIR.glob("*.json"):
        with open(path) as fp:
            data = json.load(fp)
        if data is None or "questions" not in data:
            continue
        for q in data["questions"]:
            qtype = q.get("type")
            if qtype is not None:
                type_counts[qtype] += 1

    total = sum(type_counts.values())
    if total == 0:
        print("No questions found.")
        return

    print(f"Total questions: {total}\n")
    print(f"{'Type':<12} {'Count':>8}  {'Proportion':>12}")
    print("-" * 36)
    for qtype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"{qtype:<12} {count:>8}  {count / total:>11.1%}")


if __name__ == "__main__":
    main()
