import json
import logging
import math
from pathlib import Path
import sys

from consts import DATA_DIR, OUTPUT_DIR
from llm_clients import create_client
from generate_qs import extract_json_from_response

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MODEL = "gpt-5-mini"
NUM_REWRITES = 3
PROPORTION = 0.1
NUM_PASSES = math.ceil(1 / PROPORTION)

INPUT_QUESTIONS_DIR = DATA_DIR / "questions_programmatic"
OUTPUT_QUESTIONS_DIR = DATA_DIR / "questions_naturalised"
PROMPT_OUTPUT_DIR = OUTPUT_DIR / "prompt_naturalise"


def build_system_prompt(examples, num_rewrites):
    example_lines = "\n".join(
        f"  Original: {orig}\n  Rewritten: {rewrite}" for orig, rewrite in examples
    )

    return f"""\
You are a question rewriter. You will be given a question about a 3D scene. \
Your task is to produce {num_rewrites} distinct rephrases of the question so that each sounds \
more natural, colloquial, and conversational, as if asked by a real person. \
You also want to maximise variety across the rewrites.

# Requirements:

- Each rephrased question MUST ask about exactly the same properties/relationships \
as the original so that the objects which answer the question remain the same.
- Do NOT add or remove constraints that would change which objects answer the question.
- Vary sentence structure, word choice, tense, and phrasing style across the {num_rewrites} rewrites.
- Keep each question concise and natural, without stiff or formal wording.
- Make each question sound like something someone might naturally ask.

Return a JSON object with a single key "questions" containing a list of {num_rewrites} rewritten question strings.

# Examples of good rewrites (single rewrite shown per original):

{example_lines}

# Example format:

```json
{{"questions": [{", ".join(f'"[rewrite {i}]"' for i in range(1, num_rewrites + 1))}]}}
```
"""


# Hand-written rewrites paired with sampled programmatic questions
REWRITE_PAIRS = [
    ("What is beige and clean?", "What is both beige and clean?"),
    (
        "What object is not small and to the left of a doorframe?",
        "Can you find something to the left of the doorframe that isn't small?",
    ),
    ("Which object is used for lighting?", "What in this scene provides lighting?"),
    ("Which objects are the cleanest?", "What are the cleanest things here?"),
    (
        "Which object is behind the table and not to the right of the table?",
        "What's behind the table but not to its right?",
    ),
    ("What is standing on a floor?", "What's on the floor?"),
    ("What object is behind the soap dish?", "What can you find behind the soap dish?"),
    ("What object is used for cleaning?", "What can I clean with?"),
]


def confirm(prompt):
    answer = input(f"{prompt} [y/N] ").strip().lower()
    if answer != "y":
        logger.info("Aborted.")
        sys.exit(0)


def naturalise_question(client, question_obj, system_prompt):
    """Returns a list of rewritten question strings, or None on failure."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question_obj["question"]},
    ]

    try:
        response = client.create_completion(messages)
        answer = response.choices[0].message.content
        result = extract_json_from_response(answer)

        if result and "questions" in result and isinstance(result["questions"], list):
            return result["questions"]
        else:
            logger.warning(f"Failed to parse response: {answer}")
            return None

    except Exception as e:
        logger.error(f"Error naturalising question: {e}")
        return None


def load_progress(progress_path: Path) -> dict[str, set[int]]:
    """Returns a dict mapping scan_id -> set of already-processed question indices."""
    if progress_path.exists():
        with open(progress_path) as fp:
            data = json.load(fp)
        return {k: set(v) for k, v in data.items()}
    return {}


def save_progress(progress_path: Path, progress: dict[str, set[int]]):
    with open(progress_path, "w") as fp:
        json.dump({k: sorted(v) for k, v in progress.items()}, fp)


def save_naturalised(output_path: Path, base_data: dict, new_questions: list):
    """Append new_questions to the output file, creating it if it doesn't exist."""
    if output_path.exists():
        with open(output_path) as fp:
            existing = json.load(fp)
        existing["questions"] += new_questions
        with open(output_path, "w") as fp:
            json.dump(existing, fp, indent=4)
    else:
        with open(output_path, "w") as fp:
            json.dump({**base_data, "questions": new_questions}, fp, indent=4)


def main():
    client = create_client(
        model=MODEL,
        max_completion_tokens=8192,
        temperature=0.7,
        timeout=60,
        verbosity="low",
    )

    OUTPUT_QUESTIONS_DIR.mkdir(parents=True, exist_ok=True)

    progress_path = OUTPUT_QUESTIONS_DIR / ".progress.json"
    progress = load_progress(progress_path)

    system_prompt = build_system_prompt(REWRITE_PAIRS, NUM_REWRITES)

    question_files = sorted(INPUT_QUESTIONS_DIR.glob("*.json"))
    logger.info(f"Found {len(question_files)} question files")
    logger.info(
        f"Running {NUM_PASSES} passes, naturalising ~{PROPORTION:.0%} per pass, "
        f"{NUM_REWRITES} rewrites each"
    )
    confirm(
        f"About to run {NUM_PASSES} passes over {len(question_files)} files "
        f"(~{PROPORTION:.0%} of questions per pass, {NUM_REWRITES} rewrites each). Proceed?"
    )

    for pass_idx in range(1, NUM_PASSES + 1):
        logger.info(f"--- Pass {pass_idx}/{NUM_PASSES} ---")

        for idx, qf in enumerate(question_files, 1):
            scan_id = qf.stem

            with open(qf) as fp:
                data = json.load(fp)

            questions = data["questions"]
            done_indices = progress.get(scan_id, set())
            remaining = [i for i in range(len(questions)) if i not in done_indices]

            if not remaining:
                logger.info(
                    f"[{idx}/{len(question_files)}] {scan_id}: all questions done, skipping"
                )
                continue

            batch_size = max(1, math.ceil(PROPORTION * len(questions)))
            batch_indices = remaining[:batch_size]

            output_path = OUTPUT_QUESTIONS_DIR / qf.name
            new_questions = []

            for i in batch_indices:
                q = questions[i]
                new_questions.append(q)
                rewrites = naturalise_question(client, q, system_prompt)
                if rewrites:
                    for rewrite in rewrites:
                        new_questions.append({**q, "question": rewrite})
                else:
                    logger.warning(f"No rewrites generated for: {q['question']}")

            save_naturalised(output_path, data, new_questions)
            done_indices.update(batch_indices)
            progress[scan_id] = done_indices
            save_progress(progress_path, progress)

            logger.info(
                f"[{idx}/{len(question_files)}] {scan_id}: "
                f"processed {len(batch_indices)} questions "
                f"({len(done_indices)}/{len(questions)} total done)"
            )

    logger.info("Done")
    progress_path.unlink(missing_ok=True)


def test_prompt():
    from utils import save_prompt_files

    system_prompt = build_system_prompt(REWRITE_PAIRS, NUM_REWRITES)
    qf = next(INPUT_QUESTIONS_DIR.glob("*.json"))
    with open(qf) as fp:
        data = json.load(fp)
    sample_q = data["questions"][0]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": sample_q["question"]},
    ]
    save_prompt_files(messages, PROMPT_OUTPUT_DIR)


if __name__ == "__main__":

    if "--test-prompt" in sys.argv:
        test_prompt()
    else:
        main()
