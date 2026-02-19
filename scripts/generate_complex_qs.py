import json
import logging
import math
from pathlib import Path
import random
from datetime import datetime
import sys

from consts import DATA_DIR, OUTPUT_DIR
from llm_clients import create_client
from json_validator import generate_json_schema
from ssg import load_scene_graph
from utils import (
    lock_file,
    save_prompt_files,
    unlock_file,
)
from generate_qs import extract_json_from_response, preprocess_scene_graph

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL = "gpt-5-mini"

MAX_NUM_GEN_QS = 20
TARGET_NUM_QS = 20

# Paths
SCENE_GRAPHS_DIR = DATA_DIR / "scene_graphs"
QUESTIONS_DIR = DATA_DIR / "questions_complex" / MODEL.replace(":", "-")
PROMPT_OUTPUT_DIR = OUTPUT_DIR / "prompt_complex"
LLM_OUTPUT_DIR = OUTPUT_DIR / "llm_outputs"

# JSON Schema
QUESTION_SCHEMA = generate_json_schema(
    {
        "questions": [
            {
                "question": str,
                "answerObjectIds": [str],
            }
        ]
    }
)

# Example complex questions with hand-written rewrites that demonstrate
# the kind of reasoning we want: practical/functional/inferential rather
# than direct attribute or spatial lookups.
EXAMPLE_QUESTIONS = [
    "In order to get to the door from the sofa, what might I need to be careful of?",
    "What's most likely to trip someone?",
    "What should I sit on?",
    "Which should I grab to clean the floor?",
    "What might block my path if I walked from the sofa to the window?",
    "If the lights went out, which objects would be hardest to avoid?",
    "What would a child most likely try to climb on?",
    "What could I use to store something small?",
    "I just spilled some water. What might be damaged?",
    "What could fall if someone bumped the shelf?",
    "I want to reach something high up, what should I use?",
    "I want to sit down. Where should I go?",
]


EXAMPLE_TEMPLATE = """  {{
    "question": "{q}",
    "answerObjectIds": {answer_ids}
  }}"""


def format_examples(examples):
    return ",\n".join(
        EXAMPLE_TEMPLATE.format(
            q=example,
            answer_ids='["' + str(random.randint(1, 100)) + '"]',
        )
        for example in examples
    )


QUESTION_GENERATION_PROMPT = """\
You are given a 3D scene graph in JSON format. The scene graph contains:
- objects: A list of objects with `id`, `label`, `affordances`, and `attributes`
- relationships: A list of relationships between objects `[from_id, to_id, relationship_label]`
  Relationships should be read as `from_id` is/has `relationship_label` `to_id`
  For example:
  - read ["5", "2", "to the right of"] as "5" is to the right of "2"
  - read ["10", "12", "the same symmetry as"] as "10" has the same symmetry as "12"

Your task is to generate up to {q_num} diverse **complex reasoning** questions about this scene that can be answered using the scene graph. \
The questions should be in a colloquial style, as though spoken in daily conversation. Avoid stiff wording and technical terms. \
Questions should use a variety of contexts and tenses: write some questions as hypotheticals, and write others as responses to events which happened.

Complex reasoning questions require inferential or practical thinking beyond simple attribute lookups or spatial queries. \
They ask about but are not limited to:
- **Functional reasoning**: What objects could serve a purpose (sitting, cleaning, storing, reaching)?
- **Navigation/safety**: What objects might obstruct a path, cause someone to trip, or be hard to avoid?
- **Cause and effect**: What might happen if an object fell, spilled, or was moved?
- **Practical interaction**: What would someone interact with first, grab, or use in a scenario?
- **Risk assessment**: What objects are fragile, unstable, or potentially hazardous?

# Requirements:

- Questions must be answerable using ONLY the information in the scene graph (object labels, attributes, affordances, and relationships)
- Questions should reference the scene naturally without mentioning specific object IDs
- Return all and any objects which answer the question. 
- Vary the style, form, tense, context, and reasoning across questions
- Do NOT write simple attribute questions like "What color is X?" or direct spatial questions like "What is to the left of Y?"
- Try to write as colloquially and naturally as possible.
- Try and make the questions concise. 
- Do not include unnecessary detail in the question.
- Write questions which are natural to ask.

# Example questions:

{q_examples}

# Output format:

Return a JSON object with a single key "questions" which is a list of question objects.
Each question object should have:
- "question": The question text
- "answerObjectIds": List of answer object IDs (as strings)

# Example output:

```json
{{
    "questions": [
        {{
            "question": "[Question text]",
            "answerObjectIds": ["[objectId 1]", "[objectId 2]", ...]
        }},
        ...
    ]
}}
```

**IMPORTANT**: Ensure you return valid JSON.
"""


def build_messages(scene_graph):
    processed_sg = preprocess_scene_graph(scene_graph)

    other_examples = "\n".join(f"- {q}" for q in EXAMPLE_QUESTIONS)

    prompt = QUESTION_GENERATION_PROMPT.format(
        q_num=MAX_NUM_GEN_QS,
        q_examples=other_examples,
    )
    request_content = json.dumps({"scene_graph": processed_sg}, indent=2)

    return [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": "Please generate complex reasoning questions for the following scene graph: \n```json\n"
            + request_content
            + "\n```\n\n"
            + "Please remember to output a json object.",
        },
    ]


def confirm(prompt):
    answer = input(f"{prompt} [y/N] ").strip().lower()
    if answer != "y":
        logger.info("Aborted.")
        sys.exit(0)


def generate_questions_for_scene(client, scene_graph, scan_id):
    messages = build_messages(scene_graph)

    save_prompt_files(messages, PROMPT_OUTPUT_DIR)

    try:
        response = client.create_completion(messages)
        llm_answer = response.choices[0].message.content
        resonse_path = save_response(scan_id, llm_answer)
        result = extract_json_from_response(llm_answer)

        if result and QUESTION_SCHEMA.valid(result):
            for question in result["questions"]:
                question["type"] = "complex"
            return result
        else:
            logger.warning(
                f"Failed to parse LLM response as questions JSON, saved response in {resonse_path}"
            )
            return None

    except Exception as e:
        logger.error(f"Error generating questions: {e}")
        return None


def save_questions(questions_data, output_path: Path):
    lock_file(output_path)
    questions = {"scanId": questions_data["scanId"], "questions": []}
    if output_path.exists():
        with open(output_path, "r") as fp:
            questions = json.load(fp)
        assert questions["scanId"] == questions_data["scanId"]
    questions["questions"] += questions_data["questions"]
    with open(output_path, "w") as fp:
        json.dump(questions, fp, indent=4)
    unlock_file(output_path)


def save_response(scan_id, llm_answer):
    LLM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    file_path = LLM_OUTPUT_DIR / (
        scan_id + "_complex_" + datetime.now().strftime("%Y%m%dT%H%M%S") + ".txt"
    )
    with open(file_path, "w") as fp:
        fp.write(llm_answer)
    return file_path


def load_attempted(progress_path: Path):
    if progress_path.exists():
        with open(progress_path, "r") as fp:
            return set(json.load(fp))
    return set()


def save_attempted(progress_path: Path, attempted: set):
    with open(progress_path, "w") as fp:
        json.dump(sorted(attempted), fp)


def main():
    client = create_client(
        model=MODEL,
        max_completion_tokens=65536,
        temperature=0.2,
        timeout=300,
        verbosity="low",
    )

    QUESTIONS_DIR.mkdir(parents=True, exist_ok=True)

    progress_path = QUESTIONS_DIR / ".progress.json"
    attempted = load_attempted(progress_path)

    scene_graph_files = sorted(SCENE_GRAPHS_DIR.glob("*.json"))
    total_files = len(scene_graph_files)

    num_calls_per_scene = math.ceil(TARGET_NUM_QS / MAX_NUM_GEN_QS)
    remaining = total_files - len(attempted)
    logger.info(f"Found {total_files} scene graphs, {len(attempted)} already attempted")
    logger.info("Generating complex reasoning questions")
    confirm(
        f"About to make up to {remaining * num_calls_per_scene} API calls "
        f"({num_calls_per_scene} per scene × {remaining} remaining scenes, "
        f"targeting {TARGET_NUM_QS} questions each). Proceed?"
    )

    successful = 0
    failed = 0

    for idx, sg_file in enumerate(scene_graph_files, 1):
        scan_id = sg_file.stem

        if scan_id in attempted:
            logger.info(f"[{idx}/{total_files}] Skipping {scan_id} (already attempted)")
            continue

        logger.info(f"[{idx}/{total_files}] Processing {scan_id}...")

        scene_graph = load_scene_graph(sg_file)
        output_file = QUESTIONS_DIR / f"{scan_id}.json"

        num_calls = num_calls_per_scene
        total_generated = 0
        any_success = False

        for call_idx in range(1, num_calls + 1):
            questions = generate_questions_for_scene(client, scene_graph, scan_id)
            if questions and "questions" in questions:
                questions_data = {
                    "scanId": scan_id,
                    "questions": questions["questions"],
                }
                save_questions(questions_data, output_file)
                total_generated += len(questions["questions"])
                any_success = True
                logger.info(
                    f"  Call {call_idx}/{num_calls}: got {len(questions['questions'])} questions"
                    f" ({total_generated} total)"
                )
            else:
                logger.warning(f"  Call {call_idx}/{num_calls}: failed")

        if any_success:
            logger.info(f"Generated {total_generated} questions for {scan_id}")
            successful += 1
        else:
            logger.warning(f"Failed to generate any questions for {scan_id}")
            failed += 1

        attempted.add(scan_id)
        save_attempted(progress_path, attempted)

    logger.info("=" * 50)
    logger.info(f"Completed: {successful} successful, {failed} failed")

    progress_path.unlink(missing_ok=True)


def test_prompt():
    sg_file = next(SCENE_GRAPHS_DIR.glob("*.json"))
    scene_graph = load_scene_graph(sg_file)
    messages = build_messages(scene_graph)
    save_prompt_files(messages, PROMPT_OUTPUT_DIR)


if __name__ == "__main__":

    if "--test-prompt" in sys.argv:
        test_prompt()
    else:
        main()
