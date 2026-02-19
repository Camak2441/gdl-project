import json
import logging
import math
from pathlib import Path
import re
import random
from datetime import datetime
import sys


from consts import DATA_DIR, OUTPUT_DIR, SCRIPT_DIR
from llm_clients import create_client
from json_validator import generate_json_schema
from ssg import load_scene_graph
from utils import (
    index_by_key,
    intersect_dict,
    lock_file,
    multi_get,
    save_prompt_files,
    unlock_file,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model to use for question generation
MODEL = "gpt-oss:20b"

# Prompt settings
NUM_FORMAT_EXAMPLES = 2
NUM_EXAMPLES = 8
MAX_NUM_GEN_QS = 10
TARGET_NUM_QS = 30

# Paths
SCENE_GRAPHS_DIR = DATA_DIR / "scene_graphs"
QUESTIONS_DIR = DATA_DIR / "questions"
EXAMPLES_PATH = SCRIPT_DIR / "example_questions.json"
PROMPT_OUTPUT_DIR = OUTPUT_DIR / "prompt"
LLM_OUTPUT_DIR = OUTPUT_DIR / "llm_outputs"

# JSON Schemas
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

# Load and validate the example questions
EXAMPLE_QUESTIONS = {}

if __name__ == "__main__":
    EXAMPLES_SCHEMA = generate_json_schema(
        ("dict", {"description": str, "examples": [str]})
    )
    with EXAMPLES_PATH.open() as fp:
        EXAMPLE_QUESTIONS = json.load(fp)

    EXAMPLES_SCHEMA.assert_valid(EXAMPLE_QUESTIONS)

# Prompts
QUESTION_GENERATION_PROMPT = """You are given a 3D scene graph in JSON format. The scene graph contains:
- objects: A list of objects with `id`, `label`, `affordances`, and `attributes`
- relationships: A list of relationships between objects `[from_id, to_id, relationship_label]`
  Relationships should be read as `from_id` is/has `relationship_label` `to_id`
  For example:
  - read ["5", "2", "to the right of"] as "5" is to the right of "2"
  - read ["10", "12", "the same symmetry as"] as "10" has the same symmetry as "12"

Your task is to generate less than {q_num} diverse questions about this scene that can be answered using the scene graph.

Generate questions of the {q_type} type. {q_description}

# Requirements:
- Questions must be answerable using ONLY the information in the scene graph
- Questions should reference actual objects present in the scene
- Vary the difficulty and complexity of questions
- Each question should have a clear answer based on the scene graph

# Output format:
Return a JSON object which is a list of question objects.
Each question object should have:
- "question": The question text
- "answerObjectIds": List of the single answer object ID (as strings)

# Example questions:
{q_examples}

# Example output:
```json
[
{formatted_q_examples}
]
```

**IMPORTANT**: Ensure you return valid JSON.
"""


EXAMPLE_TEMPLATE = """  {{
    "question": "{q}",
    "answerObjectIds": {answer_ids}
  }}"""


def format_examples(examples):
    return ",\n".join(
        (
            EXAMPLE_TEMPLATE.format(
                q=example,
                answer_ids='["' + str(random.randint(1, 100)) + '"]',
            )
            for example in examples
        )
    )


def extract_json_from_response(text):
    try:
        result = json.loads(text)
        match result:
            case dict():
                return result
            case list():
                return {"questions": result}
    except json.JSONDecodeError:
        pass

    match = re.search(r"```json.*```", text, re.DOTALL)
    if match:
        candidate = match.group(0)[7:-3]
        try:
            result = json.loads(candidate)
            match result:
                case dict():
                    return result
                case list():
                    return {"questions": result}
        except json.JSONDecodeError:
            pass

    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            return {"questions": json.loads(candidate)}
        except json.JSONDecodeError:
            pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return None


def same_prop_as(objs, relation):
    if (
        relation[2][:9].startswith("the same ")
        and relation[2][-3:].endswith(" as")
        and len(relation[2]) > 11
    ):
        property = relation[2][9:-3]
        if property in [
            "symmetry",
            "color",
            "material",
            "texture",
            "shape",
            "state",
        ]:
            obj0 = index_by_key(objs, "id", str(relation[0]))
            obj1 = index_by_key(objs, "id", str(relation[1]))
            return multi_get(obj0, ["attributes", property], None) == multi_get(
                obj1, ["attributes", property], None
            )
        if property == "object type":
            obj0 = index_by_key(objs, "id", str(relation[0]))
            obj1 = index_by_key(objs, "id", str(relation[1]))
            return obj0["label"] == obj1["label"]

    return False


def preprocess_scene_graph(sg_json):
    processed = {"objects": [], "relationships": []}

    for obj in sg_json.get("objects", []):
        processed_obj = intersect_dict(
            {"id", "label", "affordances", "attributes"}, obj
        )
        assert "id" in processed_obj and "label" in processed_obj
        processed["objects"].append(processed_obj)

    for rel in sg_json.get("relationships", []):
        if len(rel) >= 4:
            # [from_id, to_id, rel_id, rel_label] -> [from_id, to_id, rel_label]
            processed["relationships"].append([rel[0], rel[1], rel[3]])

    processed["relationships"] = list(
        filter(
            lambda rel: not (same_prop_as(processed["objects"], rel)),
            processed["relationships"],
        )
    )

    return processed


def build_messages(scene_graph, q_type):
    processed_sg = preprocess_scene_graph(scene_graph)

    q_type_info = EXAMPLE_QUESTIONS[q_type]
    q_type_description = q_type_info["description"]
    q_type_examples = q_type_info["examples"]
    formatted_examples = format_examples(q_type_examples[:NUM_FORMAT_EXAMPLES])
    other_examples = "\n".join(
        map(
            lambda s: "- " + s,
            q_type_examples[NUM_FORMAT_EXAMPLES : NUM_FORMAT_EXAMPLES + NUM_EXAMPLES],
        )
    )

    prompt = QUESTION_GENERATION_PROMPT.format(
        q_num=MAX_NUM_GEN_QS,
        q_type=q_type,
        q_description=q_type_description,
        q_examples=other_examples,
        formatted_q_examples=formatted_examples,
    )
    request_content = json.dumps({"scene_graph": processed_sg}, indent=2)

    return [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": f"Please generate {q_type} questions for the following scene graph: \n```json\n"
            + request_content
            + "\n```\n\n"
            + "Please remember to output the questions in the required format similar to the following example: \n"
            + f"""
# Example output:
```json
[
{formatted_examples}
]
```
""",
        },
    ]


def confirm(prompt):
    answer = input(f"{prompt} [y/N] ").strip().lower()
    if answer != "y":
        logger.info("Aborted.")
        sys.exit(0)


def generate_questions_for_scene(client, scene_graph, q_type, scan_id):
    messages = build_messages(scene_graph, q_type)

    save_prompt_files(messages, PROMPT_OUTPUT_DIR)

    try:
        response = client.create_completion(messages)
        llm_answer = response.choices[0].message.content
        save_response(scan_id, llm_answer)
        result = extract_json_from_response(llm_answer)

        if result and QUESTION_SCHEMA.valid(result):
            for question in result["questions"]:
                question["type"] = q_type
            return result
        else:
            logger.warning("Failed to parse LLM response as questions JSON")
            return None

    except Exception as e:
        logger.error(f"Error generating questions: {e}")
        return None


def save_questions(questions_data, output_path: Path):
    lock_file(output_path)
    questions = {"scan_id": questions_data["scan_id"], "questions": []}
    if output_path.exists():
        with open(output_path, "r") as fp:
            questions = json.load(fp)
        assert questions["scan_id"] == questions_data["scan_id"]
    questions["questions"] += questions_data["questions"]
    with open(output_path, "w") as fp:
        json.dump(questions, fp, indent=4)
    unlock_file(output_path)


def save_response(scan_id, llm_answer):
    file_path = LLM_OUTPUT_DIR / (
        scan_id + datetime.now().strftime("%Y%m%dT%H%M%S") + ".txt"
    )
    with open(file_path, "w") as fp:
        fp.write(llm_answer)


def main(q_type):
    client = create_client(
        model=MODEL, max_completion_tokens=65536, temperature=0.2, timeout=300
    )

    QUESTIONS_DIR.mkdir(parents=True, exist_ok=True)

    scene_graph_files = sorted(SCENE_GRAPHS_DIR.glob("*.json"))
    total_files = len(scene_graph_files)

    num_calls_per_scene = math.ceil(TARGET_NUM_QS / MAX_NUM_GEN_QS)
    logger.info(f"Found {total_files} scene graphs")
    logger.info(f"Generating questions of type {q_type}")
    confirm(
        f"About to make up to {total_files * num_calls_per_scene} API calls "
        f"({num_calls_per_scene} per scene × {total_files} scenes, "
        f"targeting {TARGET_NUM_QS} questions each). Proceed?"
    )

    successful = 0
    failed = 0

    for idx, sg_file in enumerate(scene_graph_files, 1):
        scan_id = sg_file.stem
        output_file = QUESTIONS_DIR / f"{scan_id}.json"

        logger.info(f"[{idx}/{total_files}] Processing {scan_id}...")

        scene_graph = load_scene_graph(sg_file)

        num_calls = num_calls_per_scene
        total_generated = 0
        any_success = False

        for call_idx in range(1, num_calls + 1):
            questions = generate_questions_for_scene(
                client, scene_graph, q_type, scan_id
            )
            if questions and "questions" in questions:
                questions_data = {
                    "scan_id": scan_id,
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

    logger.info("=" * 50)
    logger.info(f"Completed: {successful} successful, {failed} failed")


def test_prompt(q_type):
    sg_file = next(SCENE_GRAPHS_DIR.glob("*.json"))
    scene_graph = load_scene_graph(sg_file)
    messages = build_messages(scene_graph, q_type)
    save_prompt_files(messages, PROMPT_OUTPUT_DIR)


if __name__ == "__main__":
    if "--test-prompt" in sys.argv:
        args = [a for a in sys.argv[1:] if a != "--test-prompt"]
        q_type = args[0] if args else next(iter(EXAMPLE_QUESTIONS))
        test_prompt(q_type)
    else:
        for q_type in sys.argv[1:]:
            main(q_type)
