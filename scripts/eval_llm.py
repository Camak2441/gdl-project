"""
Ask an LLM to answer the questions in a dataset split and save the results.

Results are saved in the same format as eval_model.py:
    {
        "out":       [total_nodes] float - 1.0 if LLM selected node, else 0.0
        "y":         [total_nodes] float - ground truth labels
        "qtype":     [num_queries] int   - question type per query
        "num_nodes": [num_queries] int   - nodes per scene graph
    }

Usage:
    python scripts/llm_eval.py
    python scripts/llm_eval.py --interactive
    python scripts/llm_eval.py --splits test val
"""

import argparse
import json
import logging
import re
import time

import openai
import torch

logger = logging.getLogger(__name__)

from consts import DATA_DIR, OUTPUT_DIR
from llm_clients import create_client, models as LLM_MODELS
from utils import intersect_dict


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "gpt-4.1-nano"

EDGE_ENCODER = "all_minilm_l6v2"
NODE_ENCODER = "all_minilm_l6v2"
QUERY_ENCODER = "all_minilm_l6v2"

DATASET_DIR = (
    DATA_DIR
    / "dataset_balanced"
    / ";".join([EDGE_ENCODER, NODE_ENCODER, QUERY_ENCODER])
)

SCENE_GRAPHS_DIR = DATA_DIR / "scene_graphs"
EMBEDDED_SG_DIR = (
    DATA_DIR / "embedded_scene_graphs" / ";".join([EDGE_ENCODER, NODE_ENCODER])
)
RESULTS_OUT_DIR = OUTPUT_DIR / "results"

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are given a 3D scene graph describing a real room. The scene graph contains:
- objects: a list of objects, each with an id, label, affordances, and attributes
- relationships: a list of [from_id, to_id, relationship_label] triples
  Read as: the object with from_id is/has relationship_label relative to the object with to_id
  For example: ["5", "2", "to the right of"] means object 5 is to the right of object 2

You will be asked a question about the scene. Identify all objects whose ids answer the question.

Respond with a JSON object only, in exactly this format (no other text):
{"answerObjectIds": ["<id>", ...]}
"""


# ---------------------------------------------------------------------------
# Scene graph helpers
# ---------------------------------------------------------------------------


def preprocess_scene_graph(sg_json: dict) -> dict:
    processed = {"objects": [], "relationships": []}
    for obj in sg_json.get("objects", []):
        processed["objects"].append(
            intersect_dict({"id", "label", "affordances", "attributes"}, obj)
        )
    for rel in sg_json.get("relationships", []):
        if len(rel) >= 4:
            processed["relationships"].append([rel[0], rel[1], rel[3]])
    return processed


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------


def extract_answer_ids(text: str) -> list[str] | None:
    """Try increasingly lenient parses to extract answerObjectIds from LLM text."""
    candidates = [text]

    m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        candidates.append(m.group(1))

    m = re.search(r'\{[^{}]*"answerObjectIds"[^{}]*\}', text, re.DOTALL)
    if m:
        candidates.append(m.group(0))

    for candidate in candidates:
        try:
            result = json.loads(candidate)
            if isinstance(result, dict) and "answerObjectIds" in result:
                return [str(i) for i in result["answerObjectIds"]]
        except (json.JSONDecodeError, TypeError):
            pass
    return None


def ask_llm(client, processed_sg: dict, question: str) -> list[str]:
    """Ask the LLM which object IDs answer the question. Returns list of ID strings."""
    sg_str = json.dumps(processed_sg, indent=2)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Scene graph:\n```json\n{sg_str}\n```\n\nQuestion: {question}"
            ),
        },
    ]
    response = client.create_completion(messages)
    ids = extract_answer_ids(response.choices[0].message.content)
    return ids if ids is not None else []


# ---------------------------------------------------------------------------
# Interactive helpers
# ---------------------------------------------------------------------------


def select_interactively(options: list[str], prompt: str) -> str:
    print(f"\n{prompt}")
    for i, opt in enumerate(options):
        print(f"  [{i}] {opt}")
    while True:
        raw = input("Enter number: ").strip()
        if raw.isdigit() and 0 <= int(raw) < len(options):
            return options[int(raw)]
        print(f"Please enter a number between 0 and {len(options) - 1}.")


def get_available_datasets() -> list[str]:
    base = DATA_DIR / "dataset_balanced"
    return sorted(p.name for p in base.iterdir() if p.is_dir()) if base.exists() else []


def count_questions_in_split(split_dir) -> int:
    return sum(
        torch.load(qf, weights_only=False)["query"].shape[0]
        for qf in (split_dir / "questions").glob("*.pth")
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Ask an LLM to answer scene graph questions from a dataset split."
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Interactively select LLM model and dataset.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test"],
        metavar="SPLIT",
        help="Dataset splits to run on (default: test).",
    )
    args = parser.parse_args()

    model_name = MODEL
    dataset_dir = DATASET_DIR

    if args.interactive:
        datasets = get_available_datasets()
        if not datasets:
            print("No datasets found.")
            return
        dataset_enc = select_interactively(datasets, "Select a dataset:")
        dataset_dir = DATA_DIR / "dataset_balanced" / dataset_enc

        model_name = select_interactively(
            sorted(LLM_MODELS.keys()), "Select an LLM model:"
        )

    # Resolve splits
    available = (
        {p.name for p in dataset_dir.iterdir() if p.is_dir()}
        if dataset_dir.exists()
        else set()
    )
    splits = [s for s in args.splits if s in available]
    if not splits:
        print(f"No valid splits found in {dataset_dir}. Available: {sorted(available)}")
        return

    # Summarise calls before doing anything, accounting for existing progress
    split_counts = {s: count_questions_in_split(dataset_dir / s) for s in splits}

    out_dir = RESULTS_OUT_DIR / f"llm_{model_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    already_done = {}
    for s in splits:
        pf = out_dir / f"{s}_progress.pth"
        if pf.exists():
            p = torch.load(pf, weights_only=False)
            already_done[s] = sum(v["qtype"].shape[0] for v in p.values())
        else:
            already_done[s] = 0

    remaining = {s: split_counts[s] - already_done[s] for s in splits}
    total_remaining = sum(remaining.values())

    print(f"\nModel:    {model_name}")
    print(f"Dataset:  {dataset_dir.name}")
    for s in splits:
        done, total = already_done[s], split_counts[s]
        suffix = f" ({done} already done, {remaining[s]} remaining)" if done else ""
        print(f"  {s}: {total} questions{suffix}")
    print(f"Total LLM calls to make: {total_remaining}")

    answer = input("\nProceed? [y/N] ").strip().lower()
    if answer not in ("y", "yes"):
        print("Aborted.")
        return

    client = create_client(model=model_name, temperature=0.0, max_completion_tokens=256)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
        ],
    )
    logger.info(
        "Starting eval_llm: model=%s dataset=%s splits=%s",
        model_name,
        dataset_dir.name,
        splits,
    )

    for split in splits:
        split_total = split_counts[split]
        print(f"\n=== {split} ({split_total} questions) ===")
        split_dir = dataset_dir / split
        question_files = sorted((split_dir / "questions").glob("*.pth"))

        # Load incremental progress from a previous run if it exists
        progress_file = out_dir / f"{split}_progress.pth"
        progress: dict = (
            torch.load(progress_file, weights_only=False)
            if progress_file.exists()
            else {}
        )
        call_num = sum(v["qtype"].shape[0] for v in progress.values())
        if progress:
            logger.info(
                "Resuming %s split: %d/%d questions already answered across %d scans",
                split,
                call_num,
                split_total,
                len(progress),
            )

        all_out = []
        all_y = []
        all_qtype = []
        all_num_nodes = []

        for qf in question_files:
            q_data = torch.load(qf, weights_only=False)
            scan_id = q_data["scanId"]
            dataset_y = q_data["y"]  # [N, num_nodes]
            qtype = q_data["qtype"]  # [N]
            question_texts = q_data["question"]  # list[str], length N
            n_questions, n_nodes = dataset_y.shape

            # Resume: reuse cached scan result without making any new calls
            if scan_id in progress:
                cached = progress[scan_id]
                all_out.append(cached["out"])
                all_y.append(cached["y"])
                all_qtype.append(cached["qtype"])
                all_num_nodes.append(cached["num_nodes"])
                logger.info(
                    "Skipping scan %s (%d questions already answered)",
                    scan_id,
                    n_questions,
                )
                continue

            # Load scene graph files
            sg_emb_path = EMBEDDED_SG_DIR / (scan_id + ".pth")
            sg_path = SCENE_GRAPHS_DIR / (scan_id + ".json")

            missing = [p for p in (sg_emb_path, sg_path) if not p.exists()]
            if missing:
                logger.warning(
                    "Skipping scan %s — missing files: %s",
                    scan_id,
                    [p.name for p in missing],
                )
                continue

            node_map = torch.load(sg_emb_path, weights_only=False)["node_map"]

            with open(sg_path) as f:
                processed_sg = preprocess_scene_graph(json.load(f))

            logger.info(
                "Processing scan %s: %d questions, %d nodes",
                scan_id,
                n_questions,
                n_nodes,
            )

            scan_out = []
            for i, question_text in enumerate(question_texts):
                call_num += 1
                for attempt in range(3):
                    logger.info(
                        "CALL  [%d/%d] scan=%s q=%d question=%r attempt=%d",
                        call_num,
                        split_total,
                        scan_id,
                        i,
                        question_text,
                        attempt,
                    )
                    try:
                        answer_ids = ask_llm(client, processed_sg, question_text)
                        logger.info(
                            "REPLY [%d/%d] scan=%s q=%d answer_ids=%s",
                            call_num,
                            split_total,
                            scan_id,
                            i,
                            answer_ids,
                        )
                        answer_set = {str(a) for a in answer_ids}
                        scan_out.append(
                            torch.tensor(
                                [1.0 if str(n) in answer_set else 0.0 for n in node_map]
                            )
                        )
                        break
                    except openai.RateLimitError as e:
                        logger.error(
                            "[%d/%d] scan=%s q=%d error=%s retrying",
                            call_num,
                            split_total,
                            scan_id,
                            i,
                            e,
                        )
                        time.sleep(5)
                    except Exception as e:
                        logger.error(
                            "[%d/%d] scan=%s q=%d error=%s retrying",
                            call_num,
                            split_total,
                            scan_id,
                            i,
                            e,
                        )
                        break
                else:
                    logger.error(
                        "[%d/%d] scan=%s q=%d failed scan so adding zeros",
                        call_num,
                        split_total,
                        scan_id,
                        i,
                        e,
                    )
                    scan_out.append(torch.zeros(n_nodes))

            # Save this scan's result and flush progress to disk
            scan_result = {
                "out": torch.stack(scan_out).flatten(),
                "y": dataset_y.flatten(),
                "qtype": qtype,
                "num_nodes": torch.full((n_questions,), n_nodes, dtype=torch.long),
            }
            progress[scan_id] = scan_result
            torch.save(progress, progress_file)
            logger.info(
                "Progress saved: %d/%d questions answered", call_num, split_total
            )

            all_out.append(scan_result["out"])
            all_y.append(scan_result["y"])
            all_qtype.append(scan_result["qtype"])
            all_num_nodes.append(scan_result["num_nodes"])

        if not all_out:
            logger.warning(
                "No results to save for split '%s' — all scans skipped.", split
            )
            continue

        out_file = out_dir / f"{split}.pth"
        torch.save(
            {
                "out": torch.cat(all_out),
                "y": torch.cat(all_y),
                "qtype": torch.cat(all_qtype),
                "num_nodes": torch.cat(all_num_nodes),
            },
            out_file,
        )
        logger.info("Saved results for split '%s' to %s", split, out_file)


if __name__ == "__main__":
    main()
