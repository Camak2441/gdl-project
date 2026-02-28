"""
RAG pipeline evaluation: run GNN to select top-k nodes, then ask an LLM about
the reduced scene graph.

Results are saved in the same format as eval_llm.py:
    {
        "out":       [total_nodes] float - 1.0 if LLM selected node, else 0.0
        "y":         [total_nodes] float - ground truth labels
        "qtype":     [num_queries] int   - question type per query
        "num_nodes": [num_queries] int   - nodes per (full) scene graph
    }

Usage:
    python scripts/eval_rag.py
    python scripts/eval_rag.py --interactive
    python scripts/eval_rag.py --splits test val
    python scripts/eval_rag.py --k 5
"""

import argparse
import json
import logging
import re

import torch
from torch_geometric.data import Batch

logger = logging.getLogger(__name__)

from consts import DATA_DIR, OUTPUT_DIR
from data.query_data import QueryData
from llm_clients import create_client, models as LLM_MODELS
from models import (
    canonical_model_name,
    get_model_name,
    get_most_epochs_file,
    load_encoders_from_model,
    load_model,
)
from utils import intersect_dict


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

K = 10

LLM_MODEL = "gpt-4.1-nano"

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

MODEL_OUT_DIR = OUTPUT_DIR / "models"
RESULTS_OUT_DIR = OUTPUT_DIR / "results"

GNN_MODEL_NAME = get_model_name(
    "vngnn",
    edge_encoder=EDGE_ENCODER,
    node_encoder=NODE_ENCODER,
    query_encoder=QUERY_ENCODER,
    multi=True,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Prompts (same as eval_llm.py)
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


def preprocess_scene_graph(sg_json: dict, selected_ids: set | None = None) -> dict:
    """Preprocess scene graph, optionally filtering objects to selected_ids.

    Relationships are included only when both endpoints are in selected_ids.
    """
    processed = {"objects": [], "relationships": []}
    for obj in sg_json.get("objects", []):
        if selected_ids is None or str(obj.get("id", "")) in selected_ids:
            processed["objects"].append(
                intersect_dict({"id", "label", "affordances", "attributes"}, obj)
            )
    for rel in sg_json.get("relationships", []):
        if len(rel) >= 4:
            from_id, to_id = str(rel[0]), str(rel[1])
            if selected_ids is None or (
                from_id in selected_ids or to_id in selected_ids
            ):
                processed["relationships"].append([rel[0], rel[1], rel[3]])
    return processed


def extract_answer_ids(text: str) -> list[str] | None:
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
    sg_str = json.dumps(processed_sg, indent=2)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Scene graph:\n```json\n{sg_str}\n```\n\nQuestion: {question}",
        },
    ]
    response = client.create_completion(messages)
    ids = extract_answer_ids(response.choices[0].message.content)
    return ids if ids is not None else []


@torch.no_grad()
def run_gnn_for_scan(
    model,
    scan_data,
    queries: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """Run the GNN model for all queries on a single scan.

    Args:
        model: trained GNN model
        scan_data: scene graph Data object with x, pos, edge_index, edge_attr
        queries: [num_queries, query_dim] query embeddings
        device: torch device string

    Returns:
        [num_queries, num_nodes] tensor of node scores
    """
    num_queries = queries.shape[0]

    items = [
        QueryData(
            x=scan_data.x,
            pos=scan_data.pos,
            edge_index=scan_data.edge_index,
            edge_attr=scan_data.edge_attr,
            query=queries[i],
        )
        for i in range(num_queries)
    ]
    batch = Batch.from_data_list(items)
    batch.to(device)

    out = model(
        x=batch.x,
        edge_index=batch.edge_index,
        edge_attr=batch.edge_attr,
        query=batch.query,
        batch=batch.batch,
    )  # [num_queries * num_nodes]

    batch_cpu = batch.batch.cpu()
    out_cpu = out.cpu()
    scores = torch.stack([out_cpu[batch_cpu == q] for q in range(num_queries)])
    return scores  # [num_queries, num_nodes]


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


def get_available_models() -> list[str]:
    if not MODEL_OUT_DIR.exists():
        return []
    return sorted(p.name for p in MODEL_OUT_DIR.iterdir() if p.is_dir())


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
        description=(
            "Evaluate a RAG pipeline: use a GNN to retrieve top-k scene graph nodes, "
            "then ask an LLM about the reduced scene graph."
        )
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Interactively select GNN model, LLM model, and dataset.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test"],
        metavar="SPLIT",
        help="Dataset splits to evaluate (default: test).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=K,
        help=f"Number of top nodes to retrieve from GNN (default: {K}).",
    )
    args = parser.parse_args()

    k = args.k
    gnn_model_name = GNN_MODEL_NAME
    llm_model_name = LLM_MODEL
    dataset_dir = DATASET_DIR

    if args.interactive:
        models_list = get_available_models()
        if not models_list:
            print(f"No GNN models found in {MODEL_OUT_DIR}")
            return
        gnn_model_name = select_interactively(models_list, "Select a GNN model:")

        e_enc, n_enc, q_enc = load_encoders_from_model(gnn_model_name)
        if e_enc and n_enc and q_enc:
            inferred = DATA_DIR / "dataset_balanced" / ";".join([e_enc, n_enc, q_enc])
            if inferred.exists():
                print(f"\nDataset inferred from model name: {inferred.name}")
                use_inferred = input("Use this dataset? [Y/n]: ").strip().lower()
                if use_inferred in ("", "y", "yes"):
                    dataset_dir = inferred
                else:
                    datasets = get_available_datasets()
                    if not datasets:
                        print("No datasets found.")
                        return
                    dataset_enc = select_interactively(datasets, "Select a dataset:")
                    dataset_dir = DATA_DIR / "dataset_balanced" / dataset_enc
            else:
                datasets = get_available_datasets()
                if not datasets:
                    print("No datasets found.")
                    return
                dataset_enc = select_interactively(datasets, "Select a dataset:")
                dataset_dir = DATA_DIR / "dataset_balanced" / dataset_enc
        else:
            datasets = get_available_datasets()
            if not datasets:
                print("No datasets found.")
                return
            dataset_enc = select_interactively(datasets, "Select a dataset:")
            dataset_dir = DATA_DIR / "dataset_balanced" / dataset_enc

        llm_model_name = select_interactively(
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

    # Locate GNN checkpoint
    gnn_model_dir = MODEL_OUT_DIR / gnn_model_name
    if not gnn_model_dir.exists():
        print(f"GNN model directory not found: {gnn_model_dir}")
        return

    most_epochs = get_most_epochs_file(gnn_model_dir)
    if most_epochs is None:
        print(f"No checkpoint files found in {gnn_model_dir}")
        return
    model_path = gnn_model_dir / f"{most_epochs}.pth"

    # Derive embedded scene graph directory from GNN encoder names
    e_enc, n_enc, _ = load_encoders_from_model(gnn_model_name)
    embedded_sg_dir = DATA_DIR / "embedded_scene_graphs" / ";".join([e_enc, n_enc])

    # Summarise calls, accounting for incremental progress
    split_counts = {s: count_questions_in_split(dataset_dir / s) for s in splits}

    out_dir = RESULTS_OUT_DIR / f"rag_{llm_model_name}_k{k}" / gnn_model_name
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

    print(f"\nGNN model:  {gnn_model_name}")
    print(f"Weights:    {model_path.name} ({most_epochs} epochs)")
    print(f"LLM model:  {llm_model_name}")
    print(f"Dataset:    {dataset_dir.name}")
    print(f"k:          {k}")
    for s in splits:
        done, total = already_done[s], split_counts[s]
        suffix = f" ({done} already done, {remaining[s]} remaining)" if done else ""
        print(f"  {s}: {total} questions{suffix}")
    print(f"Total LLM calls to make: {total_remaining}")

    answer = input("\nProceed? [y/N] ").strip().lower()
    if answer not in ("y", "yes"):
        print("Aborted.")
        return

    print("\nLoading GNN model...")
    gnn_model = load_model(gnn_model_name)
    gnn_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    gnn_model.to(DEVICE)
    gnn_model.eval()

    llm_client = create_client(
        model=llm_model_name, temperature=0.0, max_completion_tokens=256
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logger.info(
        "Starting eval_rag: gnn=%s llm=%s dataset=%s splits=%s k=%d",
        gnn_model_name,
        llm_model_name,
        dataset_dir.name,
        splits,
        k,
    )

    for split in splits:
        split_total = split_counts[split]
        print(f"\n=== {split} ({split_total} questions) ===")
        split_dir = dataset_dir / split
        question_files = sorted((split_dir / "questions").glob("*.pth"))
        scan_graphs_dir = split_dir / "scene_graphs"

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
            dataset_y = q_data["y"]  # [num_queries, num_nodes]
            qtype = q_data["qtype"]  # [num_queries]
            question_texts = q_data["question"]  # list[str], length num_queries
            queries = q_data["query"]  # [num_queries, query_dim]
            n_questions, n_nodes = dataset_y.shape

            # Resume: reuse cached scan result without any new calls
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

            # Check required files exist
            sg_emb_path = embedded_sg_dir / (scan_id + ".pth")
            sg_path = SCENE_GRAPHS_DIR / (scan_id + ".json")
            scan_graph_path = scan_graphs_dir / (scan_id + ".pth")

            missing = [
                p for p in (sg_emb_path, sg_path, scan_graph_path) if not p.exists()
            ]
            if missing:
                logger.warning(
                    "Skipping scan %s — missing files: %s",
                    scan_id,
                    [p.name for p in missing],
                )
                continue

            sg_emb = torch.load(sg_emb_path, weights_only=False)
            node_map = sg_emb["node_map"]  # list: index -> object ID

            with open(sg_path) as f:
                sg_json = json.load(f)

            scan_data = torch.load(scan_graph_path, weights_only=False)

            logger.info(
                "Processing scan %s: %d questions, %d nodes",
                scan_id,
                n_questions,
                n_nodes,
            )

            # Run GNN for all queries in this scan at once
            scores = run_gnn_for_scan(gnn_model, scan_data, queries, DEVICE)
            # scores: [num_queries, num_nodes]

            scan_out = []
            for i, question_text in enumerate(question_texts):
                call_num += 1

                # Select top-k nodes by GNN score
                top_k = min(k, n_nodes)
                top_k_indices = scores[i].topk(top_k).indices.tolist()
                selected_ids = {str(node_map[idx]) for idx in top_k_indices}

                reduced_sg = preprocess_scene_graph(sg_json, selected_ids)

                logger.info(
                    "CALL  [%d/%d] scan=%s q=%d question=%r selected_ids=%s",
                    call_num,
                    split_total,
                    scan_id,
                    i,
                    question_text,
                    sorted(selected_ids),
                )

                try:
                    answer_ids = ask_llm(llm_client, reduced_sg, question_text)
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
                except Exception as e:
                    logger.error(
                        "ERROR [%d/%d] scan=%s q=%d error=%s",
                        call_num,
                        split_total,
                        scan_id,
                        i,
                        e,
                    )
                    scan_out.append(torch.zeros(n_nodes))

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
