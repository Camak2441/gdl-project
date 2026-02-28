import torch
import json
import random
from collections import defaultdict
from sentence_transformers import SentenceTransformer

from consts import DATA_DIR, Q_TYPES
from encoders import get_edge_encoder, get_node_encoder, get_query_encoder
from json_validator import generate_json_schema
from ssg import SceneGraph3D

EDGE_ENCODER = "all_minilm_l6v2"
NODE_ENCODER = "all_minilm_l6v2"
QUERY_ENCODER = "all_minilm_l6v2"

DATASET_DIR = (
    DATA_DIR
    / "dataset_single_answer"
    / ";".join([EDGE_ENCODER, NODE_ENCODER, QUERY_ENCODER])
)
TRAIN_DATASET_DIR = DATASET_DIR / "train"
TEST_DATASET_DIR = DATASET_DIR / "test"
VAL_DATASET_DIR = DATASET_DIR / "val"
EMBEDDED_SCENE_GRAPH_DIR = (
    DATA_DIR / "embedded_scene_graphs" / ";".join([EDGE_ENCODER, NODE_ENCODER])
)
QUESTIONS_DIR = DATA_DIR / "questions_single_answer"
SCENE_GRAPH_DIR = DATA_DIR / "scene_graphs"

# Split configuration
UNSEEN_SCENE_RATIO = 0.15  # Ratio of scenes held out entirely for test/val
SEEN_SCENE_HOLDOUT_RATIO = (
    0.1  # Ratio of questions from training scenes held out for test/val
)
RANDOM_SEED = 0


QUESTION_SCHEMA = generate_json_schema(
    {
        "scanId": str,
        "questions": [
            {
                "question": str,
                "answerObjectIds": [str],
                "type": ("lit", *(Q_TYPES.keys())),
            }
        ],
    }
)

device = "cuda" if torch.cuda.is_available() else "cpu"

encode_node = get_node_encoder(NODE_ENCODER)
encode_edge = get_edge_encoder(EDGE_ENCODER)
encode_query = get_query_encoder(QUERY_ENCODER)


def load_questions_metadata(q_path):
    """Load question file and return scan_id and list of question types."""
    with open(q_path) as fp:
        questions = json.load(fp)

    if questions is None or not QUESTION_SCHEMA.valid(questions):
        return None, []

    scan_id = questions["scanId"]
    question_types = [q["type"] for q in questions["questions"]]
    return scan_id, question_types


def embed_scene_graph(scan_id):
    """Embed a scene graph and return the data, using a disk cache.

    Loads from EMBEDDED_SG_DIR if a cached embedding exists, otherwise
    computes the embedding and saves it for future use.

    Returns:
        Tuple of (data, node_map) or (None, None) if failed
    """
    cache_path = EMBEDDED_SCENE_GRAPH_DIR / (scan_id + ".pth")

    if cache_path.exists():
        cached = torch.load(cache_path, weights_only=False)
        return cached["data"], cached["node_map"]

    scenegraph = SceneGraph3D.from_json(SCENE_GRAPH_DIR / (scan_id + ".json"))

    if scenegraph is None:
        return None, None

    scenegraph.load_3d_data()

    data, node_map, _ = SceneGraph3D.to_query_data(
        scenegraph,
        node_encoder=encode_node,
        edge_encoder=encode_edge,
        ret_node_maps=True,
    )

    EMBEDDED_SCENE_GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"data": data, "node_map": node_map}, cache_path)

    return data, node_map


def save_scene_graph_for_split(scan_id, data, output_dir):
    """Save the embedded scene graph Data object for a split.

    Args:
        scan_id: The scan ID (used as filename stem)
        data: Pre-computed pyg Data object
        output_dir: Split root directory (scene_graphs/ subdir created here)
    """
    scene_graphs_dir = output_dir / "scene_graphs"
    scene_graphs_dir.mkdir(parents=True, exist_ok=True)
    dest = scene_graphs_dir / (scan_id + ".pth")
    if not dest.exists():
        torch.save(data, dest)


def save_questions_for_split(
    scan_id, questions, question_indices, node_map, output_dir
):
    """Save questions for a specific split as dicts with query/y/scanId keys.

    Args:
        scan_id: The scan ID
        questions: List of question dicts
        question_indices: Set of question indices to save (None = all)
        node_map: Pre-computed node map
        output_dir: Split root directory (questions/ subdir created here)

    Returns:
        Number of questions saved
    """
    questions_dir = output_dir / "questions"
    questions_dir.mkdir(parents=True, exist_ok=True)

    question_datas = {"questions": [], "ys": [], "qtypes": []}

    for id, q in enumerate(questions):
        if question_indices is not None and id not in question_indices:
            continue

        question_datas["questions"].append(q["question"])
        question_datas["ys"].append(
            torch.tensor(
                [1.0 if node in q["answerObjectIds"] else 0.0 for node in node_map]
            )
        )
        question_datas["qtypes"].append(Q_TYPES[q["type"]])

    scene_graph_questions = {
        "question": question_datas["questions"],
        "query": encode_query(question_datas["questions"]),
        "y": torch.stack(question_datas["ys"]),
        "qtype": torch.tensor(question_datas["qtypes"]),
        "scanId": scan_id,
    }

    torch.save(
        scene_graph_questions,
        questions_dir / (scan_id + ".pth"),
    )

    return len(question_datas["questions"])


def main():
    random.seed(RANDOM_SEED)

    print("Ensuring dataset directories exist")
    for split_dir in (TRAIN_DATASET_DIR, TEST_DATASET_DIR, VAL_DATASET_DIR):
        (split_dir / "scene_graphs").mkdir(parents=True, exist_ok=True)
        (split_dir / "questions").mkdir(parents=True, exist_ok=True)

    print("Loading question metadata...")

    # Collect all question files and their metadata
    question_files = list(QUESTIONS_DIR.iterdir())
    scan_to_path = {}
    scan_to_questions = {}  # scan_id -> list of question types
    scan_to_questions_data = {}  # scan_id -> list of question dicts

    for path in question_files:
        scan_id, question_types = load_questions_metadata(path)
        if scan_id is not None:
            scan_to_path[scan_id] = path
            scan_to_questions[scan_id] = question_types
            # Also load the full questions data
            with open(path) as fp:
                scan_to_questions_data[scan_id] = json.load(fp)["questions"]

    all_scans = list(scan_to_path.keys())
    random.shuffle(all_scans)

    # Split scenes: some are unseen (only in test/val), others are in training
    n_unseen = max(2, int(len(all_scans) * UNSEEN_SCENE_RATIO))
    n_unseen_test = n_unseen // 2

    unseen_test_scans = set(all_scans[:n_unseen_test])
    unseen_val_scans = set(all_scans[n_unseen_test:n_unseen])
    train_scans = all_scans[n_unseen:]

    print(f"\nDataset split:")
    print(f"  Unseen test scenes: {len(unseen_test_scans)}")
    print(f"  Unseen val scenes: {len(unseen_val_scans)}")
    print(f"  Training scenes: {len(train_scans)}")

    # For training scenes, split questions so some go to test/val
    # Ensure all question types are represented in test/val from seen scenes
    train_scene_splits = {}  # scan_id -> {train_indices, test_indices, val_indices}

    for scan_id in train_scans:
        q_types = scan_to_questions[scan_id]

        # Group questions by type
        type_to_indices = defaultdict(list)
        for i, qtype in enumerate(q_types):
            type_to_indices[qtype].append(i)

        train_indices = set()
        test_indices = set()
        val_indices = set()

        # For each question type, hold out some for test/val
        for qtype, indices in type_to_indices.items():
            random.shuffle(indices)
            n_holdout = max(1, int(len(indices) * SEEN_SCENE_HOLDOUT_RATIO))

            # Split holdout between test and val
            n_test = max(1, n_holdout // 2) if n_holdout >= 2 else 0
            n_val = n_holdout - n_test if n_holdout >= 1 else 0

            # If only 1 question of this type, randomly assign to test or val
            if len(indices) == 1:
                if random.random() < 0.5:
                    train_indices.add(indices[0])
                else:
                    if random.random() < 0.5:
                        test_indices.add(indices[0])
                    else:
                        val_indices.add(indices[0])
            elif len(indices) == 2:
                train_indices.add(indices[0])
                if random.random() < 0.5:
                    test_indices.add(indices[1])
                else:
                    val_indices.add(indices[1])
            else:
                test_indices.update(indices[:n_test])
                val_indices.update(indices[n_test : n_test + n_val])
                train_indices.update(indices[n_test + n_val :])

        train_scene_splits[scan_id] = {
            "train": train_indices,
            "test": test_indices,
            "val": val_indices,
        }

    # Build a mapping of which splits each scan contributes to
    scan_splits = {}  # scan_id -> {split_name: indices}
    for scan_id in train_scans:
        scan_splits[scan_id] = train_scene_splits[scan_id]
    for scan_id in unseen_test_scans:
        scan_splits[scan_id] = {"test": None}  # None means all questions
    for scan_id in unseen_val_scans:
        scan_splits[scan_id] = {"val": None}  # None means all questions

    split_to_dir = {
        "train": TRAIN_DATASET_DIR,
        "test": TEST_DATASET_DIR,
        "val": VAL_DATASET_DIR,
    }

    # Process each scene graph once
    print("\nEmbedding scene graphs and generating dataset...")
    train_count = 0
    test_count = 0
    val_count = 0

    for i, scan_id in enumerate(all_scans):
        print(f"  Processing {i + 1}/{len(all_scans)}: {scan_id}")

        # Embed the scene graph once
        data, node_map = embed_scene_graph(scan_id)
        if data is None:
            print(f"    Skipping {scan_id}: failed to load scene graph")
            continue

        questions = scan_to_questions_data[scan_id]

        # Save questions and scene graph for each split this scan contributes to
        for split_name, indices in scan_splits[scan_id].items():
            if indices is None or indices:  # None means all, or non-empty set
                split_dir = split_to_dir[split_name]
                save_scene_graph_for_split(scan_id, data, split_dir)
                count = save_questions_for_split(
                    scan_id,
                    questions,
                    indices,
                    node_map,
                    split_dir,
                )
                if split_name == "train":
                    train_count += count
                elif split_name == "test":
                    test_count += count
                else:
                    val_count += count

    print(f"\n=== Dataset Generation Complete ===")
    print(f"Training samples: {train_count}")
    print(f"Test samples: {test_count}")
    print(f"Validation samples: {val_count}")
    print(f"Total: {train_count + test_count + val_count}")


if __name__ == "__main__":
    main()
