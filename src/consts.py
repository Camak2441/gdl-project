from pathlib import Path

import torch

SOURCE_DIR = Path(__file__).parent
PROJECT_ROOT = SOURCE_DIR.parent
SCRIPT_DIR = PROJECT_ROOT / "scripts"
SSG_DIR = PROJECT_ROOT / "3DSSG"
RSCAN_DIR = PROJECT_ROOT / "3RScan"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

SEMSEG_FILE = "semseg.v2.json"
PCD_FILE = "labels.instances.annotated.v2.ply"
MESH_FILE = "mesh.refined.v2.obj"
MESH_SEGS_FILE = "mesh.refined.0.010000.segs.v2.json"


Q_TYPES = {
    "semantic": 0,
    "spatial": 1,
    "support": 2,
    "compound": 3,
    "complex": 4,
}


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
