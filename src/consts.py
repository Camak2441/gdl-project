from pathlib import Path

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


Q_TYPES = {
    "semantic": 0,
    "spatial": 1,
    "support": 2,
}
