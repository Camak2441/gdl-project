import os

from consts import SSG_DIR, DATA_DIR
from ssg import SceneGraph3D
from ssg.utils import get_objects, get_relationships

# Paths to the 3DSSG dataset
REL_FILE = SSG_DIR / "relationships.json"
OBJ_FILE = SSG_DIR / "objects.json"

# Output directory for generated JSONs
OUTPUT_DIR = DATA_DIR / "scene_graphs"


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load relationships and objects
    scans = get_objects()["scans"]
    objects = get_relationships()["scans"]
    objects = {obj["scan"]: obj for obj in objects}

    # Combine objects into scans
    for scan in scans:
        id_scan = scan["scan"]
        scan["objects"] = objects[id_scan]["objects"]

    print(f"Processing {len(scans)} scans...")

    # Generate JSON for each scan
    for _, scan in enumerate(scans):
        scan_id = scan["scan"]
        g = SceneGraph3D.from_dict(scan)
        output_path = os.path.join(OUTPUT_DIR, f"{scan_id}.json")
        SceneGraph3D.to_json(g, output_path, convert=True)
