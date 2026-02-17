import json
from pathlib import Path
from typing import Any, Dict
from ssg.scene_graph_3d import SceneGraph3D


def load_scene_graph(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)
