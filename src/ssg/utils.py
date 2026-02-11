from consts import SSG_DIR
import json


OBJECTS_FILE = SSG_DIR / "objects.json"
RELATIONSHIPS_FILE = SSG_DIR / "relationships.json"
RELATIONSHIP_IDS_FILE = SSG_DIR / "relationship_ids.json"


CACHED_JSON = {}


def get_cached_json(path):
    if path in CACHED_JSON:
        return CACHED_JSON[path]
    with open(path) as fp:
        CACHED_JSON[path] = json.load(fp)
    return CACHED_JSON[path]


def get_objects():
    return get_cached_json(OBJECTS_FILE)


def get_relationships():
    return get_cached_json(RELATIONSHIPS_FILE)


def get_relationship_ids():
    path = RELATIONSHIP_IDS_FILE
    if path in CACHED_JSON:
        return CACHED_JSON[path]
    if not path.exists():
        CACHED_JSON[path] = generate_relationship_ids()
    else:
        with open(path) as fp:
            CACHED_JSON[path] = json.load(fp)
    return CACHED_JSON[path]


def generate_relationship_ids():
    relationships = get_relationships()
    relationship_ids = {}
    for scan in relationships["scans"]:
        for relationship in scan["relationships"]:
            if relationship[2] in relationship_ids:
                assert (
                    relationship_ids[relationship[2]] == relationship[3]
                ), "Invalid relationships file"
            else:
                relationship_ids[relationship[2]] = relationship[3]
    keys = list(relationship_ids)
    keys.sort()
    result = {}
    for i in keys:
        result[i] = relationship_ids[i]
    with open(RELATIONSHIP_IDS_FILE, "w") as fp:
        json.dump(result, fp, indent=2)
    return result
