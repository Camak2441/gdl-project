import json
import random
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple

from ssg import load_scene_graph

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from consts import DATA_DIR

SCENE_GRAPH_DIR = DATA_DIR / "scene_graphs"
QUESTION_DIR = DATA_DIR / "questions_programmatic"

SPATIAL_RELS = {
    "to the left of",
    "to the right of",
    "in front of",
    "behind",
    "close by",
    "higher than",
    "lower than",
    "bigger than",
    "smaller than",
    "inside",
}
SUPPORT_RELS = {
    "supported by",
    "standing on",
    "lying on",
    "hanging on",
    "attached to",
    "connected to",
    "leaning against",
    "part of",
    "belonging to",
    "built into",
    "standing in",
    "covering",
    "lying in",
    "hanging in",
}
COMPARATIVE_RELS = {
    "brighter than",
    "darker than",
    "messier than",
    "cleaner than",
    "fuller than",
    "more closed than",
    "more open than",
    "more comfortable than",
    "bigger than",
    "smaller than",
    "higher than",
    "lower than",
}
SAMENESS_RELS = {
    "the same color as",
    "the same material as",
    "the same texture as",
    "the same shape as",
    "the same state as",
    "the same object type as",
    "the same symmetry as",
    "the same as",
}

SUPERLATIVE_MAP = {
    "brighter than": "brightest",
    "darker than": "darkest",
    "messier than": "messiest",
    "cleaner than": "cleanest",
    "fuller than": "fullest",
    "more closed than": "most closed",
    "more open than": "most open",
    "more comfortable than": "most comfortable",
    "bigger than": "biggest",
    "smaller than": "smallest",
    "higher than": "highest",
    "lower than": "lowest",
}

INVERSE_COMPARATIVE = {
    "brighter than": "darker than",
    "darker than": "brighter than",
    "messier than": "cleaner than",
    "cleaner than": "messier than",
    "fuller than": None,
    "more closed than": "more open than",
    "more open than": "more closed than",
    "more comfortable than": None,
    "bigger than": "smaller than",
    "smaller than": "bigger than",
    "higher than": "lower than",
    "lower than": "higher than",
}

BETWEEN_AXIS_PAIRS = [
    ("to the left of", "to the right of"),
    ("in front of", "behind"),
    ("higher than", "lower than"),
]

RANDOM_SEED = 42
MAX_QUESTIONS_PER_SCENE = 100
MIN_ANSWERS = 1
MAX_ANSWERS = 10

MAX_ATTEMPTS_PER_QUESTION = 500


# ---------------------------------------------------------------------------
# Scene graph helpers
# ---------------------------------------------------------------------------


def build_indices(sg: Dict) -> Tuple[Dict, Dict, Dict]:
    obj_by_id = {}
    for o in sg["objects"]:
        obj_by_id[str(o["id"])] = o

    outgoing: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    incoming: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for src, tgt, _rid, rname in sg["relationships"]:
        src_s, tgt_s = str(src), str(tgt)
        outgoing[src_s].append((rname, tgt_s))
        incoming[tgt_s].append((rname, src_s))

    return obj_by_id, outgoing, incoming


def get_article(label: str) -> str:
    if label[0].lower() in "aeiou":
        return "an"
    return "a"


# ---------------------------------------------------------------------------
# Predicate classes  (unchanged)
# ---------------------------------------------------------------------------


class Predicate(ABC):
    @abstractmethod
    def evaluate(
        self, obj_id: str, obj_by_id: Dict, outgoing: Dict, incoming: Dict
    ) -> bool: ...

    @abstractmethod
    def to_text(self) -> str: ...

    @abstractmethod
    def category(self) -> str: ...

    @abstractmethod
    def resolve_references(self, obj_by_id: Dict, rng: random.Random) -> None: ...


class HasLabel(Predicate):
    def __init__(self, label: str):
        self.label = label

    def evaluate(self, obj_id, obj_by_id, outgoing, incoming):
        return obj_by_id[obj_id]["label"] == self.label

    def to_text(self):
        return f"{get_article(self.label)} {self.label}"

    def category(self):
        return None

    def resolve_references(self, obj_by_id, rng):
        pass


class HasAttribute(Predicate):
    def __init__(self, attr_value: str, attr_category: str = None):
        self.attr_value = attr_value
        self.attr_category = attr_category

    def evaluate(self, obj_id, obj_by_id, outgoing, incoming):
        attrs = obj_by_id[obj_id].get("attributes", {})
        for cat, vals in attrs.items():
            if self.attr_category and cat != self.attr_category:
                continue
            if self.attr_value in vals:
                return True
        return False

    def to_text(self):
        return self.attr_value

    def category(self):
        return "semantic"

    def resolve_references(self, obj_by_id, rng):
        pass


class HasAffordance(Predicate):
    def __init__(self, affordance: str):
        self.affordance = affordance

    def evaluate(self, obj_id, obj_by_id, outgoing, incoming):
        return self.affordance in obj_by_id[obj_id].get("affordances", [])

    def to_text(self):
        return f"used for {self.affordance}"

    def category(self):
        return "semantic"

    def resolve_references(self, obj_by_id, rng):
        pass


class HasRelationTo(Predicate):
    def __init__(
        self, rel_name: str, target_label: str, target_attrs: List[str] = None
    ):
        self.rel_name = rel_name
        self.target_label = target_label
        self.target_attrs = target_attrs or []
        self._article = "the"

    def evaluate(self, obj_id, obj_by_id, outgoing, incoming):
        for rname, tgt_id in outgoing.get(obj_id, []):
            if rname != self.rel_name:
                continue
            tgt = obj_by_id.get(tgt_id)
            if tgt is None or tgt["label"] != self.target_label:
                continue
            if self.target_attrs:
                tgt_attr_vals = set()
                for vals in tgt.get("attributes", {}).values():
                    tgt_attr_vals.update(vals)
                if not all(a in tgt_attr_vals for a in self.target_attrs):
                    continue
            return True
        return False

    def _count_matching_targets(self, obj_by_id: Dict) -> int:
        count = 0
        for obj in obj_by_id.values():
            if obj["label"] != self.target_label:
                continue
            if self.target_attrs:
                obj_attr_vals = set()
                for vals in obj.get("attributes", {}).values():
                    obj_attr_vals.update(vals)
                if not all(a in obj_attr_vals for a in self.target_attrs):
                    continue
            count += 1
        return count

    def resolve_references(self, obj_by_id, rng):
        if self._count_matching_targets(obj_by_id) == 1:
            self._article = rng.choice(["the", "the", get_article(self._target_desc())])
        else:
            self._article = get_article(self._target_desc())

    def _target_desc(self) -> str:
        if self.target_attrs:
            return " ".join(self.target_attrs) + " " + self.target_label
        return self.target_label

    def to_text(self):
        return f"{self.rel_name} {self._article} {self._target_desc()}"

    def category(self):
        if self.rel_name in SPATIAL_RELS:
            return "spatial"
        if self.rel_name in SUPPORT_RELS:
            return "support"
        return "semantic"


class HasRelationToChained(Predicate):
    """Subject has rel_name to a target_label object that itself satisfies qualifier.

    The qualifier is any Predicate evaluated on the intermediate object, enabling
    arbitrary nesting depth.
    """

    def __init__(self, rel_name: str, target_label: str, qualifier: "Predicate"):
        self.rel_name = rel_name
        self.target_label = target_label
        self.qualifier = qualifier

    def evaluate(self, obj_id, obj_by_id, outgoing, incoming):
        qualified_mids = {
            mid_id
            for mid_id, mid_obj in obj_by_id.items()
            if mid_obj["label"] == self.target_label
            and self.qualifier.evaluate(mid_id, obj_by_id, outgoing, incoming)
        }
        return any(
            rname == self.rel_name and tgt_id in qualified_mids
            for rname, tgt_id in outgoing.get(obj_id, [])
        )

    def resolve_references(self, obj_by_id, rng):
        self.qualifier.resolve_references(obj_by_id, rng)

    def to_text(self):
        return f"{self.rel_name} the {self.target_label} {self.qualifier.to_text()}"

    def category(self):
        if self.rel_name in SPATIAL_RELS:
            this_type = "spatial"
        elif self.rel_name in SUPPORT_RELS:
            this_type = "support"
        else:
            this_type = "semantic"
        qual_cat = self.qualifier.category()
        if qual_cat == "compound" or (qual_cat is not None and qual_cat != this_type):
            return "compound"
        return this_type


class IsBetween(Predicate):
    """Object lies spatially between label_a and label_b along some axis."""

    def __init__(self, label_a: str, label_b: str):
        self.label_a = label_a
        self.label_b = label_b

    def evaluate(self, obj_id, obj_by_id, outgoing, incoming):
        labels_by_rel: Dict[str, Set[str]] = defaultdict(set)
        for rname, tgt_id in outgoing.get(obj_id, []):
            tgt = obj_by_id.get(tgt_id)
            if tgt:
                labels_by_rel[rname].add(tgt["label"])
        for rel1, rel2 in BETWEEN_AXIS_PAIRS:
            cond_fwd = (
                self.label_a in labels_by_rel[rel1]
                and self.label_b in labels_by_rel[rel2]
            )
            cond_rev = (
                self.label_a in labels_by_rel[rel2]
                and self.label_b in labels_by_rel[rel1]
            )
            if cond_fwd or cond_rev:
                return True
        return False

    def resolve_references(self, obj_by_id, rng):
        pass

    def to_text(self):
        art_a = get_article(self.label_a)
        art_b = get_article(self.label_b)
        return f"between {art_a} {self.label_a} and {art_b} {self.label_b}"

    def category(self):
        return "spatial"


class IsSuperlative(Predicate):
    def __init__(self, comp_rel: str):
        self.comp_rel = comp_rel
        self.superlative = SUPERLATIVE_MAP[comp_rel]
        self.inverse_rel = INVERSE_COMPARATIVE[comp_rel]

    def evaluate(self, obj_id, obj_by_id, outgoing, incoming):
        has_outgoing = any(
            rname == self.comp_rel for rname, _ in outgoing.get(obj_id, [])
        )
        if not has_outgoing:
            return False

        has_incoming = any(
            rname == self.comp_rel for rname, _ in incoming.get(obj_id, [])
        )
        if has_incoming:
            return False

        if self.inverse_rel is not None:
            has_outgoing_inverse = any(
                rname == self.inverse_rel for rname, _ in outgoing.get(obj_id, [])
            )
            if has_outgoing_inverse:
                return False
        return True

    def to_text(self):
        return f"the {self.superlative}"

    def category(self):
        return "semantic"

    def resolve_references(self, obj_by_id, rng):
        pass


class NotPredicate(Predicate):
    def __init__(self, inner: Predicate):
        self.inner = inner

    def evaluate(self, obj_id, obj_by_id, outgoing, incoming):
        return not self.inner.evaluate(obj_id, obj_by_id, outgoing, incoming)

    def resolve_references(self, obj_by_id, rng):
        self.inner.resolve_references(obj_by_id, rng)

    def to_text(self):
        return f"not {self.inner.to_text()}"

    def category(self):
        return self.inner.category()


class OrPredicate(Predicate):
    def __init__(self, a: Predicate, b: Predicate):
        self.a = a
        self.b = b

    def evaluate(self, obj_id, obj_by_id, outgoing, incoming):
        return self.a.evaluate(
            obj_id, obj_by_id, outgoing, incoming
        ) or self.b.evaluate(obj_id, obj_by_id, outgoing, incoming)

    def resolve_references(self, obj_by_id, rng):
        self.a.resolve_references(obj_by_id, rng)
        self.b.resolve_references(obj_by_id, rng)

    def to_text(self):
        return f"{self.a.to_text()} or {self.b.to_text()}"

    def category(self):
        ca, cb = self.a.category(), self.b.category()
        return ca if ca == cb else None


# ---------------------------------------------------------------------------
# Evaluation helpers  (unchanged)
# ---------------------------------------------------------------------------


def evaluate_predicates(
    predicates: List[Predicate], obj_by_id, outgoing, incoming
) -> List[str]:
    results = []
    for obj_id in obj_by_id:
        if all(p.evaluate(obj_id, obj_by_id, outgoing, incoming) for p in predicates):
            results.append(obj_id)
    return results


def _pluralize(label: str) -> str:
    if (
        label.endswith("s")
        or label.endswith("x")
        or label.endswith("sh")
        or label.endswith("ch")
    ):
        return label + "es"
    if label.endswith("y") and label[-2] not in "aeiou":
        return label[:-1] + "ies"
    return label + "s"


def question_type(predicates: List[Predicate]) -> str:
    cats = set()
    for p in predicates:
        c = p.category()
        if c is not None:
            cats.add(c)
    if len(cats) == 1:
        return cats.pop()
    if len(cats) == 0:
        return "semantic"
    return "compound"


# ---------------------------------------------------------------------------
# Collection helpers  (unchanged)
# ---------------------------------------------------------------------------


def collect_attrs(obj_by_id: Dict) -> List[Tuple[str, str]]:
    pairs = set()
    for obj in obj_by_id.values():
        for cat, vals in obj.get("attributes", {}).items():
            if cat == "lexical":
                continue
            for v in vals:
                pairs.add((v, cat))
    return list(pairs)


def collect_rels_by_type(outgoing: Dict, obj_by_id: Dict):
    spatial, support, semantic = set(), set(), set()
    for rels in outgoing.values():
        for rname, tgt_id in rels:
            tgt = obj_by_id.get(tgt_id)
            if tgt is None:
                continue
            pair = (rname, tgt["label"])
            if rname in SPATIAL_RELS:
                spatial.add(pair)
            elif rname in SUPPORT_RELS:
                support.add(pair)
            elif rname in COMPARATIVE_RELS or rname in SAMENESS_RELS:
                semantic.add(pair)
    return list(spatial), list(support), list(semantic)


def collect_affordances(obj_by_id: Dict) -> List[str]:
    affs = set()
    for obj in obj_by_id.values():
        for a in obj.get("affordances", []):
            affs.add(a)
    return list(affs)


def collect_labels(obj_by_id: Dict) -> List[str]:
    return list({obj["label"] for obj in obj_by_id.values()})


def collect_chained_rels(
    outgoing: Dict, incoming: Dict, obj_by_id: Dict
) -> List[Tuple[str, str, str, str]]:
    chains: Set[Tuple[str, str, str, str]] = set()
    for mid_id, mid_obj in obj_by_id.items():
        mid_label = mid_obj["label"]
        inc = incoming.get(mid_id, [])
        out = outgoing.get(mid_id, [])
        if not inc or not out:
            continue
        for rel1, _ in inc:
            for rel2, tgt_id in out:
                tgt = obj_by_id.get(tgt_id)
                if tgt:
                    chains.add((rel1, mid_label, rel2, tgt["label"]))
    return list(chains)


def collect_depth3_chains(
    outgoing: Dict, incoming: Dict, obj_by_id: Dict
) -> List[Tuple[str, str, str, str, str, str]]:
    chains: Set[Tuple[str, str, str, str, str, str]] = set()
    for mid1_id, mid1_obj in obj_by_id.items():
        mid1_label = mid1_obj["label"]
        inc = incoming.get(mid1_id, [])
        out1 = outgoing.get(mid1_id, [])
        if not inc or not out1:
            continue
        for rel1, _ in inc:
            for rel2, mid2_id in out1:
                mid2_obj = obj_by_id.get(mid2_id)
                if mid2_obj is None:
                    continue
                mid2_label = mid2_obj["label"]
                for rel3, tgt_id in outgoing.get(mid2_id, []):
                    tgt = obj_by_id.get(tgt_id)
                    if tgt:
                        chains.add(
                            (rel1, mid1_label, rel2, mid2_label, rel3, tgt["label"])
                        )
    return list(chains)


def collect_between_pairs(outgoing: Dict, obj_by_id: Dict) -> List[Tuple[str, str]]:
    valid_pairs: Set[Tuple[str, str]] = set()
    for obj_id in obj_by_id:
        labels_by_rel: Dict[str, Set[str]] = defaultdict(set)
        for rname, tgt_id in outgoing.get(obj_id, []):
            tgt = obj_by_id.get(tgt_id)
            if tgt:
                labels_by_rel[rname].add(tgt["label"])
        for rel1, rel2 in BETWEEN_AXIS_PAIRS:
            for la in labels_by_rel[rel1]:
                for lb in labels_by_rel[rel2]:
                    if la != lb:
                        valid_pairs.add((la, lb))
    return list(valid_pairs)


# ---------------------------------------------------------------------------
# SceneData
# ---------------------------------------------------------------------------


@dataclass
class SceneData:
    obj_by_id: Dict
    outgoing: Dict
    incoming: Dict
    attrs: List[Tuple[str, str]]
    attrs_by_cat: Dict[str, List[str]]
    labels: List[str]
    affordances: List[str]
    spatial_rels: List[Tuple[str, str]]
    support_rels: List[Tuple[str, str]]
    semantic_rels: List[Tuple[str, str]]
    comp_rels_in_scene: List[str]
    between_pairs: List[Tuple[str, str]]
    chains2: List[Tuple[str, str, str, str]]
    chains3: List[Tuple[str, str, str, str, str, str]]


def build_scene_data(sg: Dict) -> SceneData | None:
    obj_by_id, outgoing, incoming = build_indices(sg)
    if len(obj_by_id) < 3:
        return None

    attrs = collect_attrs(obj_by_id)
    attrs_by_cat: Dict[str, List[str]] = defaultdict(list)
    for v, c in attrs:
        attrs_by_cat[c].append(v)

    labels = collect_labels(obj_by_id)
    affordances = collect_affordances(obj_by_id)
    spatial_rels, support_rels, semantic_rels = collect_rels_by_type(
        outgoing, obj_by_id
    )
    comp_rels_in_scene = list(
        {
            rname
            for rels in outgoing.values()
            for rname, _ in rels
            if rname in SUPERLATIVE_MAP
        }
    )
    chains2 = collect_chained_rels(outgoing, incoming, obj_by_id)
    chains3 = collect_depth3_chains(outgoing, incoming, obj_by_id)
    between_pairs = collect_between_pairs(outgoing, obj_by_id)

    return SceneData(
        obj_by_id=obj_by_id,
        outgoing=outgoing,
        incoming=incoming,
        attrs=attrs,
        attrs_by_cat=dict(attrs_by_cat),
        labels=labels,
        affordances=affordances,
        spatial_rels=spatial_rels,
        support_rels=support_rels,
        semantic_rels=semantic_rels,
        comp_rels_in_scene=comp_rels_in_scene,
        between_pairs=between_pairs,
        chains2=chains2,
        chains3=chains3,
    )


# ---------------------------------------------------------------------------
# Budget-driven predicate generation
# ---------------------------------------------------------------------------

# Weights for the weighted_choice of object-predicate "moves"
_OBJ_MOVE_WEIGHTS = {
    "attr": 3.0,
    "not_attr": 1.0,
    "aff": 2.0,
    "not_aff": 0.5,
    "superlative": 1.5,
    "or_attr": 1.0,
    "or_aff": 0.5,
}

# Weights for relationship predicate form selection
_REL_FORM_WEIGHTS = {
    "simple": 3.0,
    "chained": 2.0,
    "between": 1.0,
}

# Complexity distribution for question generation
_COMPLEXITY_WEIGHTS = {
    1: 1.0,
    2: 2.0,
    3: 3.0,
    4: 3.0,
    5: 2.5,
    6: 2.0,
    7: 1.5,
    8: 1.0,
    9: 0.5,
    10: 0.3,
}


def _weighted_choice(rng: random.Random, items: list, weights: dict):
    """Pick an item proportional to its weight."""
    w_list = [weights.get(item, 1.0) for item in items]
    total = sum(w_list)
    if total == 0:
        return rng.choice(items)
    r = rng.random() * total
    cumulative = 0.0
    for item, w in zip(items, w_list):
        cumulative += w
        if r <= cumulative:
            return item
    return items[-1]


def gen_object_pred(
    budget: int, rng: random.Random, sd: SceneData
) -> Tuple[List[Predicate], int]:
    """Generate object predicates (attrs/affs/label/superlatives/or/not) from a budget.

    Returns (predicate_list, actual_cost).  The predicates are ANDed at the top level.
    """
    preds: List[Predicate] = []
    remaining = budget

    # Step 1: optionally include a label
    include_label = False
    if remaining >= 1 and sd.labels:
        if remaining == 1 and not sd.attrs and not sd.affordances:
            include_label = True
        elif remaining >= 2 and rng.random() < 0.7:
            include_label = True

    if include_label:
        preds.append(HasLabel(rng.choice(sd.labels)))
        remaining -= 1

    if remaining <= 0:
        if not preds and sd.labels:
            return [HasLabel(rng.choice(sd.labels))], 1
        return preds, budget - remaining

    # Step 2: spend remaining budget on clauses
    max_clauses = 5
    retries = 0
    used_attr_vals: Set[str] = set()  # track to avoid duplicates
    used_aff_vals: Set[str] = set()
    while remaining > 0 and len(preds) < max_clauses and retries < 20:
        moves = []
        if remaining >= 1 and sd.attrs:
            moves.append("attr")
            moves.append("not_attr")
        if remaining >= 1 and sd.affordances:
            moves.append("aff")
            moves.append("not_aff")
        if remaining >= 2 and sd.comp_rels_in_scene:
            moves.append("superlative")
        if remaining >= 2 and len(sd.attrs) >= 2:
            moves.append("or_attr")
        if remaining >= 2 and len(sd.affordances) >= 2:
            moves.append("or_aff")

        if not moves:
            break

        move = _weighted_choice(rng, moves, _OBJ_MOVE_WEIGHTS)

        if move == "attr":
            v, cat = rng.choice(sd.attrs)
            if v in used_attr_vals:
                retries += 1
                continue
            used_attr_vals.add(v)
            preds.append(HasAttribute(v, cat))
            remaining -= 1
        elif move == "not_attr":
            v, cat = rng.choice(sd.attrs)
            if v in used_attr_vals:
                retries += 1
                continue
            used_attr_vals.add(v)
            preds.append(NotPredicate(HasAttribute(v, cat)))
            remaining -= 1
        elif move == "aff":
            aff = rng.choice(sd.affordances)
            if aff in used_aff_vals:
                retries += 1
                continue
            used_aff_vals.add(aff)
            preds.append(HasAffordance(aff))
            remaining -= 1
        elif move == "not_aff":
            aff = rng.choice(sd.affordances)
            if aff in used_aff_vals:
                retries += 1
                continue
            used_aff_vals.add(aff)
            preds.append(NotPredicate(HasAffordance(aff)))
            remaining -= 1
        elif move == "superlative":
            preds.append(IsSuperlative(rng.choice(sd.comp_rels_in_scene)))
            remaining -= 2
        elif move == "or_attr":
            a1 = rng.choice(sd.attrs)
            a2 = rng.choice(sd.attrs)
            if a1 == a2 or a1[0] in used_attr_vals or a2[0] in used_attr_vals:
                retries += 1
                continue
            used_attr_vals.add(a1[0])
            used_attr_vals.add(a2[0])
            preds.append(
                OrPredicate(HasAttribute(a1[0], a1[1]), HasAttribute(a2[0], a2[1]))
            )
            remaining -= 2
        elif move == "or_aff":
            a1, a2 = rng.sample(sd.affordances, 2)
            if a1 in used_aff_vals or a2 in used_aff_vals:
                retries += 1
                continue
            used_aff_vals.add(a1)
            used_aff_vals.add(a2)
            preds.append(OrPredicate(HasAffordance(a1), HasAffordance(a2)))
            remaining -= 2

        retries = 0  # reset on success

    if not preds:
        if sd.labels:
            return [HasLabel(rng.choice(sd.labels))], 1
        return [], 0

    return preds, budget - remaining


def _get_rel_pool(rel_type: str, sd: SceneData) -> List[Tuple[str, str]]:
    """Return the relationship pool for a given rel_type."""
    if rel_type == "spatial":
        return sd.spatial_rels
    elif rel_type == "support":
        return sd.support_rels
    else:
        return sd.spatial_rels + sd.support_rels


def gen_rel_pred(
    budget: int, rel_type: str, rng: random.Random, sd: SceneData, _depth: int = 0
) -> Tuple[Predicate, int] | None:
    """Generate a relationship predicate from a budget.

    rel_type is "spatial", "support", or "any".
    Returns (predicate, actual_cost) or None.
    """
    if budget < 2:
        return None

    rel_pool = _get_rel_pool(rel_type, sd)
    if not rel_pool:
        return None

    # Build available forms
    forms = ["simple"]
    if budget >= 3 and _depth < 2:
        forms.append("chained")
    if budget >= 2 and sd.between_pairs and rel_type in ("spatial", "any"):
        forms.append("between")

    form = _weighted_choice(rng, forms, _REL_FORM_WEIGHTS)

    if form == "simple":
        rname, tgt_label = rng.choice(rel_pool)
        # Budget: 1 for the relationship + 1 for the target label = 2 minimum
        target_attr_budget = budget - 2
        target_attrs = []
        if target_attr_budget > 0 and sd.attrs:
            available = [v for v, _ in sd.attrs]
            count = min(target_attr_budget, 2, len(available))
            if count > 0:
                target_attrs = rng.sample(available, count)
        actual = 2 + len(target_attrs)
        return HasRelationTo(rname, tgt_label, target_attrs or None), actual

    elif form == "chained":
        rname, mid_label = rng.choice(rel_pool)
        # The outer hop costs 1.  The inner relationship gets the remaining budget.
        inner_budget = budget - 1
        inner = gen_rel_pred(inner_budget, rel_type, rng, sd, _depth + 1)
        if inner is None:
            # Fallback to simple
            return HasRelationTo(rname, mid_label), 2
        inner_pred, inner_cost = inner
        pred = HasRelationToChained(rname, mid_label, inner_pred)
        return pred, 1 + inner_cost

    elif form == "between":
        la, lb = rng.choice(sd.between_pairs)
        return IsBetween(la, lb), 2

    return None


def gen_preds(
    question_type: str,
    complexity: int,
    rng: random.Random,
    sd: SceneData,
) -> List[Predicate] | None:
    """Generate a predicate list for a question of the given type and complexity.

    Returns a flat list of predicates (ANDed) or None on failure.
    """
    budget = complexity

    # Step 1: determine number of relationship predicates
    if question_type == "semantic":
        num_rels = 0
    else:
        max_rels = max(1, (budget - 1) // 2)
        max_rels = min(max_rels, 3)
        if budget <= 3:
            num_rels = 1
        elif budget <= 5:
            num_rels = _weighted_choice(rng, [1, 2], {1: 3, 2: 1})
        else:
            possible = [n for n in [1, 2, 3] if n * 2 + 1 <= budget and n <= max_rels]
            if not possible:
                possible = [1]
            num_rels = _weighted_choice(rng, possible, {1: 3, 2: 2, 3: 1})

    # Step 2: budget allocation
    min_rel_total = num_rels * 2
    min_obj = 1 if budget > min_rel_total else 0

    if min_rel_total + min_obj > budget:
        num_rels = max(0, (budget - 1) // 2)
        min_rel_total = num_rels * 2
        min_obj = max(0, budget - min_rel_total)
        if num_rels == 0 and question_type != "semantic":
            return None

    obj_budget = min_obj
    rel_budgets = [2] * num_rels
    slack = budget - min_obj - min_rel_total

    for _ in range(slack):
        targets = ["obj"] + list(range(num_rels))
        choice = rng.choice(targets)
        if choice == "obj":
            obj_budget += 1
        else:
            rel_budgets[choice] += 1

    # Step 3: determine rel types
    if question_type == "spatial":
        rel_types = ["spatial"] * num_rels
    elif question_type == "support":
        rel_types = ["support"] * num_rels
    elif question_type == "compound":
        rel_types = [rng.choice(["spatial", "support", "any"]) for _ in range(num_rels)]
    else:
        rel_types = []

    # Step 4: generate object predicate
    obj_preds, _ = gen_object_pred(obj_budget, rng, sd)

    # Step 5: generate relationship predicates (deduplicate by text)
    rel_preds = []
    rel_texts_seen: Set[str] = set()
    for i in range(num_rels):
        result = gen_rel_pred(rel_budgets[i], rel_types[i], rng, sd)
        if result is not None:
            pred, _ = result
            text = pred.to_text()
            if text not in rel_texts_seen:
                rel_texts_seen.add(text)
                rel_preds.append(pred)

    if question_type != "semantic" and not rel_preds:
        return None

    all_preds = obj_preds + rel_preds
    return all_preds if all_preds else None


# ---------------------------------------------------------------------------
# Question rendering
# ---------------------------------------------------------------------------


def _is_attr_like(p: Predicate) -> bool:
    if isinstance(p, (HasAttribute, IsSuperlative)):
        return True
    if isinstance(p, NotPredicate) and isinstance(
        p.inner, (HasAttribute, IsSuperlative)
    ):
        return True
    if isinstance(p, OrPredicate):
        return _is_attr_like(p.a) and _is_attr_like(p.b)
    return False


def _is_aff_like(p: Predicate) -> bool:
    if isinstance(p, HasAffordance):
        return True
    if isinstance(p, NotPredicate) and isinstance(p.inner, HasAffordance):
        return True
    if isinstance(p, OrPredicate):
        return _is_aff_like(p.a) and _is_aff_like(p.b)
    return False


def _render_attr_clause(p: Predicate) -> str:
    """Render an attribute-like predicate as an adjective phrase."""
    if isinstance(p, HasAttribute):
        return p.attr_value
    if isinstance(p, IsSuperlative):
        return f"the {p.superlative}"
    if isinstance(p, NotPredicate):
        return f"not {_render_attr_clause(p.inner)}"
    if isinstance(p, OrPredicate):
        return f"{_render_attr_clause(p.a)} or {_render_attr_clause(p.b)}"
    return p.to_text()


def _render_aff_clause(p: Predicate) -> str:
    """Render an affordance-like predicate as bare affordance text."""
    if isinstance(p, HasAffordance):
        return p.affordance
    if isinstance(p, NotPredicate) and isinstance(p.inner, HasAffordance):
        return f"not {p.inner.affordance}"
    if isinstance(p, OrPredicate):
        return f"{_render_aff_clause(p.a)} or {_render_aff_clause(p.b)}"
    return p.to_text()


def _join_and(parts: List[str]) -> str:
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + ", and " + parts[-1]


def render_question(
    predicates: List[Predicate], num_answers: int, rng: random.Random
) -> str:
    """Linearize a predicate list into a grammatically correct English question.

    Object predicates (attrs/affs/label/superlatives) render as adjectives before
    the noun and an optional "which can be used for ..." clause.  Relationship
    predicates render after the noun joined with "and".
    """
    label_pred = None
    attr_preds: List[Predicate] = []
    aff_preds: List[Predicate] = []
    rel_preds: List[Predicate] = []

    for p in predicates:
        if isinstance(p, HasLabel):
            label_pred = p
        elif _is_attr_like(p):
            attr_preds.append(p)
        elif _is_aff_like(p):
            aff_preds.append(p)
        else:
            rel_preds.append(p)

    # Render each group
    attr_texts = [_render_attr_clause(p) for p in attr_preds]
    aff_texts = [_render_aff_clause(p) for p in aff_preds]
    rel_texts = [p.to_text() for p in rel_preds]

    use_plural = num_answers > 1 and rng.random() < 0.4
    verb = "are" if use_plural else "is"
    interrogative = rng.choice(["Which", "Which", "What"])

    # Build the noun
    if label_pred:
        noun = _pluralize(label_pred.label) if use_plural else label_pred.label
    else:
        noun = "objects" if use_plural else "object"

    # Build the question in three zones:
    #   1. attr zone  – adjectives that go after "is/are": "red, tall or wide, not old"
    #   2. aff zone   – "used for X and Y but not Z"  (joined with the attr zone via "and")
    #   3. rel zone   – relationship clauses: "to the right of the table"
    # The aff zone never starts with "which can be used for" — that phrasing is only
    # used when the affordance follows a noun ("chair which can be used for sitting").
    # After "is/are" we use the bare form: "used for sitting".

    # Combine attrs + affs into a single "property" description
    prop_parts: List[str] = []
    if attr_texts:
        prop_parts.append(", ".join(attr_texts))
    for at in aff_texts:
        prop_parts.append(f"used for {at}")

    # Combine all descriptor parts (properties + relationships)
    desc_parts: List[str] = []
    if prop_parts:
        desc_parts.append(_join_and(prop_parts))
    if rel_texts:
        desc_parts.extend(rel_texts)

    if desc_parts:
        desc = _join_and(desc_parts)
        if label_pred:
            return f"{interrogative} {noun} {verb} {desc}?"
        else:
            if interrogative == "What" and not use_plural and rng.random() < 0.5:
                return f"{interrogative} {verb} {desc}?"
            return f"{interrogative} {noun} {verb} {desc}?"
    elif label_pred:
        if interrogative == "What" and not use_plural and rng.random() < 0.5:
            return f"{interrogative} {verb} {get_article(label_pred.label)} {label_pred.label}?"
        return f"{interrogative} {noun} {verb} {get_article(label_pred.label)} {label_pred.label}?"
    else:
        return None


# ---------------------------------------------------------------------------
# Question assembly
# ---------------------------------------------------------------------------


def try_make_question(
    predicates: List[Predicate],
    sd: SceneData,
    rng: random.Random,
) -> Dict | None:
    answers = evaluate_predicates(predicates, sd.obj_by_id, sd.outgoing, sd.incoming)
    if MIN_ANSWERS <= len(answers) <= MAX_ANSWERS:
        for p in predicates:
            p.resolve_references(sd.obj_by_id, rng)
        q_text = render_question(predicates, len(answers), rng)
        if q_text is None:
            return None
        return {
            "question": q_text,
            "answerObjectIds": sorted(answers),
            "type": question_type(predicates),
        }
    return None


def _type_has_data(qtype: str, sd: SceneData) -> bool:
    if qtype == "semantic":
        return bool(sd.attrs or sd.labels or sd.affordances)
    if qtype == "spatial":
        return bool(sd.spatial_rels)
    if qtype == "support":
        return bool(sd.support_rels)
    if qtype == "compound":
        return bool(sd.spatial_rels or sd.support_rels)
    return False


def generate_questions_for_scene(sg: Dict, rng: random.Random) -> List[Dict]:
    sd = build_scene_data(sg)
    if sd is None:
        return []

    available_types = [
        t
        for t in ("semantic", "spatial", "support", "compound")
        if _type_has_data(t, sd)
    ]
    if not available_types:
        return []

    _base_type_weights = {
        "semantic": 1.0,
        "spatial": 2.0,
        "support": 2.0,
        "compound": 1.0,
    }
    type_weights = {t: _base_type_weights[t] for t in available_types}

    results: List[Dict] = []
    seen: Set[str] = set()
    attempts = 0
    max_attempts = MAX_QUESTIONS_PER_SCENE * MAX_ATTEMPTS_PER_QUESTION

    while len(results) < MAX_QUESTIONS_PER_SCENE and attempts < max_attempts:
        attempts += 1

        qtype = _weighted_choice(rng, available_types, type_weights)
        max_c = 5 if qtype == "semantic" else 10
        valid_cs = [c for c in _COMPLEXITY_WEIGHTS if c <= max_c]
        complexity = _weighted_choice(
            rng, valid_cs, {c: _COMPLEXITY_WEIGHTS[c] for c in valid_cs}
        )

        preds = gen_preds(qtype, complexity, rng, sd)
        if preds is None:
            continue

        q = try_make_question(preds, sd, rng)
        if q and q["question"] not in seen:
            seen.add(q["question"])
            results.append(q)

    rng.shuffle(results)
    return results[:MAX_QUESTIONS_PER_SCENE]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    rng = random.Random(RANDOM_SEED)
    QUESTION_DIR.mkdir(parents=True, exist_ok=True)

    scene_files = sorted(SCENE_GRAPH_DIR.glob("*.json"))
    print(f"Found {len(scene_files)} scene graphs")

    total_questions = 0
    type_counts = defaultdict(int)

    for i, sg_path in enumerate(scene_files):
        sg = load_scene_graph(sg_path)
        scan_id = sg["scan"]

        questions = generate_questions_for_scene(sg, rng)

        if not questions:
            continue

        output = {
            "scanId": scan_id,
            "questions": questions,
        }

        out_path = QUESTION_DIR / f"{scan_id}.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=4)

        for q in questions:
            type_counts[q["type"]] += 1
        total_questions += len(questions)

        if (i + 1) % 50 == 0 or i == len(scene_files) - 1:
            print(
                f"  Processed {i + 1}/{len(scene_files)} scenes, "
                f"{total_questions} questions so far"
            )

    print(
        f"\nDone! Generated {total_questions} questions across {len(scene_files)} scenes"
    )
    print("Type distribution:")
    for qtype, cnt in sorted(type_counts.items()):
        print(f"  {qtype}: {cnt}")


if __name__ == "__main__":
    main()
