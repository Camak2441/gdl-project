import json
import random
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

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

RANDOM_SEED = 42
MAX_QUESTIONS_PER_SCENE = 30
MIN_ANSWERS = 1
MAX_ANSWERS = 10

MAX_CANDIDATES_PER_STRATEGY = 200


def load_scene_graph(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


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
# Predicate types
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

    def resolve_references(self, obj_by_id: Dict, rng: random.Random) -> None:
        """Optionally resolve contextual details (e.g. article choice) before
        text generation. Subclasses may override; default is a no-op."""


class HasLabel(Predicate):
    def __init__(self, label: str):
        self.label = label

    def evaluate(self, obj_id, obj_by_id, outgoing, incoming):
        return obj_by_id[obj_id]["label"] == self.label

    def to_text(self):
        return f"{get_article(self.label)} {self.label}"

    def category(self):
        # A label constraint is a filter, not a question category.
        # Returning None lets the other predicates determine the type
        # (e.g. label + support → support, not compound).
        return None


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


class HasAffordance(Predicate):
    def __init__(self, affordance: str):
        self.affordance = affordance

    def evaluate(self, obj_id, obj_by_id, outgoing, incoming):
        return self.affordance in obj_by_id[obj_id].get("affordances", [])

    def to_text(self):
        return f"used for {self.affordance}"

    def category(self):
        return "semantic"


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


class IsSuperlative(Predicate):
    def __init__(self, comp_rel: str):
        self.comp_rel = comp_rel
        self.superlative = SUPERLATIVE_MAP[comp_rel]
        self.inverse_rel = INVERSE_COMPARATIVE[comp_rel]

    def evaluate(self, obj_id, obj_by_id, outgoing, incoming):
        # Must have at least one outgoing comparative edge
        has_outgoing = any(
            rname == self.comp_rel for rname, _ in outgoing.get(obj_id, [])
        )
        if not has_outgoing:
            return False
        # Must not have any incoming edge of the same comparative
        # (i.e. nothing else claims to be "X-er than" this object)
        has_incoming = any(
            rname == self.comp_rel for rname, _ in incoming.get(obj_id, [])
        )
        if has_incoming:
            return False
        # Must not have any outgoing edge of the inverse comparative
        # (e.g. if looking for "darkest", the object must not be
        # "brighter than" anything, because that means something
        # else is darker than it)
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
# Question
# ---------------------------------------------------------------------------


def _join_parts(parts: List[str]) -> str:
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + ", and " + parts[-1]


def evaluate_predicates(
    predicates: List[Predicate], obj_by_id, outgoing, incoming
) -> List[str]:
    results = []
    for obj_id in obj_by_id:
        if all(p.evaluate(obj_id, obj_by_id, outgoing, incoming) for p in predicates):
            results.append(obj_id)
    return results


def _pluralize(label: str) -> str:
    """Simple pluralisation for object labels."""
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


def predicates_to_question(
    predicates: List[Predicate],
    num_answers: int,
    rng: random.Random,
) -> str:
    label_part = None
    prop_parts = []
    for p in predicates:
        if isinstance(p, HasLabel):
            label_part = p.label
        else:
            prop_parts.append(p.to_text())

    # Choose between "Which" and "What"
    interrogative = rng.choice(["Which", "Which", "What"])

    # Choose singular vs plural based on answer count
    use_plural = num_answers > 1 and rng.random() < 0.4

    if label_part and prop_parts:
        noun = _pluralize(label_part) if use_plural else label_part
        verb = "are" if use_plural else "is"
        return f"Which {noun} {verb} {_join_parts(prop_parts)}?"
    elif label_part:
        noun = "objects" if use_plural else "object"
        verb = "are" if use_plural else "is"
        if interrogative == "What" and not use_plural and rng.random() < 0.5:
            return f"{interrogative} {verb} {get_article(label_part)} {label_part}?"
        return f"{interrogative} {noun} {verb} {get_article(label_part)} {label_part}?"
    else:
        noun = "objects" if use_plural else "object"
        verb = "are" if use_plural else "is"
        if interrogative == "What" and not use_plural and rng.random() < 0.5:
            return f"{interrogative} {verb} {_join_parts(prop_parts)}?"
        return f"{interrogative} {noun} {verb} {_join_parts(prop_parts)}?"


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


def try_make_question(
    predicates: List[Predicate],
    obj_by_id,
    outgoing,
    incoming,
    rng: random.Random,
) -> Dict | None:
    answers = evaluate_predicates(predicates, obj_by_id, outgoing, incoming)
    if MIN_ANSWERS <= len(answers) <= MAX_ANSWERS:
        for p in predicates:
            p.resolve_references(obj_by_id, rng)
        return {
            "question": predicates_to_question(predicates, len(answers), rng),
            "answerObjectIds": sorted(answers),
            "type": question_type(predicates),
        }
    return None


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


def _sample_and_try(
    rng: random.Random,
    generators: List,
    obj_by_id,
    outgoing,
    incoming,
    count: int,
) -> List[Dict]:
    results = []
    seen = set()
    attempts = 0
    max_attempts = count * MAX_CANDIDATES_PER_STRATEGY

    while len(results) < count and attempts < max_attempts:
        gen = rng.choice(generators)
        preds = gen(rng)
        if preds is None:
            attempts += 1
            continue
        q = try_make_question(preds, obj_by_id, outgoing, incoming, rng)
        if q and q["question"] not in seen:
            seen.add(q["question"])
            results.append(q)
        attempts += 1

    return results


def generate_semantic_questions(
    obj_by_id,
    outgoing,
    incoming,
    rng,
    count,
    attrs,
    labels,
    semantic_rels,
    affordances,
) -> List[Dict]:
    if not attrs and not labels and not affordances:
        return []

    generators = []

    if labels and attrs:

        def gen_label_attr(rng):
            label = rng.choice(labels)
            v, cat = rng.choice(attrs)
            return [HasLabel(label), HasAttribute(v, cat)]

        generators.append(gen_label_attr)

    if len(attrs) >= 2:

        def gen_two_attrs(rng):
            a1 = rng.choice(attrs)
            a2 = rng.choice(attrs)
            if a1[1] == a2[1]:
                return None
            return [HasAttribute(a1[0], a1[1]), HasAttribute(a2[0], a2[1])]

        generators.append(gen_two_attrs)

    if attrs:
        cats = defaultdict(list)
        for v, c in attrs:
            cats[c].append(v)
        multi_cats = {c: vs for c, vs in cats.items() if len(vs) >= 2}
        if multi_cats:

            def gen_attr_not(rng):
                cat = rng.choice(list(multi_cats.keys()))
                vs = multi_cats[cat]
                v1, v2 = rng.sample(vs, 2)
                return [HasAttribute(v1, cat), NotPredicate(HasAttribute(v2, cat))]

            generators.append(gen_attr_not)

    if attrs and semantic_rels:

        def gen_attr_rel(rng):
            v, cat = rng.choice(attrs)
            rname, tgt = rng.choice(semantic_rels)
            return [HasAttribute(v, cat), HasRelationTo(rname, tgt)]

        generators.append(gen_attr_rel)

    comp_rels_in_scene = set()
    for rels in outgoing.values():
        for rname, _ in rels:
            if rname in SUPERLATIVE_MAP:
                comp_rels_in_scene.add(rname)
    comp_rels_list = list(comp_rels_in_scene)
    if comp_rels_list:

        def gen_superlative(rng):
            return [IsSuperlative(rng.choice(comp_rels_list))]

        generators.append(gen_superlative)

        if labels:

            def gen_label_super(rng):
                return [
                    HasLabel(rng.choice(labels)),
                    IsSuperlative(rng.choice(comp_rels_list)),
                ]

            generators.append(gen_label_super)

    if attrs and labels:

        def gen_or_attrs(rng):
            a1 = rng.choice(attrs)
            a2 = rng.choice(attrs)
            if a1 == a2:
                return None
            label = rng.choice(labels)
            return [
                HasLabel(label),
                OrPredicate(HasAttribute(a1[0], a1[1]), HasAttribute(a2[0], a2[1])),
            ]

        generators.append(gen_or_attrs)

    if affordances:

        def gen_affordance(rng):
            return [HasAffordance(rng.choice(affordances))]

        generators.append(gen_affordance)

    if labels and affordances:

        def gen_label_affordance(rng):
            label = rng.choice(labels)
            aff = rng.choice(affordances)
            return [HasLabel(label), HasAffordance(aff)]

        generators.append(gen_label_affordance)

    if attrs and affordances:

        def gen_attr_affordance(rng):
            v, cat = rng.choice(attrs)
            aff = rng.choice(affordances)
            return [HasAttribute(v, cat), HasAffordance(aff)]

        generators.append(gen_attr_affordance)

    if len(affordances) >= 2:

        def gen_affordance_not(rng):
            a1, a2 = rng.sample(affordances, 2)
            return [HasAffordance(a1), NotPredicate(HasAffordance(a2))]

        generators.append(gen_affordance_not)

    if labels and attrs and affordances:

        def gen_label_attr_affordance(rng):
            label = rng.choice(labels)
            v, cat = rng.choice(attrs)
            aff = rng.choice(affordances)
            return [HasLabel(label), HasAttribute(v, cat), HasAffordance(aff)]

        generators.append(gen_label_attr_affordance)

    if not generators:
        return []

    return _sample_and_try(rng, generators, obj_by_id, outgoing, incoming, count)


def generate_spatial_questions(
    obj_by_id,
    outgoing,
    incoming,
    rng,
    count,
    spatial_rels,
    labels,
) -> List[Dict]:
    if not spatial_rels:
        return []

    generators = []

    def gen_single(rng):
        rname, tgt = rng.choice(spatial_rels)
        return [HasRelationTo(rname, tgt)]

    generators.append(gen_single)

    if labels:

        def gen_label_spatial(rng):
            label = rng.choice(labels)
            rname, tgt = rng.choice(spatial_rels)
            return [HasLabel(label), HasRelationTo(rname, tgt)]

        generators.append(gen_label_spatial)

    if len(spatial_rels) >= 2:

        def gen_two_spatial(rng):
            r1 = rng.choice(spatial_rels)
            r2 = rng.choice(spatial_rels)
            if r1 == r2:
                return None
            return [HasRelationTo(r1[0], r1[1]), HasRelationTo(r2[0], r2[1])]

        generators.append(gen_two_spatial)

    if len(spatial_rels) >= 2:

        def gen_spatial_not(rng):
            r1 = rng.choice(spatial_rels)
            r2 = rng.choice(spatial_rels)
            if r1 == r2:
                return None
            return [
                HasRelationTo(r1[0], r1[1]),
                NotPredicate(HasRelationTo(r2[0], r2[1])),
            ]

        generators.append(gen_spatial_not)

    return _sample_and_try(rng, generators, obj_by_id, outgoing, incoming, count)


def generate_support_questions(
    obj_by_id,
    outgoing,
    incoming,
    rng,
    count,
    support_rels,
    labels,
) -> List[Dict]:
    if not support_rels:
        return []

    generators = []

    def gen_single(rng):
        rname, tgt = rng.choice(support_rels)
        return [HasRelationTo(rname, tgt)]

    generators.append(gen_single)

    if labels:

        def gen_label_support(rng):
            label = rng.choice(labels)
            rname, tgt = rng.choice(support_rels)
            return [HasLabel(label), HasRelationTo(rname, tgt)]

        generators.append(gen_label_support)

    if len(support_rels) >= 2:

        def gen_two_support(rng):
            r1 = rng.choice(support_rels)
            r2 = rng.choice(support_rels)
            if r1 == r2:
                return None
            return [HasRelationTo(r1[0], r1[1]), HasRelationTo(r2[0], r2[1])]

        generators.append(gen_two_support)

    if len(support_rels) >= 2:

        def gen_support_not(rng):
            r1 = rng.choice(support_rels)
            r2 = rng.choice(support_rels)
            if r1 == r2:
                return None
            return [
                HasRelationTo(r1[0], r1[1]),
                NotPredicate(HasRelationTo(r2[0], r2[1])),
            ]

        generators.append(gen_support_not)

    return _sample_and_try(rng, generators, obj_by_id, outgoing, incoming, count)


def generate_compound_questions(
    obj_by_id,
    outgoing,
    incoming,
    rng,
    count,
    attrs,
    labels,
    spatial_rels,
    support_rels,
    semantic_rels,
    affordances,
) -> List[Dict]:
    generators = []

    if attrs and spatial_rels:

        def gen_attr_spatial(rng):
            v, cat = rng.choice(attrs)
            rname, tgt = rng.choice(spatial_rels)
            return [HasAttribute(v, cat), HasRelationTo(rname, tgt)]

        generators.append(gen_attr_spatial)

    if attrs and support_rels:

        def gen_attr_support(rng):
            v, cat = rng.choice(attrs)
            rname, tgt = rng.choice(support_rels)
            return [HasAttribute(v, cat), HasRelationTo(rname, tgt)]

        generators.append(gen_attr_support)

    if labels and attrs and spatial_rels:

        def gen_label_attr_spatial(rng):
            label = rng.choice(labels)
            v, cat = rng.choice(attrs)
            rname, tgt = rng.choice(spatial_rels)
            return [HasLabel(label), HasAttribute(v, cat), HasRelationTo(rname, tgt)]

        generators.append(gen_label_attr_spatial)

    if labels and attrs and support_rels:

        def gen_label_attr_support(rng):
            label = rng.choice(labels)
            v, cat = rng.choice(attrs)
            rname, tgt = rng.choice(support_rels)
            return [HasLabel(label), HasAttribute(v, cat), HasRelationTo(rname, tgt)]

        generators.append(gen_label_attr_support)

    if attrs and spatial_rels:

        def gen_not_attr_spatial(rng):
            v, cat = rng.choice(attrs)
            rname, tgt = rng.choice(spatial_rels)
            return [NotPredicate(HasAttribute(v, cat)), HasRelationTo(rname, tgt)]

        generators.append(gen_not_attr_spatial)

    if spatial_rels and support_rels:

        def gen_spatial_support(rng):
            r1 = rng.choice(spatial_rels)
            r2 = rng.choice(support_rels)
            return [HasRelationTo(r1[0], r1[1]), HasRelationTo(r2[0], r2[1])]

        generators.append(gen_spatial_support)

    if attrs and spatial_rels and support_rels:

        def gen_triple(rng):
            v, cat = rng.choice(attrs)
            r1 = rng.choice(spatial_rels)
            r2 = rng.choice(support_rels)
            return [
                HasAttribute(v, cat),
                HasRelationTo(r1[0], r1[1]),
                HasRelationTo(r2[0], r2[1]),
            ]

        generators.append(gen_triple)

    if attrs and spatial_rels:

        def gen_or_attr_spatial(rng):
            v, cat = rng.choice(attrs)
            rname, tgt = rng.choice(spatial_rels)
            return [OrPredicate(HasAttribute(v, cat), HasRelationTo(rname, tgt))]

        generators.append(gen_or_attr_spatial)

    if labels and attrs and spatial_rels:

        def gen_label_attr_not_spatial(rng):
            label = rng.choice(labels)
            v, cat = rng.choice(attrs)
            rname, tgt = rng.choice(spatial_rels)
            return [
                HasLabel(label),
                HasAttribute(v, cat),
                NotPredicate(HasRelationTo(rname, tgt)),
            ]

        generators.append(gen_label_attr_not_spatial)

    if affordances and spatial_rels:

        def gen_affordance_spatial(rng):
            aff = rng.choice(affordances)
            rname, tgt = rng.choice(spatial_rels)
            return [HasAffordance(aff), HasRelationTo(rname, tgt)]

        generators.append(gen_affordance_spatial)

    if affordances and support_rels:

        def gen_affordance_support(rng):
            aff = rng.choice(affordances)
            rname, tgt = rng.choice(support_rels)
            return [HasAffordance(aff), HasRelationTo(rname, tgt)]

        generators.append(gen_affordance_support)

    if labels and affordances and spatial_rels:

        def gen_label_affordance_spatial(rng):
            label = rng.choice(labels)
            aff = rng.choice(affordances)
            rname, tgt = rng.choice(spatial_rels)
            return [HasLabel(label), HasAffordance(aff), HasRelationTo(rname, tgt)]

        generators.append(gen_label_affordance_spatial)

    if affordances and attrs and spatial_rels:

        def gen_affordance_attr_spatial(rng):
            aff = rng.choice(affordances)
            v, cat = rng.choice(attrs)
            rname, tgt = rng.choice(spatial_rels)
            return [HasAffordance(aff), HasAttribute(v, cat), HasRelationTo(rname, tgt)]

        generators.append(gen_affordance_attr_spatial)

    if not generators:
        return []

    qs = _sample_and_try(rng, generators, obj_by_id, outgoing, incoming, count)
    for q in qs:
        q["type"] = "compound"
    return qs


def deduplicate_questions(questions: List[Dict]) -> List[Dict]:
    seen = set()
    result = []
    for q in questions:
        if q["question"] not in seen:
            seen.add(q["question"])
            result.append(q)
    return result


def generate_questions_for_scene(sg: Dict, rng: random.Random) -> List[Dict]:
    obj_by_id, outgoing, incoming = build_indices(sg)

    if len(obj_by_id) < 3:
        return []

    attrs = collect_attrs(obj_by_id)
    labels = collect_labels(obj_by_id)
    affordances = collect_affordances(obj_by_id)
    spatial_rels, support_rels, semantic_rels = collect_rels_by_type(
        outgoing, obj_by_id
    )

    target = MAX_QUESTIONS_PER_SCENE // 4

    semantic_qs = generate_semantic_questions(
        obj_by_id,
        outgoing,
        incoming,
        rng,
        target,
        attrs,
        labels,
        semantic_rels,
        affordances,
    )
    spatial_qs = generate_spatial_questions(
        obj_by_id, outgoing, incoming, rng, target, spatial_rels, labels
    )
    support_qs = generate_support_questions(
        obj_by_id, outgoing, incoming, rng, target, support_rels, labels
    )
    compound_qs = generate_compound_questions(
        obj_by_id,
        outgoing,
        incoming,
        rng,
        target,
        attrs,
        labels,
        spatial_rels,
        support_rels,
        semantic_rels,
        affordances,
    )

    all_qs = semantic_qs + spatial_qs + support_qs + compound_qs
    all_qs = deduplicate_questions(all_qs)
    rng.shuffle(all_qs)
    return all_qs[:MAX_QUESTIONS_PER_SCENE]


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
