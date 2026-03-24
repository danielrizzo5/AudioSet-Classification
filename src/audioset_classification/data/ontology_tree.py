"""AudioSet hierarchy from ontology.json (child_ids → parent links)."""

import json
from typing import TypedDict

from typing_extensions import NotRequired


class OntologyNode(TypedDict):
    """Subset of fields used from AudioSet ontology.json objects."""

    id: str
    name: str
    child_ids: NotRequired[list[str]]


def load_ontology_nodes(path: str) -> list[OntologyNode]:
    """Load ontology.json as a list of node dicts."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return raw


def build_parent_map(nodes: list[OntologyNode]) -> dict[str, str]:
    """Map each child MID to its parent MID from ``child_ids`` on each node."""
    parent: dict[str, str] = {}
    for node in nodes:
        nid = str(node["id"])
        for cid in node.get("child_ids", []):
            parent[str(cid)] = nid
    return parent


def build_mid_to_name(nodes: list[OntologyNode]) -> dict[str, str]:
    """Map MID to ontology ``name`` string."""
    return {str(n["id"]): str(n["name"]) for n in nodes}


def mid_path_root_to_leaf(leaf_mid: str, parent_map: dict[str, str]) -> list[str]:
    """Return MIDs from root to ``leaf_mid`` (inclusive)."""
    chain_bottom_up: list[str] = [leaf_mid]
    cur = leaf_mid
    while cur in parent_map:
        cur = parent_map[cur]
        chain_bottom_up.append(cur)
    chain_bottom_up.reverse()
    return chain_bottom_up
