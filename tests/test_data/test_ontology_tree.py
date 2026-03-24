"""Tests for AudioSet ontology.json hierarchy helpers."""

import json

from audioset_classification.data.ontology_tree import (
    build_mid_to_name,
    build_parent_map,
    load_ontology_nodes,
    mid_path_root_to_leaf,
)


def test_parent_map_and_root_to_leaf_path(tmp_path):
    """``child_ids`` yield parent links; path walks from root to leaf."""
    nodes = [
        {"id": "/r", "name": "root", "child_ids": ["/a"]},
        {"id": "/a", "name": "a", "child_ids": ["/b"]},
        {"id": "/b", "name": "leaf", "child_ids": []},
    ]
    path_json = tmp_path / "ontology.json"
    path_json.write_text(json.dumps(nodes), encoding="utf-8")
    loaded = load_ontology_nodes(str(path_json))
    parent_map = build_parent_map(loaded)
    assert parent_map["/b"] == "/a"
    assert parent_map["/a"] == "/r"
    names = build_mid_to_name(loaded)
    mid_path = mid_path_root_to_leaf("/b", parent_map)
    assert mid_path == ["/r", "/a", "/b"]
    assert [names[m] for m in mid_path] == ["root", "a", "leaf"]
