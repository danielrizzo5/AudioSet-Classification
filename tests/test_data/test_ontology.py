"""Tests for ontology."""

from audioset_classification.data.ontology import (
    index_to_mid,
    load_ontology,
    mid_to_index,
)


def test_load_ontology(mock_audioset_dir):
    """Load ontology and verify structure."""
    path = mock_audioset_dir / "class_labels_indices.csv"
    df = load_ontology(str(path))
    assert len(df) == 3
    assert list(df.columns) == ["index", "mid", "display_name"]
    assert df.iloc[0]["mid"] == "/m/09x0r"
    assert df.iloc[0]["display_name"] == "Speech"


def test_mid_to_index(mock_audioset_dir):
    """Build mid to index mapping."""
    path = mock_audioset_dir / "class_labels_indices.csv"
    ontology = load_ontology(str(path))
    mapping = mid_to_index(ontology)
    assert mapping["/m/09x0r"] == 0
    assert mapping["/m/0k4j"] == 2


def test_index_to_mid(mock_audioset_dir):
    """Build index to mid mapping."""
    path = mock_audioset_dir / "class_labels_indices.csv"
    ontology = load_ontology(str(path))
    mapping = index_to_mid(ontology)
    assert mapping[0] == "/m/09x0r"
    assert mapping[2] == "/m/0k4j"
