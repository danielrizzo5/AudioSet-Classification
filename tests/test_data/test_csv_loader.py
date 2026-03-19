"""Tests for csv_loader."""

from audioset_classification.data.csv_loader import (
    load_segments_csv,
    parse_positive_labels,
    segment_key,
)


def test_load_segments_csv(mock_audioset_dir):
    """Load CSV and verify columns and row count."""
    path = mock_audioset_dir / "balanced_train_segments.csv"
    df = load_segments_csv(str(path), split="balanced_train")
    assert len(df) == 3
    assert list(df.columns) == [
        "ytid",
        "start_seconds",
        "end_seconds",
        "positive_labels",
        "split",
    ]
    assert df.iloc[0]["ytid"] == "-0RWZT-miFs"
    assert df.iloc[0]["start_seconds"] == 420.0
    assert df.iloc[0]["end_seconds"] == 430.0


def test_load_segments_csv_with_max_segments(mock_audioset_dir):
    """Limit segments via max_segments."""
    path = mock_audioset_dir / "balanced_train_segments.csv"
    df = load_segments_csv(str(path), max_segments=2)
    assert len(df) == 2


def test_parse_positive_labels():
    """Parse label string into list of mids."""
    assert parse_positive_labels('"/m/03v3yw,/m/0k4j"') == ["/m/03v3yw", "/m/0k4j"]
    assert parse_positive_labels("/m/09x0r") == ["/m/09x0r"]
    assert parse_positive_labels("") == []
    assert parse_positive_labels("  ") == []


def test_segment_key():
    """Build unique segment key."""
    assert segment_key("abc", 0.0, 10.0) == "abc_0.000_10.000"
    assert segment_key("xyz", 420.5, 430.5) == "xyz_420.500_430.500"
