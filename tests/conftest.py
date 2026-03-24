"""Test suite setup."""

import json
import os

os.environ.setdefault("ENV", "test")

import numpy as np
import pytest
import torch


@pytest.fixture
def random_seed():
    """Set consistent random seed for reproducible tests."""
    np.random.seed(42)
    return 42


@pytest.fixture
def mock_audioset_dir(tmp_path):
    """Create minimal AudioSet-like directory with CSV and ontology."""
    csv_content = """# Segments csv created Sun Mar  5 10:54:25 2017
# num_ytids=3, num_segs=3, num_unique_labels=3, num_positive_labels=6
# YTID, start_seconds, end_seconds, positive_labels
-0RWZT-miFs,420.000,430.000,"/m/03v3yw,/m/0k4j"
abc123xyz,0.000,10.000,"/m/09x0r"
def456uvw,100.000,110.000,"/m/03v3yw,/m/09x0r,/m/0k4j"
"""
    (tmp_path / "balanced_train_segments.csv").write_text(csv_content)

    ontology_content = """index,mid,display_name
0,/m/09x0r,Speech
1,/m/03v3yw,Keys jangling
2,/m/0k4j,Car
"""
    (tmp_path / "class_labels_indices.csv").write_text(ontology_content)

    return tmp_path


@pytest.fixture
def mock_manifest_entries():
    """Return a list of manifest entry dicts for 3 fake clips."""
    return [
        {
            "audio_path": "dev-data/audio/abc_0.000_10.000.wav",
            "labels": ["Speech"],
            "label_ids": [0],
            "ytid": "abc",
            "start": 0.0,
            "end": 10.0,
        },
        {
            "audio_path": "dev-data/audio/def_100.000_110.000.wav",
            "labels": ["Keys jangling", "Car"],
            "label_ids": [1, 2],
            "ytid": "def",
            "start": 100.0,
            "end": 110.0,
        },
        {
            "audio_path": "dev-data/audio/ghi_30.000_40.000.wav",
            "labels": ["Car"],
            "label_ids": [2],
            "ytid": "ghi",
            "start": 30.0,
            "end": 40.0,
        },
    ]


@pytest.fixture
def mock_manifests_dir(tmp_path, mock_manifest_entries):
    """Write train.jsonl, val.jsonl, test.jsonl into a temp directory."""
    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir()
    for split in ("train", "val", "test"):
        path = manifests_dir / f"{split}.jsonl"
        with open(path, "w") as f:
            for entry in mock_manifest_entries:
                f.write(json.dumps(entry) + "\n")
    return manifests_dir


@pytest.fixture
def mock_features_dir(tmp_path, mock_manifest_entries):
    """Write CLAP-shaped .pt feature files for all manifest entries."""
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    for entry in mock_manifest_entries:
        stem = os.path.splitext(os.path.basename(entry["audio_path"]))[0]
        label_vec = torch.zeros(3, dtype=torch.float32)
        for idx in entry["label_ids"]:
            label_vec[idx] = 1.0
        data = {
            "input_features": torch.randn(4, 48, 64, dtype=torch.float32),
            "is_longer": torch.zeros(1, 1, dtype=torch.bool),
            "label_ids": label_vec,
        }
        torch.save(data, features_dir / f"{stem}.pt")
    return features_dir
