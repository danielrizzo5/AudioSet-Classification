"""Test suite setup."""

import os

os.environ.setdefault("ENV", "test")

import numpy as np
import pytest


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
