"""Tests for train-manifest BCE pos_weight helpers."""

import json

import numpy as np
import pytest
import torch

from audioset_classification.data.class_weights import (
    bce_pos_weight_from_train_manifest,
    count_positive_labels_per_class,
)


def test_count_positive_labels_per_class(tmp_path):
    m = tmp_path / "train.jsonl"
    lines = [
        {"label_ids": [0, 1]},
        {"label_ids": [0]},
        {"label_ids": [1]},
    ]
    m.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    c = count_positive_labels_per_class(str(m), num_classes=2)
    np.testing.assert_array_equal(c, np.array([2, 2], dtype=np.int64))


def test_bce_pos_weight_formula(tmp_path):
    """Two classes; pos counts 2 and 1 over N=3 clips; alpha=0.5."""
    m = tmp_path / "train.jsonl"
    lines = [
        {"label_ids": [0]},
        {"label_ids": [0]},
        {"label_ids": [1]},
    ]
    m.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    w = bce_pos_weight_from_train_manifest(str(m), num_classes=2, alpha=0.5)
    assert w.shape == (2,)
    exp0 = float(np.power((3 - 2) / 2, 0.5))
    exp1 = float(np.power((3 - 1) / 1, 0.5))
    torch.testing.assert_close(w[0], torch.tensor(exp0))
    torch.testing.assert_close(w[1], torch.tensor(exp1))


def test_bce_pos_weight_errors_when_class_never_positive(tmp_path):
    m = tmp_path / "train.jsonl"
    m.write_text(json.dumps({"label_ids": [0]}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No positive train"):
        bce_pos_weight_from_train_manifest(str(m), num_classes=2, alpha=0.5)
