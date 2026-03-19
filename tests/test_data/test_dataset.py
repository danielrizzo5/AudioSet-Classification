"""Tests for AudioSetDataset."""

import torch

from audioset_classification.data.dataset import AudioSetDataset


def test_audioset_dataset_synthetic(mock_manifests_dir, mock_features_dir):
    """Dataset returns zero tensors in synthetic mode without loading .pt files."""
    dataset = AudioSetDataset(
        manifest_path=str(mock_manifests_dir / "train.jsonl"),
        features_dir=str(mock_features_dir),
        num_classes=3,
        n_mels=128,
        synthetic=True,
    )
    assert len(dataset) == 3

    x, labels = dataset[0]
    assert x.shape == (128, 100)
    assert labels.shape == (3,)
    assert x.dtype == torch.float32
    assert labels.dtype == torch.float32
    assert labels.sum() >= 1


def test_audioset_dataset_pt_files(mock_manifests_dir, mock_features_dir):
    """Dataset loads log-mel tensors and label vectors from .pt files."""
    dataset = AudioSetDataset(
        manifest_path=str(mock_manifests_dir / "train.jsonl"),
        features_dir=str(mock_features_dir),
        num_classes=3,
        n_mels=128,
    )
    assert len(dataset) == 3

    x, labels = dataset[0]
    assert x.shape == (128, 100)
    assert labels.shape == (3,)
    assert x.dtype == torch.float32
    assert labels.dtype == torch.float32
    assert labels[0] == 1.0
