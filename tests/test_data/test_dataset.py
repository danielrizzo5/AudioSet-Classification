"""Tests for AudioSetDataset."""

import torch

from audioset_classification.data.dataset import AudioSetDataset


def test_audioset_dataset_synthetic(mock_manifests_dir, mock_features_dir):
    """Synthetic mode returns CLAP-shaped tensors without reading .pt files."""
    dataset = AudioSetDataset(
        manifest_path=str(mock_manifests_dir / "train.jsonl"),
        features_dir=str(mock_features_dir),
        num_classes=3,
        synthetic=True,
        synthetic_time=100,
        synthetic_mels=64,
        synthetic_channels=4,
    )
    assert len(dataset) == 3

    feats, is_longer, labels = dataset[0]
    assert feats.shape == (4, 100, 64)
    assert is_longer.shape == (1, 1)
    assert labels.shape == (3,)
    assert feats.dtype == torch.float32
    assert labels.dtype == torch.float32
    assert labels.sum() >= 1


def test_audioset_dataset_pt_files(mock_manifests_dir, mock_features_dir):
    """Dataset loads CLAP inputs and label vectors from .pt files."""
    dataset = AudioSetDataset(
        manifest_path=str(mock_manifests_dir / "train.jsonl"),
        features_dir=str(mock_features_dir),
        num_classes=3,
    )
    assert len(dataset) == 3

    feats, is_longer, labels = dataset[0]
    assert feats.shape == (4, 48, 64)
    assert is_longer.shape == (1, 1)
    assert labels.shape == (3,)
    assert labels[0] == 1.0
