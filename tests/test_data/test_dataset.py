"""Tests for AudioSetDataset."""

import json

import pytest
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


def test_audioset_dataset_skips_rows_without_pt(tmp_path):
    """Rows without a matching .pt under features_dir are dropped."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    feat_dir = tmp_path / "features"
    feat_dir.mkdir()
    wav_a = audio_dir / "clip_a.wav"
    wav_b = audio_dir / "clip_b.wav"
    wav_a.write_text("x")
    wav_b.write_text("x")
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "audio_path": str(wav_a),
                        "label_ids": [0],
                        "ytid": "a",
                        "start": 0.0,
                        "end": 1.0,
                    }
                ),
                json.dumps(
                    {
                        "audio_path": str(wav_b),
                        "label_ids": [1],
                        "ytid": "b",
                        "start": 0.0,
                        "end": 1.0,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    torch.save(
        {
            "input_features": torch.zeros(4, 8, 64),
            "is_longer": torch.zeros(1, 1, dtype=torch.bool),
            "label_ids": torch.tensor([1.0, 0.0], dtype=torch.float32),
        },
        feat_dir / "clip_a.pt",
    )
    ds = AudioSetDataset(
        manifest_path=str(manifest),
        features_dir=str(feat_dir),
        num_classes=2,
    )
    assert len(ds) == 1
    _, _, y = ds[0]
    assert y[0] == 1.0


def test_audioset_dataset_raises_when_no_pt_files(tmp_path):
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "audio_path": str(tmp_path / "missing.wav"),
                "label_ids": [0],
                "ytid": "x",
                "start": 0.0,
                "end": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    feat_dir = tmp_path / "empty_features"
    feat_dir.mkdir()
    with pytest.raises(ValueError, match="No .pt features"):
        AudioSetDataset(
            manifest_path=str(manifest),
            features_dir=str(feat_dir),
            num_classes=1,
        )
