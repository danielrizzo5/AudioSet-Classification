"""Tests for dataset."""

import torch

from audioset_classification.data.csv_loader import load_segments_csv, segment_key
from audioset_classification.data.dataset import AudioSetDataset


def test_audioset_dataset_synthetic(mock_audioset_dir):
    """Dataset yields (embedding, labels) with synthetic mode."""
    csv_path = mock_audioset_dir / "balanced_train_segments.csv"
    ontology_path = mock_audioset_dir / "class_labels_indices.csv"
    df = load_segments_csv(str(csv_path))

    dataset = AudioSetDataset(
        segments_df=df,
        ontology_path=str(ontology_path),
        num_classes=3,
        synthetic=True,
    )
    assert len(dataset) == 3

    emb, labels = dataset[0]
    assert emb.shape == (128,)
    assert labels.shape == (3,)
    assert emb.dtype == torch.float32
    assert labels.dtype == torch.float32
    assert labels.sum() >= 1


def test_audioset_dataset_pt_files(mock_audioset_dir, tmp_path):
    """Dataset loads embeddings from .pt files."""
    csv_path = mock_audioset_dir / "balanced_train_segments.csv"
    ontology_path = mock_audioset_dir / "class_labels_indices.csv"
    df = load_segments_csv(str(csv_path))

    features_dir = tmp_path / "features"
    for idx in range(len(df)):
        row = df.iloc[idx]
        ytid = str(row["ytid"])
        start = float(row["start_seconds"].item())
        end = float(row["end_seconds"].item())
        shard = ytid[:2] if len(ytid) >= 2 else "00"
        shard_dir = features_dir / shard
        shard_dir.mkdir(parents=True, exist_ok=True)
        key = segment_key(ytid, start, end)
        emb = torch.randn(128, dtype=torch.float32)
        torch.save(emb, shard_dir / f"{key}.pt")

    dataset = AudioSetDataset(
        segments_df=df,
        ontology_path=str(ontology_path),
        features_dir=str(features_dir),
        num_classes=3,
    )
    emb, labels = dataset[0]
    assert emb.shape == (128,)
    assert emb.dtype == torch.float32
