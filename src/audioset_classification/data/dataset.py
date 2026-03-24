"""PyTorch Dataset for CLAP-aligned features loaded from JSONL manifests."""

import os

import torch
from loguru import logger
from torch.utils.data import Dataset

from audioset_classification.data.features import feature_path
from audioset_classification.data.manifest import read_manifest


def manifest_entries_with_features(
    entries: list[dict], features_dir: str
) -> list[dict]:
    """Manifest rows whose ``feature_path(audio_path, features_dir)`` exists."""
    return [
        e
        for e in entries
        if os.path.isfile(feature_path(e["audio_path"], features_dir))
    ]


class AudioSetDataset(Dataset):
    """Dataset reading manifests and loading per-clip .pt CLAP inputs.

    Each .pt file has input_features [C, T, F], is_longer [1, 1], label_ids [num_classes].
    Synthetic mode returns random small tensors for tests.
    """

    def __init__(
        self,
        manifest_path: str,
        features_dir: str,
        num_classes: int = 527,
        synthetic: bool = False,
        synthetic_time: int = 32,
        synthetic_mels: int = 64,
        synthetic_channels: int = 4,
    ):
        raw_entries = read_manifest(manifest_path)
        if synthetic:
            self.entries = raw_entries
        else:
            self.entries = manifest_entries_with_features(raw_entries, features_dir)
            n_skip = len(raw_entries) - len(self.entries)
            if n_skip:
                logger.warning(
                    f"Skipped {n_skip} manifest row(s) missing .pt under {features_dir!r} "
                    f"({manifest_path!r})"
                )
            if not self.entries:
                raise ValueError(
                    f"No .pt features for any row in {manifest_path!r} under {features_dir!r}"
                )
        self.features_dir = features_dir
        self.num_classes = num_classes
        self.synthetic = synthetic
        self.synthetic_time = synthetic_time
        self.synthetic_mels = synthetic_mels
        self.synthetic_channels = synthetic_channels

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        entry = self.entries[idx]

        if self.synthetic:
            input_features = torch.randn(
                self.synthetic_channels,
                self.synthetic_time,
                self.synthetic_mels,
                dtype=torch.float32,
            )
            is_longer = torch.zeros(1, 1, dtype=torch.bool)
            labels = torch.zeros(self.num_classes, dtype=torch.float32)
            for label_id in entry.get("label_ids", []):
                if label_id < self.num_classes:
                    labels[label_id] = 1.0
            return input_features, is_longer, labels

        pt_path = feature_path(entry["audio_path"], self.features_dir)
        data: dict[str, torch.Tensor] = torch.load(
            pt_path, map_location="cpu", weights_only=True
        )
        return data["input_features"], data["is_longer"], data["label_ids"]
