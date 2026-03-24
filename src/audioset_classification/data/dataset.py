"""PyTorch Dataset for CLAP-aligned features loaded from JSONL manifests."""

import os

import torch
from torch.utils.data import Dataset

from audioset_classification.data.manifest import read_manifest


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
        self.entries = read_manifest(manifest_path)
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

        stem = os.path.splitext(os.path.basename(entry["audio_path"]))[0]
        pt_path = os.path.join(self.features_dir, f"{stem}.pt")
        data: dict[str, torch.Tensor] = torch.load(
            pt_path, map_location="cpu", weights_only=True
        )
        return data["input_features"], data["is_longer"], data["label_ids"]
