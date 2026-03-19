"""PyTorch Dataset for AudioSet features loaded from JSONL manifests."""

import os

import torch
from torch.utils.data import Dataset

from audioset_classification.data.manifest import read_manifest


class AudioSetDataset(Dataset):
    """Dataset reading entries from a JSONL manifest and loading .pt features.

    Each .pt file is a dict with keys 'x' (log-mel tensor) and 'label_ids'
    (multi-hot float32 vector). Synthetic mode returns zero tensors for testing.
    """

    def __init__(
        self,
        manifest_path: str,
        features_dir: str,
        num_classes: int = 527,
        n_mels: int = 128,
        synthetic: bool = False,
    ):
        self.entries = read_manifest(manifest_path)
        self.features_dir = features_dir
        self.num_classes = num_classes
        self.n_mels = n_mels
        self.synthetic = synthetic

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        entry = self.entries[idx]

        if self.synthetic:
            x = torch.zeros(self.n_mels, 100, dtype=torch.float32)
            labels = torch.zeros(self.num_classes, dtype=torch.float32)
            for label_id in entry.get("label_ids", []):
                if label_id < self.num_classes:
                    labels[label_id] = 1.0
            return x, labels

        stem = os.path.splitext(os.path.basename(entry["audio_path"]))[0]
        pt_path = os.path.join(self.features_dir, f"{stem}.pt")
        data: dict[str, torch.Tensor] = torch.load(
            pt_path, map_location="cpu", weights_only=True
        )
        return data["x"], data["label_ids"]
