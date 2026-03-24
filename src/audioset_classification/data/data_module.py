"""Lightning DataModule for AudioSet."""

import os
from typing import Optional

import lightning as L
from torch.utils.data import DataLoader, Dataset

from audioset_classification.data.collate import collate_clap_batch
from audioset_classification.data.dataset import AudioSetDataset


class AudioSetDataModule(L.LightningDataModule):
    """Loads train/val/test manifests and CLAP .pt features with padded batches."""

    def __init__(
        self,
        manifests_dir: str,
        features_dir: str,
        num_classes: int = 527,
        batch_size: int = 32,
        num_workers: int = 4,
        synthetic: bool = False,
    ):
        super().__init__()
        self.manifests_dir = manifests_dir
        self.features_dir = features_dir
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.synthetic = synthetic

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def _manifest(self, split: str) -> str:
        return os.path.join(self.manifests_dir, f"{split}.jsonl")

    def _make_dataset(self, split: str) -> AudioSetDataset:
        return AudioSetDataset(
            manifest_path=self._manifest(split),
            features_dir=self.features_dir,
            num_classes=self.num_classes,
            synthetic=self.synthetic,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Load train/val datasets for fit, test dataset for test/predict."""
        if stage in ("fit", None):
            self.train_dataset = self._make_dataset("train")
            self.val_dataset = self._make_dataset("val")
        if stage in ("test", "predict", None):
            self.test_dataset = self._make_dataset("test")

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None, "Call setup('fit') first."
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_clap_batch,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None, "Call setup('fit') first."
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_clap_batch,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        assert self.test_dataset is not None, "Call setup('test') first."
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_clap_batch,
            persistent_workers=self.num_workers > 0,
        )
