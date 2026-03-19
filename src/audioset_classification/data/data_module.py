"""Lightning DataModule for AudioSet."""

from typing import Optional

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from audioset_classification.data.csv_loader import load_segments_csv
from audioset_classification.data.dataset import AudioSetDataset


class AudioSetDataModule(L.LightningDataModule):
    """PyTorch Lightning DataModule for AudioSet classification."""

    def __init__(
        self,
        data_dir: str,
        ontology_path: str,
        split: str = "balanced_train",
        batch_size: int = 32,
        train_split: float = 0.8,
        num_workers: int = 4,
        max_segments: int | None = None,
        num_classes: int | None = None,
        synthetic: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.ontology_path = ontology_path
        self.split = split
        self.batch_size = batch_size
        self.train_split = train_split
        self.num_workers = num_workers
        self.max_segments = max_segments
        self.num_classes = num_classes
        self.synthetic = synthetic

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] | None = None

    def setup(self, stage: Optional[str] = None) -> None:
        csv_name = f"{self.split}_segments.csv"
        csv_path = f"{self.data_dir}/{csv_name}"
        segments_df = load_segments_csv(
            csv_path, split=self.split, max_segments=self.max_segments
        )

        features_dir = None if self.synthetic else f"{self.data_dir}/features"
        full_dataset = AudioSetDataset(
            segments_df=segments_df,
            ontology_path=self.ontology_path,
            features_dir=features_dir,
            num_classes=self.num_classes,
            synthetic=self.synthetic,
        )

        total = len(full_dataset)
        train_size = max(1, int(self.train_split * total))
        remaining = total - train_size
        val_size = max(1, int(remaining * 0.5))
        test_size = remaining - val_size

        if test_size <= 0 and total >= 3:
            test_size = 1
            val_size = remaining - test_size

        if test_size > 0:
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                full_dataset,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42),
            )
        else:
            self.train_dataset, self.val_dataset = random_split(
                full_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )
            self.test_dataset = None

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Call setup() first.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Call setup() first.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader | None:
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )
