"""Tests for LightningModule."""

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import BackboneFinetuning
from torch.utils.data import DataLoader, Dataset

from audioset_classification.data.collate import collate_clap_batch
from audioset_classification.lightning.module import AudioSetLightningModule

CLAP_MODEL_ID = "laion/clap-htsat-fused"


class _ClapBatchDataset(Dataset):
    """Minimal dataset yielding CLAP-shaped single samples."""

    def __init__(self, n: int, num_classes: int) -> None:
        self.n = n
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = 40 + (idx % 5)
        feats = torch.randn(4, t, 64)
        is_longer = torch.zeros(1, 1, dtype=torch.bool)
        y = torch.zeros(self.num_classes, dtype=torch.float32)
        y[idx % self.num_classes] = 1.0
        return feats, is_longer, y


def test_audioset_lightning_module_forward():
    """Forward on (input_features, is_longer) yields logits [B, num_classes]."""
    model = AudioSetLightningModule(
        clap_model_id=CLAP_MODEL_ID,
        num_classes=10,
    )
    b = 2
    feats = torch.randn(b, 4, 50, 64)
    longer = torch.zeros(b, 1, dtype=torch.bool)
    out = model((feats, longer))
    assert out.shape == (b, 10)


def test_audioset_lightning_module_training_step():
    """Training step runs under Trainer with BackboneFinetuning."""
    model = AudioSetLightningModule(
        clap_model_id=CLAP_MODEL_ID,
        num_classes=3,
    )
    ds = _ClapBatchDataset(4, num_classes=3)
    dataloader = DataLoader(
        ds,
        batch_size=2,
        num_workers=0,
        collate_fn=collate_clap_batch,
    )
    finetune = BackboneFinetuning(
        unfreeze_backbone_at_epoch=9,
        lambda_func=lambda _e: 1.0,
        backbone_initial_ratio_lr=0.1,
    )
    trainer = Trainer(
        max_steps=1,
        logger=False,
        enable_checkpointing=False,
        callbacks=[finetune],
    )
    trainer.fit(
        model,
        train_dataloaders=dataloader,
        val_dataloaders=dataloader,
    )
