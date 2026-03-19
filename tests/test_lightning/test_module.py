"""Tests for LightningModule."""

import torch
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader, TensorDataset

from audioset_classification.lightning.module import AudioSetLightningModule


def test_audioset_lightning_module_forward():
    """Forward pass produces correct shape."""
    model = AudioSetLightningModule(num_classes=10, input_dim=128)
    x = torch.randn(4, 128)
    out = model(x)
    assert out.shape == (4, 10)


def test_audioset_lightning_module_training_step():
    """Training step returns loss when run through Trainer."""
    model = AudioSetLightningModule(num_classes=3)
    x = torch.randn(2, 128)
    y = torch.zeros(2, 3)
    y[0, 0] = 1
    y[1, 1] = 1
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=2, num_workers=0)
    trainer = Trainer(
        max_steps=1,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, train_dataloaders=dataloader, val_dataloaders=dataloader)
