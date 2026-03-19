"""CLI for training."""

import lightning as L
import typer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from loguru import logger

from audioset_classification.cli.data_cli import (
    FEATURES_DIR,
    MANIFESTS_DIR,
    NUM_CLASSES,
)
from audioset_classification.data.data_module import AudioSetDataModule
from audioset_classification.lightning.module import AudioSetLightningModule

TRAINING_OUTPUTS = "training-outputs"


def run_train(
    manifests_dir: str = typer.Option(
        MANIFESTS_DIR,
        "--manifests-dir",
        help="Directory containing train/val/test.jsonl",
    ),
    features_dir: str = typer.Option(
        FEATURES_DIR, "--features-dir", help="Directory containing .pt feature files"
    ),
    num_classes: int = typer.Option(NUM_CLASSES, help="Number of output classes"),
    batch_size: int = typer.Option(32, help="Batch size"),
    max_epochs: int = typer.Option(10, help="Max training epochs"),
    num_workers: int = typer.Option(4, help="DataLoader worker count"),
    synthetic: bool = typer.Option(False, help="Use synthetic zero tensors"),
) -> None:
    """Run training from JSONL manifests and precomputed .pt features."""
    datamodule = AudioSetDataModule(
        manifests_dir=manifests_dir,
        features_dir=features_dir,
        num_classes=num_classes,
        batch_size=batch_size,
        num_workers=num_workers,
        synthetic=synthetic,
    )
    model = AudioSetLightningModule(num_classes=num_classes)

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        default_root_dir=TRAINING_OUTPUTS,
        logger=TensorBoardLogger(save_dir=TRAINING_OUTPUTS),
        callbacks=[
            ModelCheckpoint(
                monitor="val/loss",
                mode="min",
                save_top_k=3,
                save_last=True,
                filename="epoch-{epoch:04d}-val_loss-{val/loss:.4f}",
            ),
        ],
    )
    logger.info("Starting training")
    trainer.fit(model, datamodule=datamodule)
    logger.info("Training complete")
