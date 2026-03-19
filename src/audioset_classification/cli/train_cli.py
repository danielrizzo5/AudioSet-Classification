"""CLI for training."""

import lightning as L
import typer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from loguru import logger

from audioset_classification.data.data_module import AudioSetDataModule
from audioset_classification.data.ontology import load_ontology
from audioset_classification.lightning.module import AudioSetLightningModule

TRAINING_OUTPUTS = "training-outputs"


def run_train(
    data_dir: str = typer.Option(..., "--data-dir", "-d", help="Path to AudioSet data"),
    split: str = typer.Option(
        "balanced_train", help="Split: eval, balanced_train, unbalanced_train"
    ),
    batch_size: int = typer.Option(32, help="Batch size"),
    max_epochs: int = typer.Option(10, help="Max training epochs"),
    max_segments: int | None = typer.Option(
        None, help="Limit segments for fast iteration"
    ),
    synthetic: bool = typer.Option(False, help="Use synthetic random embeddings"),
    num_classes: int | None = typer.Option(
        None, help="Number of classes (default: all 527)"
    ),
) -> None:
    """Run training."""
    ontology_path = f"{data_dir}/class_labels_indices.csv"
    ontology = load_ontology(ontology_path)
    n_classes = num_classes if num_classes is not None else len(ontology)

    datamodule = AudioSetDataModule(
        data_dir=data_dir,
        ontology_path=ontology_path,
        split=split,
        batch_size=batch_size,
        max_segments=max_segments,
        num_classes=n_classes,
        synthetic=synthetic,
    )
    model = AudioSetLightningModule(num_classes=n_classes)

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
