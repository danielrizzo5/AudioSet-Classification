"""CLI for training."""

import os
from typing import Optional

import lightning as L
import typer
from lightning.pytorch.callbacks import BackboneFinetuning, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from loguru import logger

from audioset_classification.cli.data_cli import (
    DEFAULT_CLAP_MODEL,
    FEATURES_DIR,
    MANIFESTS_DIR,
    NUM_CLASSES,
)
from audioset_classification.data.class_weights import bce_pos_weight_from_entries
from audioset_classification.data.data_module import AudioSetDataModule
from audioset_classification.data.dataset import manifest_entries_with_features
from audioset_classification.data.manifest import read_manifest
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
    clap_model: str = typer.Option(
        DEFAULT_CLAP_MODEL,
        "--clap-model",
        help="HuggingFace model id for CLAP (must match feature precompute)",
    ),
    num_classes: int = typer.Option(NUM_CLASSES, help="Number of output classes"),
    batch_size: int = typer.Option(32, help="Batch size"),
    max_epochs: int = typer.Option(10, help="Max training epochs"),
    num_workers: int = typer.Option(4, help="DataLoader worker count"),
    synthetic: bool = typer.Option(
        False, help="Use synthetic random CLAP-like tensors"
    ),
    head_only: bool = typer.Option(
        False,
        "--head-only",
        help="Train the projection head only; do not register BackboneFinetuning (backbone stays out of the optimizer).",
    ),
    unfreeze_backbone_at_epoch: Optional[int] = typer.Option(
        None,
        "--unfreeze-backbone-at-epoch",
        help=(
            "Epoch at which BackboneFinetuning unfreezes the CLAP encoder (Lightning epoch index). "
            "Default when omitted: floor(0.9 * max_epochs), at least 1. Ignored with --head-only."
        ),
    ),
    use_bce_pos_weight: bool = typer.Option(
        True,
        "--bce-pos-weight/--no-bce-pos-weight",
        help=(
            "If set, BCE uses train-manifest ``pos_weight`` "
            "``((N-n_c)/n_c)**alpha`` per class (requires every class in train)."
        ),
    ),
    bce_pos_weight_alpha: float = typer.Option(
        0.5,
        "--bce-pos-weight-alpha",
        help="Tempering exponent for BCE pos_weight (0 => all ones).",
    ),
) -> None:
    """Train CLAP encoder + projection head; optional staged unfreeze via BackboneFinetuning."""
    datamodule = AudioSetDataModule(
        manifests_dir=manifests_dir,
        features_dir=features_dir,
        num_classes=num_classes,
        batch_size=batch_size,
        num_workers=num_workers,
        synthetic=synthetic,
    )
    bce_pw = None
    if use_bce_pos_weight:
        train_manifest = os.path.join(manifests_dir, "train.jsonl")
        train_entries = manifest_entries_with_features(
            read_manifest(train_manifest), features_dir
        )
        bce_pw = bce_pos_weight_from_entries(
            train_entries, num_classes, bce_pos_weight_alpha
        )
    model = AudioSetLightningModule(
        clap_model_id=clap_model,
        num_classes=num_classes,
        max_epochs=max_epochs,
        bce_pos_weight=bce_pw,
    )

    callbacks: list[L.Callback] = [
        ModelCheckpoint(
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            filename="epoch-{epoch:04d}-val_loss-{val/loss:.4f}",
        ),
    ]
    if head_only:
        logger.info("Starting training (head only; CLAP backbone not in optimizer)")
    else:
        unfreeze_at = (
            unfreeze_backbone_at_epoch
            if unfreeze_backbone_at_epoch is not None
            else max(1, int(0.9 * max_epochs))
        )
        callbacks.append(
            BackboneFinetuning(
                unfreeze_backbone_at_epoch=unfreeze_at,
                lambda_func=lambda _epoch: 1.0,
                backbone_initial_ratio_lr=0.1,
                train_bn=True,
            )
        )
        logger.info(
            f"Starting training (backbone frozen until epoch {unfreeze_at} of {max_epochs})"
        )

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        default_root_dir=TRAINING_OUTPUTS,
        logger=TensorBoardLogger(save_dir=TRAINING_OUTPUTS, name=""),
        callbacks=callbacks,
    )
    trainer.fit(model, datamodule=datamodule)
    logger.info("Training complete")
