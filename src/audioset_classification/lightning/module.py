"""LightningModule for AudioSet multi-label classification with CLAP backbone."""

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from transformers import ClapModel

from audioset_classification.lightning.lr_schedulers import HeadGroupCosineAnnealingLR
from audioset_classification.models.clap_audio_backbone import ClapAudioBackbone
from audioset_classification.models.classifier import AudioSetProjectionHead


class AudioSetLightningModule(L.LightningModule):
    """CLAP audio encoder (as backbone) + projection head; BCEWithLogitsLoss.

    The projection head uses cosine annealing over ``max_epochs`` (``HeadGroupCosineAnnealingLR``),
    with ``cosine_eta_min`` defaulting to ``1e-6`` so the head lr does not reach exactly zero.
    Use with ``BackboneFinetuning``: optimize ``head`` only at first, then unfreeze
    ``backbone`` at ``unfreeze_backbone_at_epoch``; backbone param groups are not cosine-annealed.
    """

    def __init__(
        self,
        clap_model_id: str,
        num_classes: int = 527,
        max_epochs: int = 10,
        head_hidden_dim: int = 512,
        head_dropout: float = 0.3,
        learning_rate: float = 1e-3,
        cosine_eta_min: float = 1e-6,
    ):
        super().__init__()
        self.save_hyperparameters()
        clap = ClapModel.from_pretrained(clap_model_id)
        embed_dim = clap.config.audio_config.hidden_size
        self.backbone = ClapAudioBackbone(clap.audio_model)
        self.head = AudioSetProjectionHead(
            input_dim=embed_dim,
            num_classes=num_classes,
            hidden_dim=head_hidden_dim,
            dropout=head_dropout,
        )
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.cosine_eta_min = cosine_eta_min

    def forward(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """batch is (input_features, is_longer) from a step; returns logits."""
        input_features, is_longer = batch
        return self.head(self.backbone((input_features, is_longer)))

    def _step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        stage: str,
    ) -> torch.Tensor:
        input_features, is_longer, y = batch
        logits = self.head(self.backbone((input_features, is_longer)))
        loss = F.binary_cross_entropy_with_logits(logits, y)
        self.log(f"{stage}/loss", loss, on_step=(stage == "train"), on_epoch=True)
        return loss

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        return self._step(batch, "val")

    def test_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        return self._step(batch, "test")

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.head.parameters(), lr=self.learning_rate)
        scheduler = HeadGroupCosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs,
            eta_min=self.cosine_eta_min,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
