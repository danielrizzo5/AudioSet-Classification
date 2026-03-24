"""LightningModule for AudioSet multi-label classification with CLAP backbone."""

import lightning as L
import torch
import torch.nn.functional as F
from transformers import ClapModel

from audioset_classification.models.clap_audio_backbone import ClapAudioBackbone
from audioset_classification.models.classifier import AudioSetProjectionHead


class AudioSetLightningModule(L.LightningModule):
    """CLAP audio encoder (as backbone) + projection head; BCEWithLogitsLoss.

    Use with ``BackboneFinetuning``: optimize ``head`` only at first, then unfreeze
    ``backbone`` at ``unfreeze_backbone_at_epoch``.
    """

    def __init__(
        self,
        clap_model_id: str,
        num_classes: int = 527,
        head_hidden_dim: int = 512,
        head_dropout: float = 0.3,
        learning_rate: float = 1e-3,
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

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.head.parameters(), lr=self.learning_rate)
