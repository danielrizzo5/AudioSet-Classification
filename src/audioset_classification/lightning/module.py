"""LightningModule for AudioSet multi-label classification."""

import lightning as L
import torch
import torch.nn.functional as F

from audioset_classification.models.classifier import AudioSetClassifier


class AudioSetLightningModule(L.LightningModule):
    """LightningModule for multi-label AudioSet classification with BCEWithLogitsLoss."""

    def __init__(
        self,
        num_classes: int = 527,
        input_dim: int = 128,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = AudioSetClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        self.learning_rate = learning_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _step(
        self, batch: tuple[torch.Tensor, torch.Tensor], stage: str
    ) -> torch.Tensor:
        x, y = batch
        logits = self.model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        self.log(f"{stage}/loss", loss, on_step=(stage == "train"), on_epoch=True)
        return loss

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._step(batch, "val")

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._step(batch, "test")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
