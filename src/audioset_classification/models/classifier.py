"""Simple classifier head for 128-dim AudioSet embeddings."""

import torch
import torch.nn as nn


class AudioSetClassifier(nn.Module):
    """MLP classifier for multi-label AudioSet classification."""

    def __init__(
        self,
        input_dim: int = 128,
        num_classes: int = 527,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        dims = hidden_dims or [512, 256]
        layers = []
        prev = input_dim
        for d in dims:
            layers.extend(
                [
                    nn.Linear(prev, d),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev = d
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))
