"""Task head: projection MLP for multi-label classification."""

import torch
import torch.nn as nn


class AudioSetProjectionHead(nn.Module):
    """MLP from encoder embedding dim to num_classes logits."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
