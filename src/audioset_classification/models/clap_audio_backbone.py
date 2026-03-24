"""Wrap HuggingFace CLAP audio encoder for Lightning BackboneFinetuning."""

import torch
from torch import nn
from transformers import ClapAudioModel


class ClapAudioBackbone(nn.Module):
    """Runs CLAP audio_model and returns pooler_output [batch, hidden_size]."""

    def __init__(self, audio_model: ClapAudioModel):
        super().__init__()
        self.audio_model = audio_model

    def forward(self, x: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Args:
        x: (input_features, is_longer) with shapes [B, C, T, F] and [B, 1].
        """
        input_features, is_longer = x
        out = self.audio_model(
            input_features=input_features,
            is_longer=is_longer,
            return_dict=True,
        )
        return out.pooler_output
