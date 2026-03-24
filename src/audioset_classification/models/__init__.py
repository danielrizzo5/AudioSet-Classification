"""Classification models."""

from audioset_classification.models.clap_audio_backbone import ClapAudioBackbone
from audioset_classification.models.classifier import AudioSetProjectionHead

__all__ = ["AudioSetProjectionHead", "ClapAudioBackbone"]
