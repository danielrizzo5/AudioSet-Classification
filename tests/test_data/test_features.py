"""Tests for CLAP feature extraction helpers."""

import wave
from unittest import mock
from unittest.mock import MagicMock

import torch

from audioset_classification.data.features import compute_features_for_clip


def test_stdlib_wave_mono_roundtrip(tmp_path):
    """Stdlib ``wave`` writes mono PCM WAVs readable back (no TorchCodec in CI)."""
    path = tmp_path / "clip.wav"
    n_frames = 80
    sample_rate = 16000
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames)
    with wave.open(str(path), "rb") as wf:
        assert wf.getnframes() == n_frames
        assert wf.getframerate() == sample_rate
        assert wf.getnchannels() == 1


def test_compute_features_for_clip_skips_zero_sample_waveform(tmp_path):
    """No .pt when decode yields zero time samples (avoids empty Resample)."""
    path = tmp_path / "any.wav"
    path.touch()
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    with mock.patch(
        "audioset_classification.data.features.torchaudio.load",
        return_value=(torch.zeros(1, 0), 16000),
    ):
        out = compute_features_for_clip(
            str(path),
            label_ids=[0],
            features_dir=str(features_dir),
            num_classes=527,
            feature_extractor=MagicMock(),
        )
    assert out is None


def test_compute_features_for_clip_skips_torchaudio_runtime_error(tmp_path):
    """TorchCodec raises RuntimeError on some files; treat as skip."""
    path = tmp_path / "bad.wav"
    path.touch()
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    with mock.patch(
        "audioset_classification.data.features.torchaudio.load",
        side_effect=RuntimeError("Failed to decode audio samples"),
    ):
        out = compute_features_for_clip(
            str(path),
            label_ids=[0],
            features_dir=str(features_dir),
            num_classes=527,
            feature_extractor=MagicMock(),
        )
    assert out is None
