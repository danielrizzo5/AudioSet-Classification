"""Tests for CLAP feature extraction helpers."""

import wave

import torch
import torchaudio


def test_torchaudio_load_wav_mono(tmp_path):
    """torchaudio.load reads a stdlib-written mono WAV (TorchCodec backend)."""
    path = tmp_path / "clip.wav"
    n_frames = 80
    sample_rate = 16000
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames)
    waveform, sr = torchaudio.load(str(path))
    assert sr == sample_rate
    assert waveform.shape == (1, n_frames)
    assert waveform.dtype == torch.float32
