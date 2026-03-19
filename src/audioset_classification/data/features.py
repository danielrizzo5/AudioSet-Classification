"""Compute and save log-mel spectrogram features from audio clips."""

import os

import torch
import torchaudio
import torchaudio.transforms as T
from loguru import logger

from audioset_classification.data.manifest import read_manifest

N_MELS = 128
HOP_LENGTH = 160  # 10 ms at 16kHz
WIN_LENGTH = 400  # 25 ms at 16kHz
N_FFT = 512
SAMPLE_RATE = 16000


def feature_path(audio_path: str, features_dir: str) -> str:
    """Return the .pt path for a given audio file."""
    stem = os.path.splitext(os.path.basename(audio_path))[0]
    return os.path.join(features_dir, f"{stem}.pt")


def compute_log_mel(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Compute log-mel spectrogram from a waveform tensor.

    Args:
        waveform: [channels, time] float tensor.
        sample_rate: Sample rate of the waveform.

    Returns:
        Log-mel tensor of shape [n_mels, time_frames].
    """
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != SAMPLE_RATE:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)

    mel_transform = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    mel = mel_transform(waveform)  # [1, n_mels, time]
    log_mel = torch.log(mel + 1e-9).squeeze(0)  # [n_mels, time]
    return log_mel


def compute_features_for_clip(
    audio_path: str,
    label_ids: list[int],
    features_dir: str,
    num_classes: int,
) -> str | None:
    """Compute and save log-mel features for one clip.

    Returns output .pt path on success, None on failure.

    Args:
        audio_path: Path to the WAV file.
        label_ids: List of integer class indices.
        features_dir: Directory to write .pt files.
        num_classes: Total number of classes for the label vector.
    """
    out_path = feature_path(audio_path, features_dir)
    if os.path.exists(out_path):
        return out_path

    waveform, sr = torchaudio.load(audio_path)
    log_mel = compute_log_mel(waveform, sr)

    label_vec = torch.zeros(num_classes, dtype=torch.float32)
    for idx in label_ids:
        if idx < num_classes:
            label_vec[idx] = 1.0

    torch.save({"x": log_mel, "label_ids": label_vec}, out_path)
    return out_path


def compute_features(
    manifest_path: str,
    features_dir: str,
    num_classes: int,
) -> int:
    """Compute log-mel features for all clips in a manifest.

    Writes one .pt file per clip to features_dir. Returns the number of
    files written.

    Args:
        manifest_path: Path to a JSONL manifest.
        features_dir: Directory to write .pt feature files.
        num_classes: Total number of AudioSet classes.
    """
    os.makedirs(features_dir, exist_ok=True)
    entries = read_manifest(manifest_path)

    written = 0
    for i, entry in enumerate(entries):
        logger.info(f"[{i + 1}/{len(entries)}] {entry['ytid']}")
        result = compute_features_for_clip(
            audio_path=entry["audio_path"],
            label_ids=entry["label_ids"],
            features_dir=features_dir,
            num_classes=num_classes,
        )
        if result is not None:
            written += 1
        else:
            logger.warning(f"Failed to compute features for {entry['audio_path']}")

    return written
