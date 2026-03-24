"""Precompute CLAP-aligned audio features for training."""

import os

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from loguru import logger
from transformers import ClapFeatureExtractor

from audioset_classification.data.manifest import read_manifest

CLAP_SAMPLE_RATE = 48000


def feature_path(audio_path: str, features_dir: str) -> str:
    """Return the .pt path for a given audio file."""
    stem = os.path.splitext(os.path.basename(audio_path))[0]
    return os.path.join(features_dir, f"{stem}.pt")


def waveform_to_mono_numpy(
    waveform: torch.Tensor, sample_rate: int
) -> tuple[np.ndarray, int]:
    """Convert a torch waveform to mono float32 numpy at CLAP_SAMPLE_RATE."""
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != CLAP_SAMPLE_RATE:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=CLAP_SAMPLE_RATE)
        waveform = resampler(waveform)
    arr = waveform.squeeze(0).detach().cpu().numpy().astype(np.float32)
    return arr, CLAP_SAMPLE_RATE


def compute_clap_inputs_for_clip(
    audio_path: str,
    feature_extractor: ClapFeatureExtractor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load WAV, resample to 48 kHz mono, run CLAP feature extractor.

    Returns (input_features, is_longer) without batch dimension.
    input_features shape [num_fusion, time, n_mels] (typically [4, T, 64]).
    is_longer shape [1, 1] bool.
    """
    waveform, sr = torchaudio.load(audio_path)
    arr, sr_out = waveform_to_mono_numpy(waveform, sr)
    inputs = feature_extractor(
        raw_speech=arr,
        sampling_rate=sr_out,
        return_tensors="pt",
    )
    input_features = inputs["input_features"].squeeze(0)
    is_longer = inputs["is_longer"].squeeze(0)
    if is_longer.dim() == 0:
        is_longer = is_longer.view(1, 1)
    elif is_longer.dim() == 1:
        is_longer = is_longer.view(1, 1)
    return input_features, is_longer


def compute_features_for_clip(
    audio_path: str,
    label_ids: list[int],
    features_dir: str,
    num_classes: int,
    feature_extractor: ClapFeatureExtractor,
) -> str:
    """Compute CLAP inputs for one clip and save .pt dict.

    Keys: input_features, is_longer, label_ids (multi-hot float32).
    """
    out_path = feature_path(audio_path, features_dir)
    if os.path.exists(out_path):
        return out_path

    input_features, is_longer = compute_clap_inputs_for_clip(
        audio_path, feature_extractor
    )

    label_vec = torch.zeros(num_classes, dtype=torch.float32)
    for idx in label_ids:
        if idx < num_classes:
            label_vec[idx] = 1.0

    torch.save(
        {
            "input_features": input_features,
            "is_longer": is_longer,
            "label_ids": label_vec,
        },
        out_path,
    )
    return out_path


def compute_features(
    manifest_path: str,
    features_dir: str,
    num_classes: int,
    clap_model_id: str,
) -> int:
    """Compute CLAP feature extractor outputs for every manifest row with existing audio."""
    os.makedirs(features_dir, exist_ok=True)
    feature_extractor = ClapFeatureExtractor.from_pretrained(clap_model_id)
    entries = read_manifest(manifest_path)

    written = 0
    for i, entry in enumerate(entries):
        logger.info(f"[{i + 1}/{len(entries)}] {entry['ytid']}")
        compute_features_for_clip(
            audio_path=entry["audio_path"],
            label_ids=entry["label_ids"],
            features_dir=features_dir,
            num_classes=num_classes,
            feature_extractor=feature_extractor,
        )
        written += 1

    return written
