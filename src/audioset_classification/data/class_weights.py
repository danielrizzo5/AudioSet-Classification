"""Per-class positive counts and BCE ``pos_weight`` from train manifests."""

import os

import numpy as np
import torch

from audioset_classification.data.manifest import read_manifest


def _pos_counts_from_entries(entries: list[dict], num_classes: int) -> np.ndarray:
    pos_counts = np.zeros(num_classes, dtype=np.int64)
    for entry in entries:
        for lid in entry["label_ids"]:
            idx = int(lid)
            if idx < 0 or idx >= num_classes:
                raise ValueError(f"label_id {idx} out of range [0, {num_classes})")
            pos_counts[idx] += 1
    return pos_counts


def count_positive_labels_per_class(manifest_path: str, num_classes: int) -> np.ndarray:
    """Count how many manifest rows contain each label id as positive."""
    return _pos_counts_from_entries(read_manifest(manifest_path), num_classes)


def bce_pos_weight_from_train_manifest(
    manifest_path: str,
    num_classes: int,
    alpha: float,
) -> torch.Tensor:
    """``pos_weight`` for ``BCEWithLogitsLoss``: ``((N - n_c) / n_c) ** alpha`` per class.

    ``N`` is the number of manifest rows; ``n_c`` is the positive count for class ``c``.
    Requires ``n_c > 0`` for every class.
    """
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(manifest_path)
    entries = read_manifest(manifest_path)
    n_clips = len(entries)
    if n_clips == 0:
        raise ValueError(f"Train manifest is empty: {manifest_path!r}")
    pos_counts = _pos_counts_from_entries(entries, num_classes)
    zero = np.where(pos_counts <= 0)[0]
    if len(zero) > 0:
        preview = zero[:15].tolist()
        raise ValueError(
            f"No positive train examples for {len(zero)} class(es); "
            f"first indices: {preview}"
        )
    pos = pos_counts.astype(np.float64)
    neg = float(n_clips) - pos
    ratio = neg / pos
    weights = np.power(ratio, alpha)
    return torch.from_numpy(weights).to(dtype=torch.float32)
