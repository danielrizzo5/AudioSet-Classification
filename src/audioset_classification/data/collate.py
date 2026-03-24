"""Batch collation for variable-length CLAP input_features."""

import torch
import torch.nn.functional as F


def collate_clap_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad input_features along time (dim 1) to the max length in the batch.

    Args:
        batch: List of (input_features [C, T, F], is_longer [1, 1], label_ids [num_classes]).

    Returns:
        input_features [B, C, T_max, F], is_longer [B, 1], labels [B, num_classes].
    """
    feats = [b[0] for b in batch]
    longers = [b[1] for b in batch]
    labels = torch.stack([b[2] for b in batch], dim=0)

    max_t = max(f.shape[1] for f in feats)
    padded: list[torch.Tensor] = []
    for f in feats:
        _, t, freq = f.shape
        pad_t = max_t - t
        if pad_t > 0:
            f = F.pad(f, (0, 0, 0, pad_t))
        padded.append(f)
    batched_feats = torch.stack(padded, dim=0)
    batched_longers = torch.stack(longers, dim=0)
    if batched_longers.dim() == 3:
        batched_longers = batched_longers.squeeze(1)
    return batched_feats, batched_longers, labels
