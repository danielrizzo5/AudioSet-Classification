"""Materialize CLAP audio embeddings from precomputed feature .pt files."""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from transformers import ClapModel

from audioset_classification.data.collate import collate_clap_inputs
from audioset_classification.data.features import feature_path
from audioset_classification.data.manifest import read_manifest


def _module_to_device(module: nn.Module, target_device: torch.device) -> None:
    """Move ``module`` to ``target_device`` (``str`` avoids pyright/torch stub noise)."""
    module.to(str(target_device))


def compute_clap_embeddings(
    manifest_path: str,
    features_dir: str,
    out_npz_path: str,
    clap_model_id: str,
    batch_size: int,
    target_device: torch.device,
) -> int:
    """Run CLAP audio encoder on manifest rows that have a precomputed .pt feature file.

    Applies ``audio_projection`` + L2 normalization so that embeddings live in
    the unit-norm contrastive space and are comparable across splits.

    Expects ``audioset data features`` (or equivalent) to have written
    ``feature_path(audio_path, features_dir)`` for each row. Rows without a
    .pt file are skipped (warning log).

    Multi-label rows store the **first** ``label_id`` in manifest order as
    ``representative_label_id``.

    Saves ``out_npz_path`` with arrays: embedding [N,D], ytid, start, end,
    representative_label_id (int64).

    Returns N (number of rows written).
    """
    os.makedirs(os.path.dirname(out_npz_path) or ".", exist_ok=True)

    clap_model = ClapModel.from_pretrained(clap_model_id)
    _module_to_device(clap_model, target_device)
    clap_model.eval()

    entries = read_manifest(manifest_path)

    pending_inputs: list[tuple[torch.Tensor, torch.Tensor]] = []
    pending_meta: list[tuple[str, float, float, int]] = []

    all_emb: list[np.ndarray] = []
    all_ytid: list[str] = []
    all_start: list[float] = []
    all_end: list[float] = []
    all_rep_label: list[int] = []

    def flush_batch() -> None:
        nonlocal pending_inputs, pending_meta
        if not pending_inputs:
            return
        feats, longers = collate_clap_inputs(pending_inputs)
        dev_s = str(target_device)
        feats = feats.to(dev_s)
        longers = longers.to(dev_s)
        with torch.no_grad():
            audio_out = clap_model.audio_model(
                input_features=feats,
                is_longer=longers,
                return_dict=True,
            )
            projected = clap_model.audio_projection(audio_out.pooler_output)
            normalized = F.normalize(projected, dim=-1)
        emb = normalized.detach().cpu().numpy().astype(np.float32)
        for row in emb:
            all_emb.append(row)
        for ytid, start, end, rep in pending_meta:
            all_ytid.append(ytid)
            all_start.append(start)
            all_end.append(end)
            all_rep_label.append(rep)
        pending_inputs = []
        pending_meta = []

    n_skipped = 0
    for i, entry in enumerate(entries):
        logger.info(f"[{i + 1}/{len(entries)}] {entry['ytid']}")
        pt_path = feature_path(entry["audio_path"], features_dir)
        if not os.path.isfile(pt_path):
            n_skipped += 1
            logger.warning(
                "Skipped clip (no precomputed .pt): "
                f"ytid={entry['ytid']} expected={pt_path}"
            )
            continue
        payload = torch.load(pt_path, map_location="cpu", weights_only=False)
        input_features = payload["input_features"]
        is_longer = payload["is_longer"]
        label_ids: list[int] = entry["label_ids"]
        rep_label = int(label_ids[0])
        pending_inputs.append((input_features, is_longer))
        pending_meta.append(
            (str(entry["ytid"]), float(entry["start"]), float(entry["end"]), rep_label)
        )
        if len(pending_inputs) >= batch_size:
            flush_batch()

    flush_batch()

    if n_skipped:
        logger.warning(
            f"Skipped {n_skipped} manifest row(s) with missing .pt under {features_dir}."
        )

    if not all_emb:
        dim = int(clap_model.config.projection_dim)
        np.savez(
            out_npz_path,
            embedding=np.zeros((0, dim), dtype=np.float32),
            ytid=np.array([], dtype=object),
            start=np.array([], dtype=np.float64),
            end=np.array([], dtype=np.float64),
            representative_label_id=np.array([], dtype=np.int64),
        )
        return 0

    np.savez(
        out_npz_path,
        embedding=np.stack(all_emb, axis=0),
        ytid=np.array(all_ytid, dtype=object),
        start=np.asarray(all_start, dtype=np.float64),
        end=np.asarray(all_end, dtype=np.float64),
        representative_label_id=np.asarray(all_rep_label, dtype=np.int64),
    )
    return len(all_emb)
