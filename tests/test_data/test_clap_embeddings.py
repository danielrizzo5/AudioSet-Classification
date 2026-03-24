"""Tests for CLAP embeddings from precomputed .pt features."""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from audioset_classification.data.clap_embeddings import compute_clap_embeddings


def test_compute_clap_embeddings_from_pt_files(tmp_path):
    """Loads input_features/is_longer from .pt and runs the audio encoder only."""
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    audio_path = str(tmp_path / "seg_clip.wav")
    torch.save(
        {
            "input_features": torch.zeros(4, 2, 64),
            "is_longer": torch.tensor([[False]], dtype=torch.bool),
            "label_ids": torch.zeros(527),
        },
        features_dir / "seg_clip.pt",
    )
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "audio_path": audio_path,
                "label_ids": [3],
                "ytid": "x",
                "start": 0.0,
                "end": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out_npz = tmp_path / "out.npz"

    mock_model = MagicMock()
    mock_model.config.projection_dim = 8

    def _audio_forward1(
        *,
        input_features: torch.Tensor,
        is_longer: torch.Tensor,
        return_dict: bool = True,
    ) -> MagicMock:
        b = input_features.shape[0]
        r = MagicMock()
        r.pooler_output = torch.ones(b, 16, dtype=torch.float32)
        return r

    mock_model.audio_model.side_effect = _audio_forward1
    mock_model.audio_projection.side_effect = lambda x: torch.ones(
        x.shape[0], 8, dtype=torch.float32
    )

    with patch(
        "audioset_classification.data.clap_embeddings.ClapModel.from_pretrained",
        return_value=mock_model,
    ):
        n = compute_clap_embeddings(
            str(manifest),
            str(features_dir),
            str(out_npz),
            clap_model_id="dummy",
            batch_size=4,
            target_device=torch.device("cpu"),
        )

    assert n == 1
    data = np.load(out_npz, allow_pickle=True)
    assert data["embedding"].shape == (1, 8)
    assert int(data["representative_label_id"][0]) == 3


def test_compute_clap_embeddings_skips_missing_pt(tmp_path):
    """Manifest row without a matching .pt is omitted from the npz."""
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    manifest = tmp_path / "m.jsonl"
    lines = [
        {
            "audio_path": str(tmp_path / "a.wav"),
            "label_ids": [0],
            "ytid": "a",
            "start": 0.0,
            "end": 1.0,
        },
        {
            "audio_path": str(tmp_path / "b.wav"),
            "label_ids": [1],
            "ytid": "b",
            "start": 0.0,
            "end": 1.0,
        },
    ]
    manifest.write_text(
        "\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8"
    )
    torch.save(
        {
            "input_features": torch.zeros(4, 2, 64),
            "is_longer": torch.tensor([[False]], dtype=torch.bool),
            "label_ids": torch.zeros(527),
        },
        features_dir / "a.pt",
    )
    out_npz = tmp_path / "out.npz"

    mock_model = MagicMock()
    mock_model.config.projection_dim = 4

    def _audio_forward2(
        *,
        input_features: torch.Tensor,
        is_longer: torch.Tensor,
        return_dict: bool = True,
    ) -> MagicMock:
        b = input_features.shape[0]
        r = MagicMock()
        r.pooler_output = torch.ones(b, 8, dtype=torch.float32)
        return r

    mock_model.audio_model.side_effect = _audio_forward2
    mock_model.audio_projection.side_effect = lambda x: torch.ones(
        x.shape[0], 4, dtype=torch.float32
    )

    with patch(
        "audioset_classification.data.clap_embeddings.ClapModel.from_pretrained",
        return_value=mock_model,
    ):
        n = compute_clap_embeddings(
            str(manifest),
            str(features_dir),
            str(out_npz),
            clap_model_id="dummy",
            batch_size=4,
            target_device=torch.device("cpu"),
        )

    assert n == 1
    data = np.load(out_npz, allow_pickle=True)
    assert data["embedding"].shape[0] == 1
    assert data["ytid"][0] == "a"
