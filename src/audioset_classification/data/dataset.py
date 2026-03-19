"""PyTorch Dataset for AudioSet embeddings and labels."""

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from audioset_classification.data.csv_loader import (
    parse_positive_labels,
    segment_key,
)
from audioset_classification.data.ontology import load_ontology, mid_to_index


class AudioSetDataset(Dataset):
    """
    Dataset yielding (embedding, labels) for AudioSet classification.

    Loads embeddings from .pt files (torch.save) at
    {features_dir}/{shard}/{ytid}_{start}_{end}.pt
    or synthetic random embeddings for testing.
    """

    def __init__(
        self,
        segments_df,
        ontology_path: str,
        features_dir: str | None = None,
        num_classes: int | None = None,
        embedding_dim: int = 128,
        synthetic: bool = False,
    ):
        self.segments_df = segments_df.reset_index(drop=True)
        self.ontology = load_ontology(ontology_path)
        self.mid_to_idx = mid_to_index(self.ontology)
        self.num_classes = (
            num_classes if num_classes is not None else len(self.ontology)
        )
        self.embedding_dim = embedding_dim
        self.synthetic = synthetic
        self.features_dir = features_dir

        if not synthetic and features_dir is None:
            raise ValueError("Either features_dir or synthetic=True must be provided")

    def _labels_to_vector(self, mids: list[str]) -> torch.Tensor:
        """Convert list of mids to multi-hot vector."""
        vec = torch.zeros(self.num_classes, dtype=torch.float32)
        for mid in mids:
            if mid in self.mid_to_idx:
                idx = self.mid_to_idx[mid]
                if idx < self.num_classes:
                    vec[idx] = 1.0
        return vec

    def _load_embedding(self, idx: int) -> torch.Tensor:
        """Load embedding for segment at idx. Returns 128-dim vector from .pt file."""
        if self.synthetic:
            rng = np.random.default_rng(seed=idx)
            return torch.from_numpy(rng.random(self.embedding_dim, dtype=np.float32))

        assert self.features_dir is not None
        row = self.segments_df.iloc[idx]
        ytid = str(row["ytid"])
        start = float(row["start_seconds"])
        end = float(row["end_seconds"])
        shard = ytid[:2] if len(ytid) >= 2 else "00"
        key = segment_key(ytid, start, end)
        pt_path = os.path.join(self.features_dir, shard, f"{key}.pt")
        if not os.path.exists(pt_path):
            return torch.zeros(self.embedding_dim, dtype=torch.float32)

        with open(pt_path, "rb") as f:
            emb = torch.load(f, map_location="cpu", weights_only=True)
        emb = emb.to(dtype=torch.float32)
        if emb.dim() == 2:
            emb = emb.mean(dim=0)
        return emb

    def __len__(self) -> int:
        return len(self.segments_df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.segments_df.iloc[idx]
        labels_str = str(row["positive_labels"])
        mids = parse_positive_labels(labels_str)
        label_vec = self._labels_to_vector(mids)
        embedding = self._load_embedding(idx)
        return embedding, label_vec
