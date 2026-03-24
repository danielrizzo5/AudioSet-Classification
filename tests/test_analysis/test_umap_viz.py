"""Tests for UMAP visualization (cached projection, tier PNGs)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from audioset_classification.analysis.umap_viz import (
    _umap_cache_path,
    run_umap_plots,
    run_umap_plots_combined,
    tab20_palette,
)


def _tiny_ontology_json(path: Path) -> None:
    nodes = [
        {"id": "/r", "name": "root", "child_ids": ["/a"]},
        {"id": "/a", "name": "a", "child_ids": ["/b"]},
        {"id": "/b", "name": "leaf", "child_ids": []},
    ]
    path.write_text(json.dumps(nodes), encoding="utf-8")


def _class_labels_csv(path: Path) -> None:
    df = pd.DataFrame(
        {
            "index": [0],
            "mid": ["/b"],
            "display_name": ["LeafB"],
        }
    )
    df.to_csv(path, index=False)


def test_tab20_palette_length_and_blend():
    """Beyond 20 categories, colors blend adjacent tab20 entries."""
    p25 = tab20_palette(25)
    assert len(p25) == 25
    assert len(p25[0]) == 4
    assert p25[19] != p25[20]


def test_run_umap_plots_writes_tier_pngs_and_reuses_umap_cache(tmp_path):
    """One UMAP fit; tier figures exist; second run does not refit UMAP."""
    ont = tmp_path / "ontology.json"
    _tiny_ontology_json(ont)
    csv_path = tmp_path / "class_labels_indices.csv"
    _class_labels_csv(csv_path)

    emb_path = tmp_path / "train.npz"
    np.savez(
        emb_path,
        embedding=np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32),
        ytid=np.array(["a", "b"], dtype=object),
        start=np.array([0.0, 0.0], dtype=np.float64),
        end=np.array([1.0, 1.0], dtype=np.float64),
        representative_label_id=np.array([0, 0], dtype=np.int64),
    )

    out_dir = tmp_path / "umap_out"

    fits: list[int] = []

    def fake_umap(embedding: np.ndarray, **kwargs: object) -> np.ndarray:
        fits.append(embedding.shape[0])
        n = embedding.shape[0]
        return np.column_stack(
            [np.arange(n, dtype=np.float32), np.zeros(n, dtype=np.float32)]
        )

    pngs = run_umap_plots(
        embeddings_npz_path=str(emb_path),
        ontology_json_path=str(ont),
        class_labels_csv_path=str(csv_path),
        out_dir=str(out_dir),
        umap_fitter=fake_umap,
    )
    assert len(fits) == 1
    assert len(pngs) == 3
    for p in pngs:
        assert Path(p).is_file()

    cache = Path(_umap_cache_path(str(out_dir), "train", 2, min(200, 1), 0.002))
    assert cache.is_file()

    def boom(*args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("UMAP should not refit when cache matches N")

    pngs2 = run_umap_plots(
        embeddings_npz_path=str(emb_path),
        ontology_json_path=str(ont),
        class_labels_csv_path=str(csv_path),
        out_dir=str(out_dir),
        umap_fitter=boom,
    )
    assert len(pngs2) == 3


def test_run_umap_plots_combined_stacks_and_one_cache(tmp_path: Path):
    """Multiple npz files are vstacked; one UMAP cache stem; tier PNGs once."""
    ont = tmp_path / "ontology.json"
    _tiny_ontology_json(ont)
    csv_path = tmp_path / "class_labels_indices.csv"
    _class_labels_csv(csv_path)

    a = tmp_path / "train.npz"
    np.savez(
        a,
        embedding=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        ytid=np.array(["a", "b"], dtype=object),
        start=np.zeros(2),
        end=np.ones(2),
        representative_label_id=np.array([0, 0], dtype=np.int64),
    )
    b = tmp_path / "val.npz"
    np.savez(
        b,
        embedding=np.array([[2.0, 2.0], [3.0, 3.0]], dtype=np.float32),
        ytid=np.array(["c", "d"], dtype=object),
        start=np.zeros(2),
        end=np.ones(2),
        representative_label_id=np.array([0, 0], dtype=np.int64),
    )
    out_dir = tmp_path / "umap_out"
    fits: list[int] = []

    def fake_umap(embedding: np.ndarray, **kwargs: object) -> np.ndarray:
        fits.append(embedding.shape[0])
        n = embedding.shape[0]
        return np.column_stack(
            [np.arange(n, dtype=np.float32), np.zeros(n, dtype=np.float32)]
        )

    pngs = run_umap_plots_combined(
        [str(a), str(b)],
        ontology_json_path=str(ont),
        class_labels_csv_path=str(csv_path),
        out_dir=str(out_dir),
        output_stem="all_splits",
        umap_fitter=fake_umap,
    )
    assert fits == [4]
    assert len(pngs) == 3
    cache = Path(_umap_cache_path(str(out_dir), "all_splits", 4, min(200, 3), 0.002))
    assert cache.is_file()
    assert (out_dir / "all_splits_tier_0.png").is_file()
