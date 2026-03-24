"""UMAP projection of CLAP embeddings with ontology-tier colors."""

import os
from typing import Callable

import numpy as np
import pandas as pd
from loguru import logger

from audioset_classification.data.ontology import index_to_mid, load_ontology
from audioset_classification.data.ontology_tree import (
    build_mid_to_name,
    build_parent_map,
    load_ontology_nodes,
    mid_path_root_to_leaf,
)


def tab20_palette(n: int) -> list[tuple[float, float, float, float]]:
    """Categorical RGBA colors: ``tab20`` for the first 20, then blends of adjacent tab20 colors.

    Matches the palette construction in ``observer`` ``ai.pipelines.cluster.umap.write_umap``.
    """
    import matplotlib.pyplot as plt

    base_cmap = plt.get_cmap("tab20")
    base_colors = [base_cmap(i) for i in range(20)]
    palette: list[tuple[float, float, float, float]] = []
    for i in range(n):
        c1 = base_colors[i % 20]
        if i < 20:
            r, g, b, a = (float(c1[j]) for j in range(4))
            palette.append((r, g, b, a))
        else:
            c2 = base_colors[(i + 1) % 20]
            blended = (
                (float(c1[0]) + float(c2[0])) / 2.0,
                (float(c1[1]) + float(c2[1])) / 2.0,
                (float(c1[2]) + float(c2[2])) / 2.0,
                (float(c1[3]) + float(c2[3])) / 2.0,
            )
            palette.append(blended)
    return palette


def _umap_cache_path(
    out_dir: str, stem: str, n: int, n_neighbors: int, min_dist: float
) -> str:
    """Sidecar path for 2D UMAP; includes hyperparameters so cache stays consistent."""
    md = f"{min_dist:g}".replace(".", "p")
    return os.path.join(out_dir, f"{stem}_umap2d_n{n_neighbors}_md{md}_N{n}.npy")


def fit_umap_2d(
    embedding: np.ndarray,
    random_state: int,
    n_neighbors: int,
    min_dist: float,
) -> np.ndarray:
    """Fit UMAP and return ``float32`` ``[N, 2]`` (import-heavy; patchable in tests)."""
    import umap

    reducer = umap.UMAP(
        n_components=2,
        metric="euclidean",
        random_state=random_state,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_jobs=1,
        verbose=False,
    )
    xy = np.asarray(reducer.fit_transform(embedding), dtype=np.float32)
    return xy


def _tier_label_names(
    representative_label_id: np.ndarray,
    idx_to_mid: dict[int, str],
    parent_map: dict[str, str],
    mid_to_name: dict[str, str],
    all_mids: set[str],
) -> list[list[str]]:
    """Root-to-leaf MID paths as display names for each row."""
    paths: list[list[str]] = []
    for lid in representative_label_id.astype(int):
        leaf_mid = idx_to_mid[lid]
        if leaf_mid not in all_mids:
            raise ValueError(
                f"class_labels mid {leaf_mid!r} (index {lid}) not in ontology.json"
            )
        mid_path = mid_path_root_to_leaf(leaf_mid, parent_map)
        paths.append([mid_to_name[m] for m in mid_path])
    return paths


def _run_umap_plots_core(
    embedding: np.ndarray,
    rep_ids: np.ndarray,
    stem: str,
    ontology_json_path: str,
    class_labels_csv_path: str,
    out_dir: str,
    random_state: int,
    n_neighbors: int,
    min_dist: float,
    point_size: int,
    alpha: float,
    tier_min: int,
    tier_max: int | None,
    umap_fitter: Callable[..., np.ndarray] | None,
) -> list[str]:
    """Single UMAP fit (or cache load) and tier PNGs for stacked ``embedding`` / ``rep_ids``."""
    umap_fn = umap_fitter if umap_fitter is not None else fit_umap_2d

    n = embedding.shape[0]
    os.makedirs(out_dir, exist_ok=True)
    nn = min(n_neighbors, n - 1)
    umap_npy_path = _umap_cache_path(out_dir, stem, n, nn, min_dist)

    if n == 0:
        logger.warning("Embedding matrix is empty; no UMAP or plots written.")
        return []

    if n < 2:
        raise ValueError("UMAP requires at least 2 embedding rows.")

    xy: np.ndarray
    if os.path.isfile(umap_npy_path):
        cached = np.load(umap_npy_path)
        if cached.shape == (n, 2):
            xy = cached.astype(np.float32)
            logger.info(f"Loaded UMAP coordinates from {umap_npy_path}")
        else:
            logger.info(
                f"UMAP cache shape {cached.shape} != ({n}, 2); recomputing UMAP."
            )
            xy = umap_fn(
                embedding,
                random_state=random_state,
                n_neighbors=nn,
                min_dist=min_dist,
            )
            np.save(umap_npy_path, xy)
    else:
        xy = umap_fn(
            embedding,
            random_state=random_state,
            n_neighbors=nn,
            min_dist=min_dist,
        )
        np.save(umap_npy_path, xy)
        logger.info(f"Saved UMAP coordinates to {umap_npy_path}")

    nodes = load_ontology_nodes(ontology_json_path)
    parent_map = build_parent_map(nodes)
    mid_to_name = build_mid_to_name(nodes)
    all_mids = set(mid_to_name.keys())

    ontology_df = load_ontology(class_labels_csv_path)
    idx_to_mid = index_to_mid(ontology_df)

    name_paths = _tier_label_names(
        rep_ids, idx_to_mid, parent_map, mid_to_name, all_mids
    )
    max_tier_index = max(len(p) - 1 for p in name_paths)
    t_end = (
        max_tier_index + 1
        if tier_max is None
        else min(tier_max + 1, max_tier_index + 1)
    )
    t_start = max(0, tier_min)

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    written: list[str] = []
    for t in range(t_start, t_end):
        labels_t: list[str] = []
        idx_ok: list[int] = []
        for i, names in enumerate(name_paths):
            if len(names) > t:
                idx_ok.append(i)
                labels_t.append(names[t])
        n_drop = n - len(idx_ok)
        if n_drop:
            logger.info(f"tier {t}: omitted {n_drop} point(s) (path shorter than tier)")

        if not idx_ok:
            logger.warning(f"tier {t}: no points to plot; skipping figure")
            continue

        sel = np.asarray(idx_ok, dtype=np.intp)
        xys = xy[sel]
        cats = pd.Categorical(labels_t)
        codes = cats.codes.astype(np.intp)
        n_cat = len(cats.categories)
        palette = tab20_palette(n_cat)
        point_colors = [palette[int(codes[i])] for i in range(len(codes))]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(
            xys[:, 0],
            xys[:, 1],
            c=point_colors,
            s=point_size,
            alpha=alpha,
        )
        ax.set_title(f"UMAP (tier {t}) — {n_cat} categories")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=str(cats.categories[i]),
                markerfacecolor=palette[i],
                markersize=8,
            )
            for i in range(n_cat)
        ]
        ax.legend(
            handles=legend_elements,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.0,
            fontsize=8,
        )
        fig.tight_layout()
        png_path = os.path.join(out_dir, f"{stem}_tier_{t}.png")
        fig.savefig(png_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        written.append(png_path)
        logger.info(f"Wrote {png_path}")

    return written


def run_umap_plots(
    embeddings_npz_path: str,
    ontology_json_path: str,
    class_labels_csv_path: str,
    out_dir: str,
    random_state: int = 42,
    n_neighbors: int = 200,
    min_dist: float = 0.002,
    point_size: int = 8,
    alpha: float = 0.8,
    tier_min: int = 0,
    tier_max: int | None = None,
    umap_fitter: Callable[..., np.ndarray] | None = None,
) -> list[str]:
    """Load one .npz, reuse or fit 2D UMAP once, write one PNG per ontology tier.

    Tier ``t`` colors points by the display name at depth ``t`` on the path from
    root to the representative class (first manifest ``label_id``). Rows with
    ``len(path) <= t`` are omitted from that scatter; the log records the count.

    Returns paths of written PNG files.
    """
    data = np.load(embeddings_npz_path, allow_pickle=True)
    embedding = data["embedding"].astype(np.float32)
    rep_ids = data["representative_label_id"]
    stem = os.path.splitext(os.path.basename(embeddings_npz_path))[0]
    return _run_umap_plots_core(
        embedding=embedding,
        rep_ids=rep_ids,
        stem=stem,
        ontology_json_path=ontology_json_path,
        class_labels_csv_path=class_labels_csv_path,
        out_dir=out_dir,
        random_state=random_state,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        point_size=point_size,
        alpha=alpha,
        tier_min=tier_min,
        tier_max=tier_max,
        umap_fitter=umap_fitter,
    )


def run_umap_plots_combined(
    embeddings_npz_paths: list[str],
    ontology_json_path: str,
    class_labels_csv_path: str,
    out_dir: str,
    output_stem: str = "all_splits",
    random_state: int = 42,
    n_neighbors: int = 200,
    min_dist: float = 0.002,
    point_size: int = 8,
    alpha: float = 0.8,
    tier_min: int = 0,
    tier_max: int | None = None,
    umap_fitter: Callable[..., np.ndarray] | None = None,
) -> list[str]:
    """Load multiple .npz files (same embedding dim), concatenate rows, one UMAP + tier set.

    Order follows ``embeddings_npz_paths``. Cache and PNGs use ``output_stem``
    (e.g. ``all_splits_tier_0.png``).
    """
    if not embeddings_npz_paths:
        logger.warning("No embedding paths given; nothing to plot.")
        return []

    blocks_emb: list[np.ndarray] = []
    blocks_rep: list[np.ndarray] = []
    dim: int | None = None
    for p in embeddings_npz_paths:
        data = np.load(p, allow_pickle=True)
        e = data["embedding"].astype(np.float32)
        r = data["representative_label_id"]
        if e.shape[0] == 0:
            continue
        if dim is None:
            dim = e.shape[1]
        elif e.shape[1] != dim:
            raise ValueError(
                f"embedding dim mismatch for {p}: got {e.shape[1]}, expected {dim}"
            )
        if r.shape[0] != e.shape[0]:
            raise ValueError(
                f"row count mismatch in {p}: embedding {e.shape[0]} vs labels {r.shape[0]}"
            )
        blocks_emb.append(e)
        blocks_rep.append(r.astype(np.int64, copy=False))

    if not blocks_emb:
        logger.warning("All provided .npz files are empty; no UMAP or plots written.")
        return []

    embedding = np.vstack(blocks_emb)
    rep_ids = np.concatenate(blocks_rep, axis=0)
    logger.info(
        f"Combined UMAP: {len(blocks_emb)} non-empty file(s) "
        f"({len(embeddings_npz_paths)} path(s) given), "
        f"{embedding.shape[0]} rows, stem={output_stem!r}"
    )
    return _run_umap_plots_core(
        embedding=embedding,
        rep_ids=rep_ids,
        stem=output_stem,
        ontology_json_path=ontology_json_path,
        class_labels_csv_path=class_labels_csv_path,
        out_dir=out_dir,
        random_state=random_state,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        point_size=point_size,
        alpha=alpha,
        tier_min=tier_min,
        tier_max=tier_max,
        umap_fitter=umap_fitter,
    )
