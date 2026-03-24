"""CLI for offline analysis (UMAP plots, etc.)."""

import os

import typer
from loguru import logger

from audioset_classification.analysis.umap_viz import (
    run_umap_plots,
    run_umap_plots_combined,
)
from audioset_classification.cli.data_cli import (
    CSV_DIR,
    DEV_DATA,
    EMBEDDINGS_DIR,
    ONTOLOGY_CSV,
)

ANALYSIS_UMAP_DIR = os.path.join(DEV_DATA, "analysis", "umap")
ONTOLOGY_JSON_DEFAULT = os.path.join(CSV_DIR, "ontology.json")

analysis_cli = typer.Typer(help="Embedding analysis and plots", no_args_is_help=True)

_UMAP_SPLITS = ("train", "val", "test")


@analysis_cli.command("umap")
def umap_cmd(
    embeddings: str = typer.Option(
        os.path.join(EMBEDDINGS_DIR, "train.npz"),
        "--embeddings",
        "-e",
        help="Path to one CLAP embeddings .npz (ignored when ``--all-splits`` is set)",
    ),
    embeddings_dir: str = typer.Option(
        EMBEDDINGS_DIR,
        "--embeddings-dir",
        help="Directory containing ``{train,val,test}.npz`` when using ``--all-splits``",
    ),
    all_splits: bool = typer.Option(
        False,
        "--all-splits",
        help=(
            "Concatenate train/val/test ``.npz`` under ``--embeddings-dir``, "
            "fit one UMAP, write one ``all_splits`` tier plot set"
        ),
    ),
    combined_stem: str = typer.Option(
        "all_splits",
        "--combined-stem",
        help="Filename stem for cache/PNGs when using ``--all-splits``",
    ),
    ontology_json: str = typer.Option(
        ONTOLOGY_JSON_DEFAULT,
        "--ontology-json",
        help="AudioSet ontology.json (see README)",
    ),
    class_labels_csv: str = typer.Option(
        ONTOLOGY_CSV,
        "--class-labels-csv",
        help="class_labels_indices.csv for index→mid",
    ),
    out_dir: str = typer.Option(
        ANALYSIS_UMAP_DIR,
        "--out-dir",
        "-o",
        help="Output directory for UMAP .npy cache and tier PNGs",
    ),
    random_state: int = typer.Option(
        42,
        "--random-state",
        "--seed",
        help="UMAP random seed",
    ),
    n_neighbors: int = typer.Option(200, "--n-neighbors", help="UMAP n_neighbors"),
    min_dist: float = typer.Option(0.002, "--min-dist", help="UMAP min_dist"),
    point_size: int = typer.Option(8, "--point-size", help="Scatter marker size"),
    alpha: float = typer.Option(0.8, "--alpha", help="Scatter alpha"),
    tier_min: int = typer.Option(0, "--tier-min", help="First ontology tier index"),
    tier_max: int | None = typer.Option(
        None,
        "--tier-max",
        help="Last ontology tier index (inclusive); omit for all tiers",
    ),
) -> None:
    """UMAP tier plots: one file, or ``--all-splits`` to combine train/val/test into one UMAP."""
    common_kw = {
        "ontology_json_path": ontology_json,
        "class_labels_csv_path": class_labels_csv,
        "out_dir": out_dir,
        "random_state": random_state,
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "point_size": point_size,
        "alpha": alpha,
        "tier_min": tier_min,
        "tier_max": tier_max,
    }

    if all_splits:
        npz_paths: list[str] = []
        for split in _UMAP_SPLITS:
            p = os.path.join(embeddings_dir, f"{split}.npz")
            if os.path.isfile(p):
                npz_paths.append(p)
            else:
                logger.warning(f"No embeddings file for split '{split}': {p}")
        if not npz_paths:
            typer.echo(
                f"No split npz files found under {embeddings_dir} "
                f"(expected train.npz, val.npz, and/or test.npz).",
                err=True,
            )
            raise typer.Exit(1)
        logger.info(
            f"Combined UMAP for {len(npz_paths)} split file(s) -> {out_dir} "
            f"(stem={combined_stem!r})"
        )
        paths = run_umap_plots_combined(
            npz_paths,
            output_stem=combined_stem,
            **common_kw,
        )
        logger.info(f"Wrote {len(paths)} figure(s)")
        return

    logger.info(f"UMAP from {embeddings} -> {out_dir}")
    paths = run_umap_plots(embeddings_npz_path=embeddings, **common_kw)
    logger.info(f"Wrote {len(paths)} figure(s)")
