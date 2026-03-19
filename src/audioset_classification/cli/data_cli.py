"""CLI for AudioSet data pipeline: download, manifest, features, inspect."""

import os
import shutil

import typer
from loguru import logger

from audioset_classification.data.csv_loader import load_segments_csv
from audioset_classification.data.download import download_clips
from audioset_classification.data.features import compute_features
from audioset_classification.data.manifest import write_manifest
from audioset_classification.data.ontology import load_ontology

DEV_DATA = "dev-data"
CSV_DIR = os.path.join(DEV_DATA, "audioset-csv")
AUDIO_DIR = os.path.join(DEV_DATA, "audio")
MANIFESTS_DIR = os.path.join(DEV_DATA, "manifests")
FEATURES_DIR = os.path.join(DEV_DATA, "features")

TRAIN_CSV = os.path.join(CSV_DIR, "balanced_train_segments.csv")
EVAL_CSV = os.path.join(CSV_DIR, "eval_segments.csv")
ONTOLOGY_CSV = os.path.join(CSV_DIR, "class_labels_indices.csv")

NUM_CLASSES = 527

data_cli = typer.Typer(help="AudioSet data pipeline", no_args_is_help=True)


def _require_ffmpeg() -> None:
    """Abort with a helpful message if ffmpeg is not on PATH."""
    if shutil.which("ffmpeg") is None:
        typer.echo("ffmpeg not found. Install it with: brew install ffmpeg", err=True)
        raise typer.Exit(1)


@data_cli.command()
def download(
    split: str = typer.Option(
        "train", "--split", "-s", help="Split to download: train or eval"
    ),
    max_clips: int = typer.Option(
        None, "--max-clips", "-n", help="Limit number of clips"
    ),
    audio_dir: str = typer.Option(
        AUDIO_DIR, "--audio-dir", help="Output WAV directory"
    ),
    sample_rate: int = typer.Option(16000, "--sample-rate", help="Target sample rate"),
) -> None:
    """Download and trim audio clips for a split using yt-dlp and ffmpeg."""
    _require_ffmpeg()
    csv_path = TRAIN_CSV if split == "train" else EVAL_CSV
    df = load_segments_csv(csv_path, split=split, max_segments=max_clips)
    logger.info(f"Downloading {len(df)} clips for split '{split}'")
    result = download_clips(df, audio_dir=audio_dir, sample_rate=sample_rate)
    n_ok = result["audio_path"].notna().sum()
    logger.info(f"Downloaded {n_ok}/{len(df)} clips -> {audio_dir}")


@data_cli.command()
def manifest(
    audio_dir: str = typer.Option(AUDIO_DIR, "--audio-dir", help="WAV directory"),
    manifests_dir: str = typer.Option(
        MANIFESTS_DIR, "--manifests-dir", help="Output manifests directory"
    ),
    val_fraction: float = typer.Option(
        0.5,
        "--val-fraction",
        help="Fraction of eval split used for val (remainder is test)",
    ),
) -> None:
    """Write JSONL manifests (train.jsonl, val.jsonl, test.jsonl) from downloaded audio."""
    ontology = load_ontology(ONTOLOGY_CSV)

    train_df = load_segments_csv(TRAIN_CSV, split="train")
    train_df = train_df.assign(
        audio_path=train_df.apply(
            lambda r: _audio_path(
                r["ytid"], r["start_seconds"], r["end_seconds"], audio_dir
            ),
            axis=1,
        )
    )
    n_train = write_manifest(
        train_df, ontology, audio_dir, os.path.join(manifests_dir, "train.jsonl")
    )
    logger.info(f"train.jsonl: {n_train} entries")

    eval_df = load_segments_csv(EVAL_CSV, split="eval")
    eval_df = eval_df.assign(
        audio_path=eval_df.apply(
            lambda r: _audio_path(
                r["ytid"], r["start_seconds"], r["end_seconds"], audio_dir
            ),
            axis=1,
        )
    )
    split_idx = int(len(eval_df) * val_fraction)
    val_df = eval_df.iloc[:split_idx]
    test_df = eval_df.iloc[split_idx:]

    n_val = write_manifest(
        val_df, ontology, audio_dir, os.path.join(manifests_dir, "val.jsonl")
    )
    n_test = write_manifest(
        test_df, ontology, audio_dir, os.path.join(manifests_dir, "test.jsonl")
    )
    logger.info(f"val.jsonl: {n_val} entries, test.jsonl: {n_test} entries")


@data_cli.command()
def features(
    split: str = typer.Option(
        "train",
        "--split",
        "-s",
        help="Split to compute features for: train, val, or test",
    ),
    manifests_dir: str = typer.Option(
        MANIFESTS_DIR, "--manifests-dir", help="Manifests directory"
    ),
    features_dir: str = typer.Option(
        FEATURES_DIR, "--features-dir", help="Output features directory"
    ),
    num_classes: int = typer.Option(
        NUM_CLASSES, "--num-classes", help="Number of output classes"
    ),
) -> None:
    """Compute log-mel spectrogram features and save as .pt files."""
    manifest_path = os.path.join(manifests_dir, f"{split}.jsonl")
    logger.info(f"Computing features for '{split}' from {manifest_path}")
    written = compute_features(manifest_path, features_dir, num_classes)
    logger.info(f"Wrote {written} feature files -> {features_dir}")


@data_cli.command()
def inspect(
    csv_dir: str = typer.Option(CSV_DIR, "--csv-dir", help="AudioSet CSV directory"),
) -> None:
    """Print summary of CSV segments and ontology."""
    ontology = load_ontology(os.path.join(csv_dir, "class_labels_indices.csv"))
    typer.echo(f"Ontology: {len(ontology)} classes")

    for split, path in [("train", TRAIN_CSV), ("eval", EVAL_CSV)]:
        df = load_segments_csv(path, split=split)
        typer.echo(f"{split}: {len(df)} segments")


def _audio_path(ytid: str, start: float, end: float, audio_dir: str) -> str:
    """Build the expected WAV path for a segment."""
    from audioset_classification.data.download import audio_path as _ap

    return _ap(ytid, start, end, audio_dir)
