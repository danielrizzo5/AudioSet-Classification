"""CLI for data download and inspection."""

import typer

from audioset_classification.data.csv_loader import load_segments_csv
from audioset_classification.data.ontology import load_ontology

data_cli = typer.Typer(help="Data download and inspection", no_args_is_help=True)


@data_cli.command()
def inspect(
    data_dir: str = typer.Option(
        ..., "--data-dir", "-d", help="Path to AudioSet data directory"
    ),
) -> None:
    """Print summary of CSV segments and ontology."""
    ontology_path = f"{data_dir}/class_labels_indices.csv"
    ontology = load_ontology(ontology_path)
    typer.echo(f"Ontology: {len(ontology)} classes")

    for split in ["eval", "balanced_train", "unbalanced_train"]:
        csv_path = f"{data_dir}/{split}_segments.csv"
        try:
            df = load_segments_csv(csv_path, split=split)
            typer.echo(f"{split}: {len(df)} segments")
        except FileNotFoundError:
            typer.echo(f"{split}: file not found")


@data_cli.command()
def download(
    data_dir: str = typer.Option(..., "--data-dir", "-d", help="Output directory"),
    region: str = typer.Option("us", help="Region: us, eu, or asia"),
) -> None:
    """Download AudioSet CSV and ontology. Features require gsutil rsync."""
    typer.echo(
        f"Download CSV and ontology to {data_dir}. "
        "For features, run: gsutil rsync -d -r features gs://{region}_audioset/youtube_corpus/v1/features"
    )
    typer.echo("See https://research.google.com/audioset/download.html")
