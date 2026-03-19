"""CLI entry point for data processing."""

import typer

cli = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)


@cli.command()
def convert(
    tfrecord_dir: str = typer.Option(
        ..., "--tfrecord-dir", "-i", help="Input TFRecord directory"
    ),
    output_dir: str = typer.Option(
        ..., "--output-dir", "-o", help="Output .pt directory"
    ),
) -> None:
    """Convert TFRecord embeddings to .pt files (placeholder)."""
    typer.echo(f"Convert {tfrecord_dir} -> {output_dir} (not yet implemented)")


if __name__ == "__main__":
    cli()
