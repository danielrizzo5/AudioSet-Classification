"""CLI entry point."""

import typer

from audioset_classification.cli.data_cli import data_cli
from audioset_classification.cli.train_cli import run_train
from audioset_classification.utils.logging import configure_logger

cli = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
cli.add_typer(data_cli, name="data")
cli.command(name="train")(run_train)


@cli.callback(invoke_without_command=True)
def init(
    log_level: str = typer.Option("INFO", help="Logging level", envvar="LOG_LEVEL"),
) -> None:
    configure_logger(log_level)


if __name__ == "__main__":
    cli()
