"""Tests for data CLI."""

from typer.testing import CliRunner

from audioset_data.main import cli

runner = CliRunner()


def test_convert_help():
    """Convert command shows help."""
    result = runner.invoke(cli, ["convert", "--help"])
    assert result.exit_code == 0
