"""Tests for data CLI."""

from typer.testing import CliRunner

from audioset_classification.cli.data_cli import data_cli

runner = CliRunner()


def test_download_help():
    """Download command shows help."""
    result = runner.invoke(data_cli, ["download", "--help"])
    assert result.exit_code == 0


def test_manifest_help():
    """Manifest command shows help."""
    result = runner.invoke(data_cli, ["manifest", "--help"])
    assert result.exit_code == 0


def test_features_help():
    """Features command shows help."""
    result = runner.invoke(data_cli, ["features", "--help"])
    assert result.exit_code == 0


def test_inspect_help():
    """Inspect command shows help."""
    result = runner.invoke(data_cli, ["inspect", "--help"])
    assert result.exit_code == 0
