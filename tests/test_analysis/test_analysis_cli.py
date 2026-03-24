"""Tests for analysis CLI."""

import re
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from audioset_classification.main import cli

runner = CliRunner()


def _strip_ansi(s: str) -> str:
    """Remove ANSI escape codes so assertions work regardless of terminal color settings."""
    return re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", s)


def test_umap_help_lists_all_splits():
    """UMAP command help mentions all-splits."""
    result = runner.invoke(cli, ["analysis", "umap", "--help"])
    assert result.exit_code == 0
    assert "--all-splits" in _strip_ansi(result.stdout)


def test_umap_all_splits_calls_combined_once(tmp_path: Path):
    """``--all-splits`` concatenates splits and runs one combined UMAP."""
    emb = tmp_path / "emb"
    emb.mkdir()
    for name in ("train.npz", "val.npz", "test.npz"):
        (emb / name).write_bytes(b"")

    with patch(
        "audioset_classification.cli.analysis_cli.run_umap_plots_combined",
        return_value=["a.png"],
    ) as mock_run:
        result = runner.invoke(
            cli,
            [
                "analysis",
                "umap",
                "--all-splits",
                "--embeddings-dir",
                str(emb),
                "--out-dir",
                str(tmp_path / "out"),
            ],
        )

    assert result.exit_code == 0
    assert mock_run.call_count == 1
    args, kwargs = mock_run.call_args
    assert list(args[0]) == [
        str(emb / "train.npz"),
        str(emb / "val.npz"),
        str(emb / "test.npz"),
    ]
    assert kwargs["output_stem"] == "all_splits"


def test_umap_all_splits_errors_when_no_npz(tmp_path: Path):
    """``--all-splits`` exits 1 if no split files exist."""
    emb = tmp_path / "empty"
    emb.mkdir()
    result = runner.invoke(
        cli,
        ["analysis", "umap", "--all-splits", "--embeddings-dir", str(emb)],
    )
    assert result.exit_code == 1
