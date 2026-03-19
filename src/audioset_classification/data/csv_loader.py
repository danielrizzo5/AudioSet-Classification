"""Parse AudioSet segment CSV files."""

import io
import re

import pandas as pd


def load_segments_csv(
    path: str,
    split: str | None = None,
    max_segments: int | None = None,
) -> pd.DataFrame:
    """
    Load an AudioSet segments CSV (eval, balanced_train, or unbalanced_train).

    CSV format: YTID, start_seconds, end_seconds, positive_labels
    Header lines start with #. The third header line defines columns.
    """
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    data_lines = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        data_lines.append(line)

    if not data_lines:
        return pd.DataFrame(
            columns=["ytid", "start_seconds", "end_seconds", "positive_labels"]
        )

    df = pd.read_csv(
        io.StringIO("\n".join(data_lines)),
        names=["ytid", "start_seconds", "end_seconds", "positive_labels"],
    )

    df["ytid"] = df["ytid"].astype(str).str.strip()
    df["start_seconds"] = pd.to_numeric(df["start_seconds"], errors="coerce")
    df["end_seconds"] = pd.to_numeric(df["end_seconds"], errors="coerce")
    df["positive_labels"] = df["positive_labels"].astype(str).str.strip()

    if split:
        df["split"] = split

    if max_segments is not None and len(df) > max_segments:
        df = df.iloc[:max_segments].copy()

    return df


def parse_positive_labels(labels_str: str) -> list[str]:
    """Parse positive_labels string into list of mid strings (e.g. /m/03v3yw,/m/0k4j)."""
    if not labels_str or pd.isna(labels_str):
        return []
    return [s.strip() for s in re.split(r'[",\s]+', labels_str) if s.strip()]


def segment_key(ytid: str, start_seconds: float, end_seconds: float) -> str:
    """Build a unique key for a segment (used for .pt file lookup)."""
    return f"{ytid}_{start_seconds:.3f}_{end_seconds:.3f}"
