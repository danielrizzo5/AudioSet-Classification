"""Parse AudioSet segment CSV files."""

import re

import pandas as pd

COLUMNS = ["ytid", "start_seconds", "end_seconds", "positive_labels"]


def load_segments_csv(
    path: str,
    split: str | None = None,
    max_segments: int | None = None,
) -> pd.DataFrame:
    """Load an AudioSet segments CSV (eval, balanced_train, or unbalanced_train).

    The positive_labels field is unquoted and may itself contain commas, so
    each line is split on the first three commas only.

    CSV format: YTID, start_seconds, end_seconds, positive_labels
    """
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    rows = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",", 3)
        if len(parts) == 4:
            rows.append(parts)

    if not rows:
        return pd.DataFrame(columns=COLUMNS)

    df = pd.DataFrame(rows, columns=COLUMNS)

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
