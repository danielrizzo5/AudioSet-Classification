"""Write and read JSONL manifests for AudioSet clips."""

import json
import os

import pandas as pd

from audioset_classification.data.csv_loader import parse_positive_labels
from audioset_classification.data.ontology import mid_to_index


def write_manifest(
    df: pd.DataFrame,
    ontology: pd.DataFrame,
    audio_dir: str,
    manifest_path: str,
) -> int:
    """Write a JSONL manifest from a segments DataFrame.

    Only rows with a valid 'audio_path' column value (non-null, file exists)
    are written. Returns the number of entries written.

    Each line:
        {"audio_path": "...", "labels": ["Speech"], "label_ids": [0],
         "ytid": "abc", "start": 30.0, "end": 40.0}

    Args:
        df: Segments DataFrame; must have ytid, start_seconds, end_seconds,
            positive_labels, and audio_path columns.
        ontology: Ontology DataFrame from load_ontology().
        audio_dir: Base directory for audio files (used to make paths relative).
        manifest_path: Output .jsonl file path.
    """
    mid_to_idx = mid_to_index(ontology)
    name_map: dict[str, str] = dict(
        zip(ontology["mid"].astype(str), ontology["display_name"].astype(str))
    )
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    written = 0
    with open(manifest_path, "w", encoding="utf-8") as f:
        for record in df.to_dict("records"):
            audio_path = record.get("audio_path")
            if not audio_path or not os.path.exists(str(audio_path)):
                continue

            mids = parse_positive_labels(str(record["positive_labels"]))
            label_ids = [mid_to_idx[m] for m in mids if m in mid_to_idx]
            labels = [name_map.get(m, m) for m in mids if m in mid_to_idx]

            entry = {
                "audio_path": str(audio_path),
                "labels": labels,
                "label_ids": label_ids,
                "ytid": str(record["ytid"]),
                "start": float(record["start_seconds"]),
                "end": float(record["end_seconds"]),
            }
            f.write(json.dumps(entry) + "\n")
            written += 1

    return written


def read_manifest(manifest_path: str) -> list[dict]:
    """Read a JSONL manifest into a list of dicts."""
    with open(manifest_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
