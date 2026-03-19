"""Load AudioSet class_labels_indices.csv ontology."""

import pandas as pd


def load_ontology(path: str) -> pd.DataFrame:
    """
    Load class_labels_indices.csv.

    Format: index,mid,display_name
    Maps integer index to mid (e.g. /m/09x0r) and display_name (e.g. Speech).
    """
    df = pd.read_csv(path)
    df = df.rename(columns=str.strip)
    result: pd.DataFrame = df.loc[:, ["index", "mid", "display_name"]].copy()
    return result


def mid_to_index(ontology: pd.DataFrame) -> dict[str, int]:
    """Build mid -> index mapping."""
    return dict(zip(ontology["mid"].astype(str), ontology["index"].astype(int)))


def index_to_mid(ontology: pd.DataFrame) -> dict[int, str]:
    """Build index -> mid mapping."""
    return dict(zip(ontology["index"].astype(int), ontology["mid"].astype(str)))
