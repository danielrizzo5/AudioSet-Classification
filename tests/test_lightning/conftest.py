"""Lightning test setup."""

import warnings

import pytest


@pytest.fixture(autouse=True)
def _suppress_lightning_warnings():
    """Quiet down expected Lightning/torch warnings in tests."""
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
