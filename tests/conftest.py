"""Shared pytest fixtures for the 3C-Rrs-O25 test suite."""

from pathlib import Path

import pytest

from rrs3c import rrs_model_3C_O25

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATA_FOLDER = REPOSITORY_ROOT / "data"


@pytest.fixture
def model() -> rrs_model_3C_O25:
    """Return a fresh model instance using the repository data files."""
    return rrs_model_3C_O25(
        data_folder=DATA_FOLDER,
    )
