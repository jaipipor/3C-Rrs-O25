"""Shared pytest fixtures for the 3C-Rrs-O25 test suite."""

from pathlib import Path

import pytest

from rrs3c import rrs_model_3C_O25

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATA_FOLDER = REPOSITORY_ROOT / "data"


@pytest.fixture
def data_folder() -> Path:
    """Return the repository ancillary-data directory."""
    return DATA_FOLDER.resolve()


@pytest.fixture
def model(data_folder: Path) -> rrs_model_3C_O25:
    """Return a fresh model using the repository ancillary data."""
    return rrs_model_3C_O25(
        data_folder=data_folder,
    )
