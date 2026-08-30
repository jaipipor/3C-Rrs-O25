from pathlib import Path

import numpy as np
import pytest

from rrs3c import rrs_model_3C_O25

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def model():
    return rrs_model_3C_O25(data_folder=ROOT / "data")


@pytest.mark.parametrize(
    ("solar_zenith", "view_zenith", "azimuth"),
    [
        (0.0, 0.0, 0.0),
        (30.0, 40.0, 60.0),
        (80.0, 70.0, 180.0),
        (87.5, 87.5, 180.0),
    ],
)
def test_g_evaluation_is_finite_at_table_nodes(
    model,
    solar_zenith,
    view_zenith,
    azimuth,
):
    values = model._G_eval(
        solar_zenith,
        view_zenith,
        azimuth,
    )

    assert len(values) == 4
    assert np.isfinite(values).all()
