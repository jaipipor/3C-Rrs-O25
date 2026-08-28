from pathlib import Path

import lmfit as lm
import numpy as np
import pytest

from rrs3c import rrs_model_3C_O25

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def model():
    return rrs_model_3C_O25(data_folder=ROOT / "data")


@pytest.fixture
def valid_inputs():
    return {
        "wl": np.array([500.0, 550.0, 600.0]),
        "LiEs": np.array([0.10, 0.11, 0.12]),
        "LtEs": np.array([0.02, 0.025, 0.03]),
        "params": lm.Parameters(),
        "weights": np.ones(3),
        "geom": (59.0, 35.0, 100.0),
        "anc": (4.0, 60.0, 1013.25),
        "verbose": False,
    }


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("wl", [[500.0, 550.0, 600.0]]),
        ("LiEs", [[0.10, 0.11, 0.12]]),
        ("LtEs", [[0.02, 0.025, 0.03]]),
        ("weights", [[1.0, 1.0, 1.0]]),
    ],
)
def test_rejects_non_one_dimensional_inputs(
    model,
    valid_inputs,
    name,
    value,
):
    inputs = valid_inputs.copy()
    inputs[name] = np.asarray(value)

    with pytest.raises(
        ValueError,
        match="must be 1D arrays",
    ):
        model.fit_LtEs(**inputs)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("LiEs", [0.10, 0.11]),
        ("LtEs", [0.02, 0.025]),
        ("weights", [1.0, 1.0]),
    ],
)
def test_rejects_mismatched_lengths(
    model,
    valid_inputs,
    name,
    value,
):
    inputs = valid_inputs.copy()
    inputs[name] = np.asarray(value)

    with pytest.raises(
        ValueError,
        match="must have the same shape",
    ):
        model.fit_LtEs(**inputs)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("wl", [500.0, np.nan, 600.0]),
        ("LiEs", [0.10, np.inf, 0.12]),
        ("LtEs", [0.02, np.nan, 0.03]),
        ("weights", [1.0, np.inf, 1.0]),
    ],
)
def test_rejects_nonfinite_values(
    model,
    valid_inputs,
    name,
    value,
):
    inputs = valid_inputs.copy()
    inputs[name] = np.asarray(value)

    with pytest.raises(
        ValueError,
        match="contains non-finite values",
    ):
        model.fit_LtEs(**inputs)


def test_rejects_negative_weights(model, valid_inputs):
    inputs = valid_inputs.copy()
    inputs["weights"] = np.array([1.0, -1.0, 1.0])

    with pytest.raises(
        ValueError,
        match="weights must be non-negative",
    ):
        model.fit_LtEs(**inputs)
