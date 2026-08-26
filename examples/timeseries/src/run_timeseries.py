"""Tests for the time-series example utilities."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

TIMESERIES_SOURCE = REPOSITORY_ROOT / "examples" / "timeseries" / "src"

sys.path.insert(
    0,
    str(TIMESERIES_SOURCE),
)

from utils import (  # noqa: E402
    angular_difference_deg,
    interpolate_spectrum,
)


def test_interpolation_uses_each_record_wavelength_metadata():
    """Each row must use its own wavelength origin."""
    l1 = np.array(
        [
            320.0,
            317.0,
        ]
    )

    l_end = np.array(
        [
            950.0,
            950.0,
        ]
    )

    delta_l = np.array(
        [
            1.0,
            1.0,
        ]
    )

    spectral_values = np.full(
        (2, 634),
        np.nan,
    )

    es_wavelengths = np.arange(
        320.0,
        951.0,
    )

    west_surface_wavelengths = np.arange(
        317.0,
        951.0,
    )

    # Using wavelength as the synthetic measurement makes any spectral
    # shift directly visible in the interpolated result.
    spectral_values[
        0,
        : es_wavelengths.size,
    ] = es_wavelengths

    spectral_values[
        1,
        : west_surface_wavelengths.size,
    ] = west_surface_wavelengths

    target = np.array(
        [
            350.0,
            500.0,
            920.0,
        ]
    )

    es_result = interpolate_spectrum(
        row_index=0,
        l1=l1,
        l_end=l_end,
        delta_l=delta_l,
        spec=spectral_values,
        target_wavelengths=target,
    )

    west_result = interpolate_spectrum(
        row_index=1,
        l1=l1,
        l_end=l_end,
        delta_l=delta_l,
        spec=spectral_values,
        target_wavelengths=target,
    )

    np.testing.assert_allclose(
        es_result,
        target,
    )

    np.testing.assert_allclose(
        west_result,
        target,
    )


def test_interpolation_rejects_extrapolation():
    """The helper must not silently use endpoint extrapolation."""
    l1 = np.array([400.0])
    l_end = np.array([700.0])
    delta_l = np.array([1.0])

    source_wavelengths = np.arange(
        400.0,
        701.0,
    )

    spectral_values = source_wavelengths[np.newaxis, :]

    with pytest.raises(
        ValueError,
        match="extends beyond",
    ):
        interpolate_spectrum(
            row_index=0,
            l1=l1,
            l_end=l_end,
            delta_l=delta_l,
            spec=spectral_values,
            target_wavelengths=np.array(
                [
                    350.0,
                    500.0,
                ]
            ),
        )


def test_interpolation_rejects_inconsistent_metadata():
    """The metadata must define an integer sample count."""
    with pytest.raises(
        ValueError,
        match="integer number of samples",
    ):
        interpolate_spectrum(
            row_index=0,
            l1=np.array([320.0]),
            l_end=np.array([950.0]),
            delta_l=np.array([1.3]),
            spec=np.ones((1, 600)),
            target_wavelengths=np.array(
                [
                    400.0,
                    500.0,
                ]
            ),
        )


@pytest.mark.parametrize(
    (
        "angle_a",
        "angle_b",
        "expected",
    ),
    [
        (0.0, 225.0, 135.0),
        (350.0, 10.0, 20.0),
        (10.0, 350.0, 20.0),
        (90.0, 270.0, 180.0),
        (45.0, 45.0, 0.0),
        (725.0, 5.0, 0.0),
    ],
)
def test_angular_difference_deg(
    angle_a,
    angle_b,
    expected,
):
    """Angular differences must use the shortest circular distance."""
    assert (
        angular_difference_deg(
            angle_a,
            angle_b,
        )
        == expected
    )


def test_angular_difference_rejects_nonfinite_values():
    """Nonfinite geometry values must be rejected."""
    with pytest.raises(
        ValueError,
        match="finite",
    ):
        angular_difference_deg(
            np.nan,
            135.0,
        )
