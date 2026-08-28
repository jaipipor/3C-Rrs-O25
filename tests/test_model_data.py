from pathlib import Path

import numpy as np

from rrs3c import rrs_model_3C_O25

ROOT = Path(__file__).resolve().parents[1]


def test_ancillary_arrays_are_coherent():
    model = rrs_model_3C_O25(
        data_folder=ROOT / "data",
    )

    water = model.lw_aw_bw

    assert water.ndim == 2
    assert water.shape[1] == 3
    assert np.isfinite(water).all()
    assert np.all(np.diff(water[:, 0]) > 0)
    assert np.all(water[:, 1:] >= 0)

    assert model.l_int.ndim == 1
    assert model._aph_norm_55.ndim == 2
    assert model._aph670_bounds.ndim == 1

    assert model._aph_norm_55.shape[1] == model.l_int.size
    assert model._aph_norm_55.shape[0] == model._aph670_bounds.size + 1

    assert np.isfinite(model.l_int).all()
    assert np.isfinite(model._aph_norm_55).all()
    assert np.isfinite(model._aph670_bounds).all()

    assert np.all(np.diff(model.l_int) > 0)
    assert np.all(np.diff(model._aph670_bounds) > 0)

    expected_g_shape = (
        model._G_ts.size,
        model._G_tv.size,
        model._G_az.size,
    )

    for table in (
        model._G0w,
        model._G1w,
        model._G0p,
        model._G1p,
    ):
        assert table.shape == expected_g_shape
        assert np.isfinite(table).all()

    assert np.all(np.diff(model._G_ts) > 0)
    assert np.all(np.diff(model._G_tv) > 0)
    assert np.all(np.diff(model._G_az) > 0)


def test_water_iop_table_structure(model):
    water_iops = model.lw_aw_bw

    assert water_iops.ndim == 2
    assert water_iops.shape[1] == 3
    assert water_iops.shape[0] > 1


def test_water_iop_table_contains_finite_values(model):
    assert np.all(np.isfinite(model.lw_aw_bw))


def test_water_iop_wavelengths_are_strictly_increasing(model):
    wavelengths = model.lw_aw_bw[:, 0]

    assert np.all(np.diff(wavelengths) > 0)


def test_water_iop_sentinel_is_removed(model):
    assert not np.array_equal(
        model.lw_aw_bw[-1],
        np.array([-1.0, -1.0, -1.0]),
    )


def test_water_iop_values_are_nonnegative(model):
    absorption = model.lw_aw_bw[:, 1]
    backscattering = model.lw_aw_bw[:, 2]

    assert np.all(absorption >= 0)
    assert np.all(backscattering >= 0)
