from pathlib import Path

import lmfit as lm
import numpy as np
import pandas as pd

from rrs3c import rrs_model_3C_O25

ROOT = Path(__file__).resolve().parents[1]


def test_single_spectrum_regression():
    data = pd.read_csv(
        ROOT / "examples" / "example_single_spectrum.csv",
        skiprows=15,
        index_col=0,
    )
    wl = data.index.to_numpy(dtype=float)
    li = data.iloc[:, 0].to_numpy(dtype=float)
    lt = data.iloc[:, 1].to_numpy(dtype=float)
    es = data.iloc[:, 2].to_numpy(dtype=float)

    li_es = li / es
    lt_es = lt / es

    weights = np.ones_like(wl)
    weights[(wl >= 760) & (wl <= 765)] = 0
    weights[wl > 800] = 5

    params = lm.Parameters()
    params.add_many(
        ("C", 5.0, True, 0.1, 50.0),
        ("N", 1.0, True, 0.01, 100.0),
        ("Y", 0.5, True, 0.01, 5.0),
        ("SNAP", 0.015, True, 0.005, 0.03),
        ("Sg", 0.015, True, 0.005, 0.03),
        ("rho", 0.02, False, 0.0, 0.03),
        ("rho_d", 0.0, True, 0.0, 10.0),
        ("rho_s", 0.0, True, -0.01, 0.01),
        ("alpha", 0.2, True, 0.0, 2.0),
        ("beta", 0.05, True, 0.0, 1.0),
    )

    model = rrs_model_3C_O25(data_folder=ROOT / "data")

    result, rrs_model, rg = model.fit_LtEs(
        wl=wl,
        LiEs=li_es,
        LtEs=lt_es,
        params=params,
        weights=weights,
        geom=(59.0, 35.0, 100.0),
        anc=(4.0, 60.0, 1013.25),
        verbose=False,
    )

    assert result.success, result.message
    assert rrs_model.shape == rg.shape == wl.shape
    assert np.isfinite(rrs_model).all()
    assert np.isfinite(rg).all()

    expected = {
        "C": 2.80776,
        "N": 5.68994,
        "Y": 0.598448,
        "SNAP": 0.028786,
        "Sg": 0.0100113,
        "rho": 0.02,
        "rho_d": 0.000950574,
        "rho_s": 0.00596296,
        "alpha": 0.200014,
        "beta": 0.0499663,
    }

    fitted = {name: result.params[name].value for name in expected}

    np.testing.assert_allclose(
        list(fitted.values()),
        list(expected.values()),
        rtol=5e-4,
        atol=1e-8,
    )
