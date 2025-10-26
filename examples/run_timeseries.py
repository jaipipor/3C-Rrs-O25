"""Example wrapper script to run the model on example data.

Usage:
    python examples/run_timeseries.py --data-folder data --input example_data_NIOZ_jetty_2.csv

This script uses the packaged `rrs_model_3C` in `src/rrs3c` and writes a small
NetCDF output for inspection. It intentionally mirrors your original pipeline
but keeps hard paths relative to `--data-folder` so it is easy to run in CI and
on Windows.
"""
import argparse
import os
import numpy as np
import pandas as pd
from rrs3c.model import rrs_model_3C
import lmfit as lm

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-folder", default="data", help="Folder containing vars_aph_v2.npz and G-tables")
    p.add_argument("--input", default="example_data_NIOZ_jetty_2.csv", help="CSV with example Li,Lt,Es columns")
    args = p.parse_args()

    data_folder = args.data_folder
    input_path = os.path.join(data_folder, args.input)

    data = pd.read_csv(input_path, index_col=0, skiprows=15)
    wl = data.index.values
    Li = data.iloc[:, 0].values
    Lt = data.iloc[:, 1].values
    Es = data.iloc[:, 2].values

    geom = (59, 35, 100)
    anc = (4, 60, 1013.25)

    model = rrs_model_3C(data_folder=data_folder)

    params = lm.Parameters()
    params.add_many(
        ("C", 5, True, 0.1, 50, None),
        ("N", 1, True, 0.01, 100, None),
        ("Y", 0.5, True, 0.01, 5, None),
        ("SNAP", 0.015, True, 0.005, 0.03, None),
        ("Sg", 0.015, True, 0.005, 0.03, None),
        ("rho", 0.02, False, 0, 0.03, None),
        ("rho_d", 0.0, True, 0, 10, None),
        ("rho_s", 0.0, True, -0.01, 0.01, None),
        ("alpha", 0.2, True, 0, 2, None),
        ("beta", 0.05, True, 0, 1, None),
    )

    weights = np.where((wl >= 760) & (wl <= 765), 0, np.where(wl < 400, 1, np.where(wl > 800, 5, 1)))

    out, Rrs_mod, Rg = model.fit_LtEs(wl, Li/Es, Lt/Es, params, weights, geom, anc)

    print("Fit completed. Inspect Rrs_mod and Rg arrays or write to file as needed.")

if __name__ == "__main__":
    main()
