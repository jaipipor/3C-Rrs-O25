#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple runner for the 3C Rrs time-series example.

Assumptions:
 - This script is at repo/examples/timeseries/src/run_timeseries.py
 - The package model is at repo/src/rrs3c/model.py
 - Helper routines (load_jetty_data, solar_az_el, flags_jetty) are in utils.py
   located next to this script: repo/examples/timeseries/src/utils.py

Usage (examples):
    python run_timeseries.py
    python run_timeseries.py --input-file example_time_series_data.csv --output-folder ../output --plot
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import lmfit as lm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from utils import flags_jetty, load_jetty_data, solar_az_el

# Directory containing this script
SCRIPT_DIR = Path(__file__).resolve().parent

# Find the repo root by looking for 'src/rrs3c' in parent folders
repo_root = next(
    (p for p in SCRIPT_DIR.parents if (p / "src" / "rrs3c").exists()), None
)

# Fallback: if not found, try the common relative location (three levels up)
if repo_root is None:
    repo_root = SCRIPT_DIR.parents[3] if len(SCRIPT_DIR.parents) > 3 else None

if repo_root is not None:
    SRC_DIR = repo_root / "src"
    if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
        # insert at front so local package wins over any installed package
        sys.path.insert(0, str(SRC_DIR))
else:
    # No repo root detected — keep going, but imports may fail with a clear error
    # (we don't raise here to keep the script usable in unusual setups)
    pass

from rrs3c.model import rrs_model_3C  # noqa: E402


# ---------------------------------------------------------------------------
# CLI and logging
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Process a jetty time series with 3C (O25)."
    )
    p.add_argument(
        "--input-file",
        default="example_time_series_data.csv",
        help="CSV file name (default: example_time_series_data.csv next to this script).",
    )
    p.add_argument(
        "--input-folder",
        default=str(SCRIPT_DIR),
        help="Folder containing the input CSV (default: script folder).",
    )
    p.add_argument(
        "--output-folder",
        default=str(SCRIPT_DIR / ".." / "output"),
        help="Folder where outputs will be written (default: ../output).",
    )
    p.add_argument("--date", default=None, help="Date tag for outputs (e.g. 20200530).")
    p.add_argument("--plot", action="store_true", help="Save diagnostic PNGs.")
    p.add_argument("--verbose", action="store_true", help="Debug logging.")
    return p.parse_args()


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s: %(message)s")


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    input_folder = Path(args.input_folder).expanduser().resolve()
    input_path = input_folder / ".." / "data" / args.input_file
    output_folder = Path(args.output_folder).expanduser().resolve()
    output_folder.mkdir(parents=True, exist_ok=True)

    date_tag = args.date or datetime.now().strftime("%Y%m%d")

    logging.info("Input file: %s", input_path)
    logging.info("Output folder: %s", output_folder)
    logging.info("Date tag: %s", date_tag)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Load raw spectra using provided helper (keeps original contract)
    instrument, variable, times, l1, l_end, delta_l, spec = load_jetty_data(input_path)

    # Convert times to pandas datetime if needed
    if not isinstance(times, pd.DatetimeIndex):
        times = pd.to_datetime(times, format="%Y%m%d %H%M%S")

    # Convert times to pandas datetime and factorize into unique slots
    times_pd = pd.to_datetime(times, format="%Y%m%d %H%M%S")
    idx_inv, uniques = pd.factorize(times_pd)
    unique_times = uniques.to_pydatetime()
    anc = (4, 60, 1013.25)
    N = len(unique_times)
    wl = np.arange(350, 921)
    nl = len(wl)

    # Pre-allocate matrices for Es, Li, Lt, Rrs, etc.
    Es_mat = np.full((N, nl), np.nan)
    Li_east = np.full((N, nl), np.nan)
    Lt_east = np.full((N, nl), np.nan)
    Rrs_east = np.full((N, nl), np.nan)
    Rrs_mod_east = np.full((N, nl), np.nan)
    Rg_east = np.full((N, nl), np.nan)

    Li_west = np.full((N, nl), np.nan)
    Lt_west = np.full((N, nl), np.nan)
    Rrs_west = np.full((N, nl), np.nan)
    Rrs_mod_west = np.full((N, nl), np.nan)
    Rg_west = np.full((N, nl), np.nan)

    # --- Diagnostics and storage for fit objects (initialized before the loop) ---
    idx550 = np.argmin(np.abs(wl - 550))
    resid_rmse_east = np.full(N, np.nan)
    resid_rmse_west = np.full(N, np.nan)
    resid550_east = np.full(N, np.nan)
    resid550_west = np.full(N, np.nan)
    rho_east = np.full(N, np.nan)
    rho_d_east = np.full(N, np.nan)
    rho_s_east = np.full(N, np.nan)
    rho_west = np.full(N, np.nan)
    rho_d_west = np.full(N, np.nan)
    rho_s_west = np.full(N, np.nan)
    Rg550_east = np.full(N, np.nan)
    Rg550_west = np.full(N, np.nan)

    # Lists to keep whole fit Result objects for later inspection
    res_east_list = [None] * N
    res_west_list = [None] * N

    # Build 3C model instance
    model = rrs_model_3C()

    start_time = time.time()
    # Loop over unique time bins
    for i, t0 in enumerate(unique_times):
        mask = idx_inv == i
        vars_i = variable[mask]

        # select Es tag
        if "ES" in vars_i:
            tag_es = "ES"
        elif "ES-VIS" in vars_i:
            tag_es = "ES-VIS"
        else:
            continue

        # Extract Es spectrum and interpolate to global wl
        idx_es = np.where((variable == tag_es) & mask)[0]
        l_start, l_stop = float(l1[idx_es[0]]), float(l_end[idx_es[0]])
        raw = spec[idx_es, : int(l_stop - l_start + 1)][0]
        # build local wavelength axis for this spectrum
        wl_row = np.linspace(l_start, l_stop, raw.size)
        # interpolate onto global wl grid
        Es_interp = np.interp(wl, wl_row, raw)
        Es_mat[i, :] = Es_interp

        # Flag dark/red
        dark, red, anom = flags_jetty(wl, Es_interp)
        # Compute solar position
        Az, El = solar_az_el(t0, lat=53.001788, lon=4.789151)

        if not dark and not red and El >= 5:
            # process east and west
            for side in ("EAST", "WEST"):
                sky_tag = f"LSKY-{side}"
                surf_tag = f"LSFC-{side}"
                if sky_tag in vars_i and surf_tag in vars_i:
                    # extract raw Li, Lt
                    idx_ls = np.where((variable == sky_tag) & mask)[0]
                    idx_lu = np.where((variable == surf_tag) & mask)[0]
                    raw_Li = spec[idx_ls, : int(l_stop - l_start + 1)][0]
                    raw_Lt = spec[idx_lu, : int(l_stop - l_start + 1)][0]

                    # build local wavelength axis and interpolate to global wl
                    wl_row = np.linspace(l_start, l_stop, raw_Li.size)
                    Li_interp = np.interp(wl, wl_row, raw_Li)
                    Lt_interp = np.interp(wl, wl_row, raw_Lt)

                    # store interpolated spectra
                    # fit 3C model
                    params = lm.Parameters()
                    params.add_many(
                        ("C", 5, True, 0.1, 50, None),
                        ("N", 1, True, 0.01, 100, None),
                        ("Y", 0.5, True, 0.01, 5, None),
                        ("SNAP", 0.015, True, 0.005, 0.03, None),
                        ("Sg", 0.015, True, 0.005, 0.03, None),
                        ("rho", 0.02, True, 0, 0.03, None),
                        ("rho_d", 0.0, True, 0, 10, None),
                        ("rho_s", 0.01, True, -0.1, 0.1, None),
                        ("delta", 0.0, False, -0.01, 0.01, None),
                        ("alpha", 0.2, True, 0, 2, None),
                        ("beta", 0.05, True, 0, 1, None),
                    )

                    weights = np.where(
                        (wl >= 750) & (wl <= 775),
                        0,  # Remove the ugly H2O feature
                        np.where(
                            wl < 400,
                            1,  # Some potential troubles at the UV
                            np.where(wl > 800, 5, 1),
                        ),
                    )  # Let's power-up the NIR

                    geom = (90 - El, 35, abs(Az - (135 if side == "EAST" else 225)))

                    res, Rrs_mod, Rg = model.fit_LtEs(
                        wl,
                        Li_interp / Es_interp,
                        Lt_interp / Es_interp,
                        params,
                        weights,
                        geom,
                        anc,
                    )
                    LtEs_obs = Lt_interp / Es_interp
                    LtEs_mod = Rrs_mod + Rg

                    # Compute residual in Lt/Es space as requested by the user
                    # resid_vec = LtEs_mod - LtEs_obs
                    resid_vec = LtEs_mod - LtEs_obs
                    rmse_val = np.sqrt(np.nanmean((resid_vec) ** 2))
                    resid550_val = resid_vec[idx550]

                    # Observed and modelled Rrs (Lt/Es - Rg)
                    Rrs_obs = LtEs_obs - Rg

                    # Store into pre-allocated arrays and diagnostics
                    if side == "EAST":
                        Li_east[i, :] = Li_interp
                        Lt_east[i, :] = Lt_interp
                        Rrs_east[i, :] = Rrs_obs
                        Rrs_mod_east[i, :] = Rrs_mod
                        Rg_east[i, :] = Rg

                        resid_rmse_east[i] = rmse_val
                        resid550_east[i] = resid550_val
                        Rg550_east[i] = Rg[idx550]

                        # save fitted parameter values safely
                        try:
                            rho_east[i] = res.params["rho"].value
                            rho_d_east[i] = res.params["rho_d"].value
                            rho_s_east[i] = res.params["rho_s"].value
                        except Exception:
                            pass

                        # save full Result object for later inspection
                        res_east_list[i] = res

                    else:  # WEST
                        Li_west[i, :] = Li_interp
                        Lt_west[i, :] = Lt_interp
                        Rrs_west[i, :] = Rrs_obs
                        Rrs_mod_west[i, :] = Rrs_mod
                        Rg_west[i, :] = Rg

                        resid_rmse_west[i] = rmse_val
                        resid550_west[i] = resid550_val
                        Rg550_west[i] = Rg[idx550]

                        try:
                            rho_west[i] = res.params["rho"].value
                            rho_d_west[i] = res.params["rho_d"].value
                            rho_s_west[i] = res.params["rho_s"].value
                        except Exception:
                            pass

                        res_west_list[i] = res

    end_time = time.time()
    elapsed_time = end_time - start_time  # in seconds
    logging.info("Processing finished in %.2f s", elapsed_time)

    # Build minimal xarray dataset and save
    coords = {"time": unique_times, "wavelength": wl}
    ds = xr.Dataset(
        {
            "Es": (("time", "wavelength"), Es_mat),
            "Rrs_east": (("time", "wavelength"), Rrs_east),
            "Rg_east": (("time", "wavelength"), Rg_east),
            "resid_rmse_east": (("time",), resid_rmse_east),
            "Rrs_west": (("time", "wavelength"), Rrs_west),
            "Rg_west": (("time", "wavelength"), Rg_west),
            "resid_rmse_west": (("time",), resid_rmse_west),
        },
        coords=coords,
        attrs={"created": datetime.now().isoformat()},
    )

    out_file = output_folder / f"Rrs_timeseries_{date_tag}.nc"
    ds.to_netcdf(out_file)
    logging.info("Saved NetCDF to %s", out_file)

    if args.plot:
        try:
            idx550 = int(np.argmin(np.abs(wl - 550)))
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(unique_times, Es_mat[:, idx550], ".-", label="Es@550")
            ax.plot(unique_times, Rrs_east[:, idx550], ".-", label="Rrs_east@550")
            ax.plot(unique_times, Rrs_west[:, idx550], ".-", label="Rrs_west@550")
            ax.legend()
            ax.grid(True)
            fig.autofmt_xdate()
            fig.savefig(output_folder / f"timeseries_550nm_{date_tag}.png", dpi=150)
            plt.close(fig)
            logging.info("Saved plot to %s", output_folder)
        except Exception:
            logging.exception("Failed to save plot")

    logging.info("Done.")


if __name__ == "__main__":
    main()
