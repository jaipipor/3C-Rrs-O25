#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Process a time series of jetty spectra with the 3C model (O25 variant).

This script mirrors the original processing logic but:
 - Defaults to loading 'example_time_series_data.csv' from the examples folder.
 - Provides CLI flags to override input/output paths.
 - Attempts to import the model from two common locations (local module or package).
 - Logs progress and timing, and continues on single-fit errors.
 - Saves results to a NetCDF and writes diagnostic PNGs into the output folder.

Usage (from repo root, with venv active):
    python examples/process_timeseries.py
    python examples/process_timeseries.py --input-file example_time_series_data.csv
    python examples/process_timeseries.py --input /full/path/to/20200530_jetty_awo_spectra.csv --output-folder C:/path/to/output
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import datetime

import lmfit as lm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# Try to import helper modules and the model. Keep original imports but provide
# a fallback to the package import if available inside `src/rrs3c`.
try:
    from load_jetty_data import load_jetty_data
except Exception:
    # If running in the repo, ensure src is importable and try again
    import sys

    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
    )
    from load_jetty_data import load_jetty_data  # type: ignore

# try the exact model module you used; if not present try packaged import
try:
    from model import rrs_model_3C  # original working module name
except Exception:
    try:
        from rrs3c.model import rrs_model_3C  # fallback if installed/packaged
    except Exception as e:
        raise ImportError(
            "Could not import rrs_model_3C. Make sure your working model module is on PYTHONPATH "
            "or installed in the environment. Original error: %s" % e
        )

try:
    from solar_az_el import solar_az_el
except Exception:
    # solar_az_el should be present in examples or src; raise if missing
    raise ImportError(
        "Missing helper 'solar_az_el'. Place it next to this script or in src/."
    )

try:
    from flags_jetty import flags_jetty
except Exception:
    raise ImportError(
        "Missing helper 'flags_jetty'. Place it next to this script or in src/."
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Process jetty time series with 3C model (O25)."
    )
    p.add_argument(
        "--input-file",
        default="example_time_series_data.csv",
        help="Input CSV filename (default: example_time_series_data.csv in examples/).",
    )
    p.add_argument(
        "--input-folder",
        default=os.path.abspath(os.path.dirname(__file__)),
        help="Folder containing the input CSV (default: script folder).",
    )
    p.add_argument(
        "--output-folder",
        default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "output")
        ),
        help="Folder to write NetCDF and plots (default: ../output).",
    )
    p.add_argument(
        "--date",
        default=None,
        help="Optional date string used to build output filename (e.g. 20200530). If not given the script will use today.",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="Save PNG diagnostic plots to output folder.",
    )
    p.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return p.parse_args()


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=level)


def ensure_folder(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    args = parse_args()
    setup_logging(args.verbose)

    input_folder = os.path.abspath(args.input_folder)
    input_path = os.path.join(input_folder, args.input_file)
    output_folder = os.path.abspath(args.output_folder)
    ensure_folder(output_folder)

    my_date = args.date or datetime.now().strftime("%Y%m%d")
    logging.info("Input: %s", input_path)
    logging.info("Output folder: %s", output_folder)
    logging.info("Date tag: %s", my_date)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # --- Load raw data using your helper (keeps original contract) ---
    instrument, variable, times, l1, l_end, delta_l, spec = load_jetty_data(input_path)

    # Convert times to pandas datetime if needed
    if not isinstance(times, pd.DatetimeIndex):
        # expected format in your pipeline: 'YYYYmmdd HHMMSS'
        times = pd.to_datetime(times, format="%Y%m%d %H%M%S")

    # factorize into unique slots and build list of unique datetimes
    times_pd = pd.to_datetime(times, format="%Y%m%d %H%M%S")
    idx_inv, uniques = pd.factorize(times_pd)
    unique_times = uniques.to_pydatetime()

    # Ancillary atmospheric parameters used by the model
    anc = (4, 60, 1013.25)

    N = len(unique_times)
    wl = np.arange(350, 921)
    nl = len(wl)

    # Pre-allocate arrays (same shapes as your original)
    Es_mat = np.full((N, nl), np.nan)
    Ls_east = np.full((N, nl), np.nan)
    Lu_east = np.full((N, nl), np.nan)
    Rrs_east = np.full((N, nl), np.nan)
    Rrs_mod_east = np.full((N, nl), np.nan)
    Rg_east = np.full((N, nl), np.nan)

    Ls_west = np.full((N, nl), np.nan)
    Lu_west = np.full((N, nl), np.nan)
    Rrs_west = np.full((N, nl), np.nan)
    Rrs_mod_west = np.full((N, nl), np.nan)
    Rg_west = np.full((N, nl), np.nan)

    # Diagnostics arrays
    idx550 = np.argmin(np.abs(wl - 550))
    resid_rmse_east = np.full(N, np.nan)
    resid_rmse_west = np.full(N, np.nan)
    resid550_east = np.full(N, np.nan)
    resid550_west = np.full(N, np.nan)
    rho_s_east = np.full(N, np.nan)
    rho_dd_east = np.full(N, np.nan)
    rho_ds_east = np.full(N, np.nan)
    rho_s_west = np.full(N, np.nan)
    rho_dd_west = np.full(N, np.nan)
    rho_ds_west = np.full(N, np.nan)
    Rg550_east = np.full(N, np.nan)
    Rg550_west = np.full(N, np.nan)

    res_east_list = [None] * N
    res_west_list = [None] * N

    # Build model instance (data_folder param can be provided inside rrs_model_3C if needed)
    model = rrs_model_3C()

    start_time = time.time()
    # Loop over unique time bins
    for i, t0 in enumerate(unique_times):
        t_iter_start = time.time()
        mask = idx_inv == i
        vars_i = variable[mask]

        # select Es tag (preserve original priority ES -> ES-VIS)
        if "ES" in vars_i:
            tag_es = "ES"
        elif "ES-VIS" in vars_i:
            tag_es = "ES-VIS"
        else:
            logging.debug("No Es present for time index %d (%s). Skipping.", i, t0)
            continue

        # Extract Es spectrum and interpolate to global wl
        idx_es = np.where((variable == tag_es) & mask)[0]
        l_start, l_stop = float(l1[idx_es[0]]), float(l_end[idx_es[0]])
        raw = spec[idx_es, : int(l_stop - l_start + 1)][0]
        wl_row = np.linspace(l_start, l_stop, raw.size)
        Es_interp = np.interp(wl, wl_row, raw)
        Es_mat[i, :] = Es_interp

        # Flag dark/red and compute solar geometry
        dark, red, anom = flags_jetty(wl, Es_interp)
        Az, El = solar_az_el(t0, lat=53.001788, lon=4.789151)

        # Only proceed if the Es measurement is valid and sun elevation >= 5 deg
        if not dark and not red and El >= 5:
            for side in ("EAST", "WEST"):
                sky_tag = f"LSKY-{side}"
                surf_tag = f"LSFC-{side}"
                if sky_tag in vars_i and surf_tag in vars_i:
                    # extract raw Ls, Lu
                    idx_ls = np.where((variable == sky_tag) & mask)[0]
                    idx_lu = np.where((variable == surf_tag) & mask)[0]
                    raw_Ls = spec[idx_ls, : int(l_stop - l_start + 1)][0]
                    raw_Lu = spec[idx_lu, : int(l_stop - l_start + 1)][0]

                    # interpolate to global wl
                    wl_row_loc = np.linspace(l_start, l_stop, raw_Ls.size)
                    Ls_interp = np.interp(wl, wl_row_loc, raw_Ls)
                    Lu_interp = np.interp(wl, wl_row_loc, raw_Lu)

                    # prepare parameters for this fit (same as your O25 block)
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

                    # weighting vector (preserves your choices)
                    weights = np.where(
                        (wl >= 750) & (wl <= 775),
                        0,
                        np.where(wl < 400, 1, np.where(wl > 800, 5, 1)),
                    )

                    geom = (90 - El, 35, abs(Az - (135 if side == "EAST" else 225)))

                    # call the model fit inside a try/except so one failure doesn't abort the run
                    try:
                        res, Rrs_mod, Rg = model.fit_LtEs(
                            wl,
                            Ls_interp / Es_interp,
                            Lu_interp / Es_interp,
                            params,
                            weights,
                            geom,
                            anc,
                        )
                    except Exception as exc:
                        logging.exception(
                            "Fit failed at time %s side %s (index %d): %s",
                            t0,
                            side,
                            i,
                            exc,
                        )
                        # store NaNs and continue
                        continue

                    LuEd_obs = Lu_interp / Es_interp
                    LuEd_mod = Rrs_mod + Rg

                    resid_vec = LuEd_mod - LuEd_obs
                    rmse_val = np.sqrt(np.nanmean(resid_vec**2))
                    resid550_val = resid_vec[idx550]

                    Rrs_obs = LuEd_obs - Rg

                    # Store results into arrays
                    if side == "EAST":
                        Ls_east[i, :] = Ls_interp
                        Lu_east[i, :] = Lu_interp
                        Rrs_east[i, :] = Rrs_obs
                        Rrs_mod_east[i, :] = Rrs_mod
                        Rg_east[i, :] = Rg

                        resid_rmse_east[i] = rmse_val
                        resid550_east[i] = resid550_val
                        Rg550_east[i] = Rg[idx550]

                        try:
                            rho_s_east[i] = res.params["rho_s"].value
                            rho_dd_east[i] = res.params["rho_dd"].value
                            rho_ds_east[i] = res.params["rho_ds"].value
                        except Exception:
                            pass

                        res_east_list[i] = res
                    else:  # WEST
                        Ls_west[i, :] = Ls_interp
                        Lu_west[i, :] = Lu_interp
                        Rrs_west[i, :] = Rrs_obs
                        Rrs_mod_west[i, :] = Rrs_mod
                        Rg_west[i, :] = Rg

                        resid_rmse_west[i] = rmse_val
                        resid550_west[i] = resid550_val
                        Rg550_west[i] = Rg[idx550]

                        try:
                            rho_s_west[i] = res.params["rho_s"].value
                            rho_dd_west[i] = res.params["rho_dd"].value
                            rho_ds_west[i] = res.params["rho_ds"].value
                        except Exception:
                            pass

                        res_west_list[i] = res

        # per-iteration timing log
        t_iter_end = time.time()
        logging.info(
            "Processed index %d / %d (time %s) in %.2f s",
            i + 1,
            N,
            np.datetime_as_string(np.datetime64(unique_times[i])),
            t_iter_end - t_iter_start,
        )

    # end loop
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info("Total processing time: %.2f s", elapsed_time)

    # --- Create xarray Dataset and save to NetCDF ---
    dates = unique_times
    dims = ("time", "wavelength")
    coords = {"time": dates, "wavelength": wl}

    ds = xr.Dataset(
        {
            "Es": (
                dims,
                Es_mat,
                {"long_name": "Downwelling irradiance Es", "units": "mW m^-2 nm^-1"},
            ),
            "Ls_east": (
                dims,
                Ls_east,
                {"long_name": "Sky radiance East", "units": "mW m^-2 nm^-1 sr^-1"},
            ),
            "Lu_east": (
                dims,
                Lu_east,
                {"long_name": "Surface radiance East", "units": "mW m^-2 nm^-1 sr^-1"},
            ),
            "Rrs_east": (
                dims,
                Rrs_east,
                {
                    "long_name": "Observed remote sensing reflectance East",
                    "units": "sr^-1",
                },
            ),
            "Rrs_mod_east": (
                dims,
                Rrs_mod_east,
                {
                    "long_name": "Modeled remote sensing reflectance East",
                    "units": "sr^-1",
                },
            ),
            "Rg_east": (
                dims,
                Rg_east,
                {
                    "long_name": "Surface-reflected reflectance East (R_g)",
                    "units": "sr^-1",
                },
            ),
            "Ls_west": (
                dims,
                Ls_west,
                {"long_name": "Sky radiance West", "units": "mW m^-2 nm^-1 sr^-1"},
            ),
            "Lu_west": (
                dims,
                Lu_west,
                {"long_name": "Surface radiance West", "units": "mW m^-2 nm^-1 sr^-1"},
            ),
            "Rrs_west": (
                dims,
                Rrs_west,
                {
                    "long_name": "Observed remote sensing reflectance West",
                    "units": "sr^-1",
                },
            ),
            "Rrs_mod_west": (
                dims,
                Rrs_mod_west,
                {
                    "long_name": "Modeled remote sensing reflectance West",
                    "units": "sr^-1",
                },
            ),
            "Rg_west": (
                dims,
                Rg_west,
                {
                    "long_name": "Surface-reflected reflectance West (R_g)",
                    "units": "sr^-1",
                },
            ),
            # diagnostics
            "resid_rmse_east": (
                ("time",),
                resid_rmse_east,
                {"long_name": "RMSE in Lu/Ed residuals (east)"},
            ),
            "resid_rmse_west": (
                ("time",),
                resid_rmse_west,
                {"long_name": "RMSE in Lu/Ed residuals (west)"},
            ),
            "rho_s_east": (("time",), rho_s_east),
            "rho_s_west": (("time",), rho_s_west),
        },
        coords=coords,
        attrs={
            "title": f"3C Rrs time-series ({my_date})",
            "institution": "NIOZ/CNR",
            "source": "Jetty field measurements and 3C model adapted to NumPy, with O25",
            "history": f"Created {datetime.now().isoformat()}",
            "references": "Groetsch et al. (2017), Pitarch (2026, in prep.)",
        },
    )

    # Save to NetCDF file in the specified output folder
    nc_path = os.path.join(output_folder, f"Rrs_timeseries_{my_date}_O25.nc")
    ds.to_netcdf(nc_path)
    logging.info("Saved Dataset to %s", nc_path)

    # Optional: save plots (if requested)
    if args.plot:
        # Es at 550 nm
        try:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(unique_times, Es_mat[:, idx550], "k.-")
            ax.set_ylabel("E_s(550)")
            fig.autofmt_xdate()
            fig.tight_layout()
            fig_path = os.path.join(output_folder, f"Es_550_{my_date}.png")
            fig.savefig(fig_path, dpi=150)
            plt.close(fig)
            logging.info("Saved plot: %s", fig_path)
        except Exception:
            logging.exception("Failed to produce Es plot.")

        # Rrs at 550 nm east/west
        try:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(unique_times, Rrs_east[:, idx550], "b.-", label="East")
            ax.plot(unique_times, Rrs_west[:, idx550], "r.-", label="West")
            ax.set_ylabel("Rrs(550) (sr^-1)")
            fig.autofmt_xdate()
            ax.legend()
            fig.tight_layout()
            fig_path = os.path.join(output_folder, f"Rrs_550_timeseries_{my_date}.png")
            fig.savefig(fig_path, dpi=150)
            plt.close(fig)
            logging.info("Saved plot: %s", fig_path)
        except Exception:
            logging.exception("Failed to produce Rrs timeseries plot.")

    logging.info("Processing complete.")


if __name__ == "__main__":
    main()
