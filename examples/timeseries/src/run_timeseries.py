"""Process the bundled radiometry time series with the 3C-O25 model."""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import lmfit as lm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from utils import (
    angular_difference_deg,
    flags_jetty,
    interpolate_spectrum,
    load_jetty_data,
    solar_az_el,
)

from rrs3c import rrs_model_3C_O25

LOGGER = logging.getLogger(__name__)
SIDES = {
    "east": 135.0,
    "west": 225.0,
}
SITE_LATITUDE = 53.001788
SITE_LONGITUDE = 4.789151
VIEW_ZENITH = 35.0
ANCILLARY_ATMOSPHERE = (4.0, 60.0, 1013.25)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    script_dir = Path(__file__).resolve().parent
    example_dir = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Process a radiometric time series with the 3C-O25 model."
    )
    parser.add_argument(
        "--input-file",
        default="example_time_series_data.csv",
        help="Input CSV filename.",
    )
    parser.add_argument(
        "--input-folder",
        type=Path,
        default=example_dir / "data",
        help="Directory containing the input CSV.",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        default=example_dir / "output",
        help="Directory for the NetCDF and optional plot.",
    )
    parser.add_argument(
        "--date",
        help="Optional UTC date to process, formatted as YYYYMMDD.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save a diagnostic PNG alongside the NetCDF file.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed progress messages.",
    )
    return parser.parse_args()


def build_parameters() -> lm.Parameters:
    """Return the parameter configuration used by the example."""
    params = lm.Parameters()
    params.add_many(
        ("C", 5.0, True, 0.1, 50.0, None),
        ("N", 1.0, True, 0.01, 100.0, None),
        ("Y", 0.5, True, 0.01, 5.0, None),
        ("SNAP", 0.015, True, 0.005, 0.03, None),
        ("Sg", 0.015, True, 0.005, 0.03, None),
        ("rho", 0.02, True, 0.0, 0.03, None),
        ("rho_d", 0.0, True, 0.0, 10.0, None),
        ("rho_s", 0.01, True, -0.1, 0.1, None),
        ("alpha", 0.2, True, 0.0, 2.0, None),
        ("beta", 0.05, True, 0.0, 1.0, None),
    )
    return params


def build_weights(wavelengths: np.ndarray) -> np.ndarray:
    """Return the wavelength weights used by the example."""
    weights = np.ones(wavelengths.size, dtype=float)
    weights[(wavelengths >= 750.0) & (wavelengths <= 775.0)] = 0.0
    weights[wavelengths > 800.0] = 5.0
    return weights


def select_single_record(
    variable: np.ndarray,
    group_indices: np.ndarray,
    tag: str,
) -> int | None:
    """Return one record index for a variable tag, or None when absent."""
    matches = group_indices[variable[group_indices] == tag]
    if matches.size == 0:
        return None
    if matches.size > 1:
        LOGGER.warning("Using the first of %d records tagged %s.", matches.size, tag)
    return int(matches[0])


def allocate_output(number_of_times: int, number_of_wavelengths: int) -> dict:
    """Allocate output arrays."""
    shape = (number_of_times, number_of_wavelengths)
    output: dict[str, np.ndarray] = {
        "Es": np.full(shape, np.nan),
        "solar_azimuth": np.full(number_of_times, np.nan),
        "solar_elevation": np.full(number_of_times, np.nan),
        "dark_flag": np.zeros(number_of_times, dtype=bool),
        "red_flag": np.zeros(number_of_times, dtype=bool),
        "anomaly_flag": np.full(number_of_times, np.nan),
    }

    for side in SIDES:
        for name in ("Li", "Lt", "Rrs", "Rrs_model", "Rg"):
            output[f"{name}_{side}"] = np.full(shape, np.nan)
        for name in ("fit_success",):
            output[f"{name}_{side}"] = np.zeros(number_of_times, dtype=bool)
        for name in ("rmse", "residual_550", "Rg_550", "rho", "rho_d", "rho_s"):
            output[f"{name}_{side}"] = np.full(number_of_times, np.nan)

    return output


def make_dataset(
    times: pd.DatetimeIndex,
    wavelengths: np.ndarray,
    output: dict[str, np.ndarray],
    source_file: Path,
    elapsed_seconds: float,
) -> xr.Dataset:
    """Create the output dataset and attach metadata."""
    spectral_metadata = {
        "Es": ("Downwelling irradiance", "mW m-2 nm-1"),
        "Li": ("Sky radiance", "mW m-2 nm-1 sr-1"),
        "Lt": ("Surface radiance", "mW m-2 nm-1 sr-1"),
        "Rrs": ("Retrieved remote-sensing reflectance", "sr-1"),
        "Rrs_model": ("Modeled remote-sensing reflectance", "sr-1"),
        "Rg": ("Modeled surface-reflection term", "sr-1"),
    }

    data_vars: dict[str, tuple] = {
        "Es": (
            ("time", "wavelength"),
            output["Es"],
            {
                "long_name": spectral_metadata["Es"][0],
                "units": spectral_metadata["Es"][1],
            },
        ),
        "solar_azimuth": (
            "time",
            output["solar_azimuth"],
            {"long_name": "Solar azimuth", "units": "degree"},
        ),
        "solar_elevation": (
            "time",
            output["solar_elevation"],
            {"long_name": "Solar elevation", "units": "degree"},
        ),
        "dark_flag": ("time", output["dark_flag"]),
        "red_flag": ("time", output["red_flag"]),
        "anomaly_flag": ("time", output["anomaly_flag"]),
    }

    for side in SIDES:
        for name in ("Li", "Lt", "Rrs", "Rrs_model", "Rg"):
            long_name, units = spectral_metadata[name]
            data_vars[f"{name}_{side}"] = (
                ("time", "wavelength"),
                output[f"{name}_{side}"],
                {"long_name": f"{long_name}, {side}", "units": units},
            )
        data_vars[f"fit_success_{side}"] = (
            "time",
            output[f"fit_success_{side}"],
        )
        for name, long_name, units in (
            ("rmse", "Lt/Es residual RMSE", "sr-1"),
            ("residual_550", "Lt/Es residual at 550 nm", "sr-1"),
            ("Rg_550", "Surface-reflection term at 550 nm", "sr-1"),
            ("rho", "Fitted rho", "1"),
            ("rho_d", "Fitted rho_d", "1"),
            ("rho_s", "Fitted rho_s", "1"),
        ):
            data_vars[f"{name}_{side}"] = (
                "time",
                output[f"{name}_{side}"],
                {"long_name": f"{long_name}, {side}", "units": units},
            )

    return xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": times,
            "wavelength": (
                "wavelength",
                wavelengths,
                {"long_name": "Wavelength", "units": "nm"},
            ),
        },
        attrs={
            "title": "3C-O25 time-series example",
            "source_file": str(source_file.resolve()),
            "site_latitude": SITE_LATITUDE,
            "site_longitude": SITE_LONGITUDE,
            "processing_seconds": elapsed_seconds,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def save_plot(dataset: xr.Dataset, output_path: Path) -> None:
    """Save a compact diagnostic plot at 550 nm."""
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    dataset["Es"].sel(wavelength=550.0).plot(ax=axes[0], color="black")
    axes[0].set_title("Downwelling irradiance at 550 nm")
    axes[0].grid(True)

    for side, color in (("east", "tab:blue"), ("west", "tab:red")):
        dataset[f"Rrs_{side}"].sel(wavelength=550.0).plot(
            ax=axes[1], label=side.capitalize(), color=color
        )
    axes[1].set_title("Retrieved Rrs at 550 nm")
    axes[1].grid(True)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> int:
    """Run the time-series workflow."""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    LOGGER.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    input_path = args.input_folder / args.input_file
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    args.output_folder.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Loading %s", input_path)

    _, variable, raw_times, l1, l_end, delta_l, spectra = load_jetty_data(input_path)
    times = pd.DatetimeIndex(pd.to_datetime(raw_times))

    if args.date:
        try:
            selected_date = datetime.strptime(args.date, "%Y%m%d").date()
        except ValueError as error:
            raise ValueError("--date must use the YYYYMMDD format.") from error
        keep = np.asarray(times.date == selected_date)
        if not np.any(keep):
            raise ValueError(f"No records were found for {args.date}.")
        variable = variable[keep]
        times = times[keep]
        l1 = l1[keep]
        l_end = l_end[keep]
        delta_l = delta_l[keep]
        spectra = spectra[keep]

    unique_times = pd.DatetimeIndex(sorted(times.unique()))
    wavelengths = np.arange(350.0, 921.0, dtype=float)
    output = allocate_output(unique_times.size, wavelengths.size)
    weights = build_weights(wavelengths)
    index_550 = int(np.where(wavelengths == 550.0)[0][0])

    repository_root = Path(__file__).resolve().parents[3]
    model = rrs_model_3C_O25(data_folder=repository_root / "data")
    started = time.perf_counter()
    successful_fits = 0

    for time_index, acquisition_time in enumerate(unique_times):
        group_indices = np.flatnonzero(times == acquisition_time)
        es_index = select_single_record(variable, group_indices, "ES")
        if es_index is None:
            LOGGER.warning("%s: no ES record; skipping.", acquisition_time)
            continue

        try:
            es = interpolate_spectrum(
                es_index, l1, l_end, delta_l, spectra, wavelengths
            )
        except ValueError as error:
            LOGGER.warning("%s: invalid ES record: %s", acquisition_time, error)
            continue

        output["Es"][time_index] = es
        dark, red, anomaly = flags_jetty(wavelengths, es)
        output["dark_flag"][time_index] = dark
        output["red_flag"][time_index] = red
        output["anomaly_flag"][time_index] = anomaly

        azimuth_array, elevation_array = solar_az_el(
            acquisition_time,
            lat=SITE_LATITUDE,
            lon=SITE_LONGITUDE,
        )
        azimuth = float(np.asarray(azimuth_array).item())
        elevation = float(np.asarray(elevation_array).item())
        output["solar_azimuth"][time_index] = azimuth
        output["solar_elevation"][time_index] = elevation

        if dark or red or elevation < 5.0:
            LOGGER.info(
                "%s: rejected by quality control (dark=%s, red=%s, elevation=%.1f).",
                acquisition_time,
                dark,
                red,
                elevation,
            )
            continue

        for side, viewing_azimuth in SIDES.items():
            sky_index = select_single_record(
                variable, group_indices, f"LSKY-{side.upper()}"
            )
            surface_index = select_single_record(
                variable, group_indices, f"LSFC-{side.upper()}"
            )
            if sky_index is None or surface_index is None:
                LOGGER.warning("%s: incomplete %s observation.", acquisition_time, side)
                continue

            try:
                li = interpolate_spectrum(
                    sky_index, l1, l_end, delta_l, spectra, wavelengths
                )
                lt = interpolate_spectrum(
                    surface_index, l1, l_end, delta_l, spectra, wavelengths
                )

                if np.any(es <= 0.0):
                    raise ValueError("Es must be positive across the fitted spectrum.")

                geometry = (
                    90.0 - elevation,
                    VIEW_ZENITH,
                    angular_difference_deg(azimuth, viewing_azimuth),
                )
                result, rrs_model, rg = model.fit_LtEs(
                    wl=wavelengths,
                    LiEs=li / es,
                    LtEs=lt / es,
                    params=build_parameters(),
                    weights=weights,
                    geom=geometry,
                    anc=ANCILLARY_ATMOSPHERE,
                    verbose=False,
                )
            except (KeyError, ValueError, FloatingPointError) as error:
                LOGGER.warning("%s %s fit failed: %s", acquisition_time, side, error)
                continue

            observed_ltes = lt / es
            modeled_ltes = rrs_model + rg
            residual = modeled_ltes - observed_ltes

            output[f"Li_{side}"][time_index] = li
            output[f"Lt_{side}"][time_index] = lt
            output[f"Rrs_{side}"][time_index] = observed_ltes - rg
            output[f"Rrs_model_{side}"][time_index] = rrs_model
            output[f"Rg_{side}"][time_index] = rg
            output[f"fit_success_{side}"][time_index] = bool(result.success)
            output[f"rmse_{side}"][time_index] = float(np.sqrt(np.mean(residual**2)))
            output[f"residual_550_{side}"][time_index] = residual[index_550]
            output[f"Rg_550_{side}"][time_index] = rg[index_550]
            for parameter_name in ("rho", "rho_d", "rho_s"):
                output[f"{parameter_name}_{side}"][time_index] = result.params[
                    parameter_name
                ].value
            successful_fits += int(bool(result.success))

        LOGGER.info("Processed %s", acquisition_time)

    elapsed_seconds = time.perf_counter() - started
    dataset = make_dataset(
        unique_times,
        wavelengths,
        output,
        input_path,
        elapsed_seconds,
    )

    date_label = args.date or unique_times[0].strftime("%Y%m%d")
    output_path = args.output_folder / f"Rrs_timeseries_{date_label}.nc"
    dataset.to_netcdf(output_path)

    if args.plot:
        plot_path = args.output_folder / f"Rrs_timeseries_{date_label}.png"
        save_plot(dataset, plot_path)
        print(f"Saved plot: {plot_path.resolve()}")

    print(f"Saved NetCDF: {output_path.resolve()}")
    print(f"Successful fits: {successful_fits}")
    print(f"Elapsed time: {elapsed_seconds:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
