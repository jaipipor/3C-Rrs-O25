"""Utility functions for the 3C-O25 time-series example."""

from __future__ import annotations

import math
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def load_jetty_data(
    filename: str | Path,
    skiprows: int = 0,
    encoding: str = "utf-8",
    parse_datetime: bool = True,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Load a ragged jetty CSV file.

    The expected columns are:

    ``instrument, variable, datetime, l1, l_end, delta_l, spectral values...``

    Parameters
    ----------
    filename : str or pathlib.Path
        Input CSV path.
    skiprows : int, optional
        Number of initial rows to skip.
    encoding : str, optional
        Text encoding used to read the file.
    parse_datetime : bool, optional
        Parse the timestamp column with pandas when true.

    Returns
    -------
    instrument : np.ndarray
        Instrument identifiers with shape ``(n_rows,)``.
    variable : np.ndarray
        Measurement-variable identifiers with shape ``(n_rows,)``.
    times : np.ndarray
        Parsed timestamps or timestamp strings with shape ``(n_rows,)``.
    l1 : np.ndarray
        Starting wavelength of each record, in nanometres.
    l_end : np.ndarray
        Ending wavelength of each record, in nanometres.
    delta_l : np.ndarray
        Wavelength increment of each record, in nanometres.
    spec : np.ndarray
        Spectral values with shape ``(n_rows, max_length)``. Shorter
        rows are padded with ``NaN``.

    Raises
    ------
    ValueError
        If ``skiprows`` is negative or no valid records are found.
    """
    filename = Path(filename)

    if skiprows < 0:
        raise ValueError("skiprows must be greater than or equal to zero.")

    instrument_list: list[str] = []
    variable_list: list[str] = []
    time_list: list[object] = []
    l1_list: list[float] = []
    l_end_list: list[float] = []
    delta_l_list: list[float] = []
    spectral_rows: list[np.ndarray] = []

    def to_float(value: str) -> float:
        """Convert text to float, returning NaN for invalid input."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return np.nan

    with filename.open(
        "r",
        encoding=encoding,
        errors="replace",
    ) as file_handle:
        for line_index, raw_line in enumerate(file_handle):
            if line_index < skiprows:
                continue

            line = raw_line.rstrip("\n\r")

            if not line:
                continue

            # Split into the first six metadata fields and one
            # spectral payload.
            parts = line.split(",", 6)

            if len(parts) < 7:
                warnings.warn(
                    f"Line {line_index + 1}: expected at least "
                    f"7 fields, got {len(parts)}. The line was "
                    "skipped.",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            (
                instrument_text,
                variable_text,
                datetime_text,
                l1_text,
                l_end_text,
                delta_l_text,
                payload,
            ) = parts

            if parse_datetime:
                try:
                    parsed_time: object = pd.to_datetime(
                        datetime_text.strip(),
                        format="%Y%m%d %H%M%S",
                        errors="raise",
                    )
                except (TypeError, ValueError):
                    warnings.warn(
                        f"Line {line_index + 1}: timestamp "
                        f"{datetime_text.strip()!r} could not be "
                        "parsed. The original text was retained.",
                        UserWarning,
                        stacklevel=2,
                    )
                    parsed_time = datetime_text.strip()
            else:
                parsed_time = datetime_text.strip()

            spectral_values = [value.strip() for value in payload.split(",")]

            spectral_row = np.array(
                [to_float(value) if value else np.nan for value in spectral_values],
                dtype=float,
            )

            instrument_list.append(instrument_text.strip())
            variable_list.append(variable_text.strip())
            time_list.append(parsed_time)
            l1_list.append(to_float(l1_text.strip()))
            l_end_list.append(to_float(l_end_text.strip()))
            delta_l_list.append(to_float(delta_l_text.strip()))
            spectral_rows.append(spectral_row)

    if not spectral_rows:
        raise ValueError(f"No valid spectral records were found in {filename}.")

    instrument = np.asarray(
        instrument_list,
        dtype=object,
    )
    variable = np.asarray(
        variable_list,
        dtype=object,
    )
    times = np.asarray(
        time_list,
        dtype=object,
    )
    l1 = np.asarray(
        l1_list,
        dtype=float,
    )
    l_end = np.asarray(
        l_end_list,
        dtype=float,
    )
    delta_l = np.asarray(
        delta_l_list,
        dtype=float,
    )

    maximum_length = max(row.size for row in spectral_rows)

    spec = np.full(
        (
            len(spectral_rows),
            maximum_length,
        ),
        np.nan,
        dtype=float,
    )

    for row_index, row in enumerate(spectral_rows):
        spec[
            row_index,
            : row.size,
        ] = row

    return (
        instrument,
        variable,
        times,
        l1,
        l_end,
        delta_l,
        spec,
    )


def interpolate_spectrum(
    row_index: int,
    l1: np.ndarray,
    l_end: np.ndarray,
    delta_l: np.ndarray,
    spec: np.ndarray,
    target_wavelengths: np.ndarray,
) -> np.ndarray:
    """Interpolate one spectral record onto a target grid.

    Every record is interpreted using its own starting wavelength,
    ending wavelength, and wavelength increment.

    Parameters
    ----------
    row_index : int
        Row containing the spectral record.
    l1 : array_like
        Starting wavelength of every record, in nanometres.
    l_end : array_like
        Ending wavelength of every record, in nanometres.
    delta_l : array_like
        Wavelength increment of every record, in nanometres.
    spec : array_like
        Two-dimensional spectral array. Rows represent records;
        shorter rows may be padded with ``NaN``.
    target_wavelengths : array_like
        Strictly increasing wavelength grid onto which the record
        is interpolated, in nanometres.

    Returns
    -------
    np.ndarray
        Spectrum interpolated onto ``target_wavelengths``.

    Raises
    ------
    IndexError
        If ``row_index`` is outside the available record range.
    ValueError
        If the wavelength metadata, spectral data, or target grid
        are invalid, or if interpolation would require
        extrapolation.
    """
    l1 = np.asarray(
        l1,
        dtype=float,
    )
    l_end = np.asarray(
        l_end,
        dtype=float,
    )
    delta_l = np.asarray(
        delta_l,
        dtype=float,
    )
    spec = np.asarray(
        spec,
        dtype=float,
    )
    target_wavelengths = np.asarray(
        target_wavelengths,
        dtype=float,
    )

    if spec.ndim != 2:
        raise ValueError(
            "Expected a two-dimensional spectral array, " f"got shape {spec.shape}."
        )

    number_of_records = spec.shape[0]

    if row_index < 0 or row_index >= number_of_records:
        raise IndexError(
            f"Spectral row index {row_index} is outside the "
            f"valid range 0 to {number_of_records - 1}."
        )

    metadata_arrays = (
        l1,
        l_end,
        delta_l,
    )

    if any(array.ndim != 1 for array in metadata_arrays):
        raise ValueError("Wavelength metadata must be " "one-dimensional arrays.")

    if any(array.size != number_of_records for array in metadata_arrays):
        raise ValueError(
            "Wavelength metadata must contain one entry " "per spectral record."
        )

    if target_wavelengths.ndim != 1 or target_wavelengths.size == 0:
        raise ValueError(
            "Target wavelengths must be a nonempty " "one-dimensional array."
        )

    if not np.all(np.isfinite(target_wavelengths)):
        raise ValueError("Target wavelengths contain nonfinite values.")

    if np.any(np.diff(target_wavelengths) <= 0):
        raise ValueError("Target wavelengths must be strictly increasing.")

    wavelength_start = float(l1[row_index])
    wavelength_end = float(l_end[row_index])
    wavelength_step = float(delta_l[row_index])

    metadata = np.array(
        [
            wavelength_start,
            wavelength_end,
            wavelength_step,
        ],
        dtype=float,
    )

    if not np.all(np.isfinite(metadata)):
        raise ValueError(
            f"Record {row_index} contains nonfinite " "wavelength metadata."
        )

    if wavelength_step <= 0:
        raise ValueError(
            f"Record {row_index} has an invalid wavelength "
            f"increment: {wavelength_step} nm."
        )

    if wavelength_end < wavelength_start:
        raise ValueError(
            f"Record {row_index} ends before it starts: "
            f"{wavelength_start} to {wavelength_end} nm."
        )

    sample_count_float = (wavelength_end - wavelength_start) / wavelength_step + 1.0

    sample_count = int(round(sample_count_float))

    if not np.isclose(
        sample_count_float,
        sample_count,
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError(
            f"Record {row_index} wavelength metadata do not "
            "define an integer number of samples: "
            f"start={wavelength_start}, "
            f"end={wavelength_end}, "
            f"step={wavelength_step}."
        )

    available_columns = spec.shape[-1]

    if sample_count > available_columns:
        raise ValueError(
            f"Record {row_index} requires {sample_count} spectral "
            f"values, but the spectral array has only "
            f"{available_columns} columns."
        )

    values = np.asarray(
        spec[
            row_index,
            :sample_count,
        ],
        dtype=float,
    )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"Record {row_index} contains missing or "
            "nonfinite spectral values within its declared "
            "wavelength range."
        )

    source_wavelengths = wavelength_start + wavelength_step * np.arange(
        sample_count,
        dtype=float,
    )

    if not np.isclose(
        source_wavelengths[-1],
        wavelength_end,
        rtol=0.0,
        atol=1e-8,
    ):
        raise ValueError(
            f"Record {row_index} wavelength axis ends at "
            f"{source_wavelengths[-1]} nm instead of "
            f"{wavelength_end} nm."
        )

    target_minimum = float(target_wavelengths[0])
    target_maximum = float(target_wavelengths[-1])
    source_minimum = float(source_wavelengths[0])
    source_maximum = float(source_wavelengths[-1])

    if target_minimum < source_minimum or target_maximum > source_maximum:
        raise ValueError(
            f"Target wavelength range {target_minimum} to "
            f"{target_maximum} nm extends beyond record "
            f"{row_index}, which covers {source_minimum} to "
            f"{source_maximum} nm."
        )

    return np.interp(
        target_wavelengths,
        source_wavelengths,
        values,
    )


def angular_difference_deg(
    angle_a: float,
    angle_b: float,
) -> float:
    """Return the smallest angular separation in degrees.

    Parameters
    ----------
    angle_a : float
        First angle in degrees.
    angle_b : float
        Second angle in degrees.

    Returns
    -------
    float
        Smallest angular separation in the closed interval
        ``[0, 180]``.

    Raises
    ------
    ValueError
        If either angle is nonfinite.
    """
    angle_a = float(angle_a)
    angle_b = float(angle_b)

    if not np.isfinite(angle_a) or not np.isfinite(angle_b):
        raise ValueError("Angles must be finite.")

    difference = abs(angle_a - angle_b) % 360.0

    return min(
        difference,
        360.0 - difference,
    )


def flags_jetty(
    wl: np.ndarray,
    es: np.ndarray,
) -> tuple[bool, bool, bool | float]:
    """Calculate quality-control flags for jetty spectra.

    Parameters
    ----------
    wl : array_like
        Wavelengths in nanometres.
    es : array_like
        Measured downwelling irradiance spectrum.

    Returns
    -------
    dark : bool
        True when ``Es(480) < 20``.
    red : bool
        True when ``Es(680) > Es(470)``.
    anomaly : bool or float
        True when ``Es(370) / Es(940) > 3.5``. Returns
        ``NaN`` when 370 or 940 nm is unavailable, or when
        ``Es(940)`` is zero.

    Raises
    ------
    ValueError
        If wavelength and irradiance are incompatible, or if
        the wavelengths required for the dark and red flags
        are absent.
    """
    wl = np.asarray(
        wl,
        dtype=float,
    )
    es = np.asarray(
        es,
        dtype=float,
    )

    if wl.ndim != 1 or es.ndim != 1:
        raise ValueError("Wavelength and irradiance must be " "one-dimensional arrays.")

    if wl.size != es.size:
        raise ValueError(
            "Wavelength and irradiance arrays must have " "the same length."
        )

    if wl.size == 0:
        raise ValueError("Wavelength and irradiance arrays must not " "be empty.")

    def wavelength_index(
        wavelength: float,
    ) -> int | None:
        matches = np.where(
            np.isclose(
                wl,
                wavelength,
                rtol=0.0,
                atol=1e-8,
            )
        )[0]

        return int(matches[0]) if matches.size else None

    index_470 = wavelength_index(470.0)
    index_480 = wavelength_index(480.0)
    index_680 = wavelength_index(680.0)
    index_370 = wavelength_index(370.0)
    index_940 = wavelength_index(940.0)

    if index_480 is None:
        raise ValueError("480 nm was not found in the wavelength array.")

    if index_470 is None or index_680 is None:
        raise ValueError("470 or 680 nm was not found in the " "wavelength array.")

    dark = bool(es[index_480] < 20.0)
    red = bool(es[index_680] > es[index_470])

    if index_370 is None or index_940 is None or es[index_940] == 0:
        anomaly: bool | float = np.nan
    else:
        anomaly = bool((es[index_370] / es[index_940]) > 3.5)

    return dark, red, anomaly


def solar_az_el(
    utc,
    lat,
    lon,
    alt_km=0.0,
):
    """Compute solar azimuth and elevation angles.

    Parameters
    ----------
    utc : str, datetime, or array_like
        UTC time or times. Strings must use
        ``YYYY/MM/DD HH:MM:SS``. Naive datetime objects are
        interpreted as UTC.
    lat : float or array_like
        Latitude in degrees, with southern latitudes negative.
    lon : float or array_like
        Longitude in degrees, with western longitudes negative.
    alt_km : float or array_like, optional
        Site altitude above sea level in kilometres.

    Returns
    -------
    azimuth : np.ndarray
        Solar azimuth in degrees from north, increasing
        eastward.
    elevation : np.ndarray
        Solar elevation in degrees above the horizon.

    Raises
    ------
    TypeError
        If a UTC value has an unsupported type.
    ValueError
        If location values are nonfinite or outside the
        supported geographic ranges.
    """
    utc_array = np.atleast_1d(utc)
    latitude = np.atleast_1d(lat).astype(float)
    longitude = np.atleast_1d(lon).astype(float)
    altitude = np.atleast_1d(alt_km).astype(float)

    if not np.all(np.isfinite(latitude)):
        raise ValueError("Latitude must be finite.")

    if not np.all(np.isfinite(longitude)):
        raise ValueError("Longitude must be finite.")

    if not np.all(np.isfinite(altitude)):
        raise ValueError("Altitude must be finite.")

    if np.any((latitude < -90.0) | (latitude > 90.0)):
        raise ValueError("Latitude must be between -90 and 90 degrees.")

    if np.any((longitude < -180.0) | (longitude > 180.0)):
        raise ValueError("Longitude must be between -180 and 180 degrees.")

    def parse_utc(value) -> datetime:
        """Convert one supported UTC value to a datetime."""
        if isinstance(value, str):
            parsed = datetime.strptime(
                value,
                "%Y/%m/%d %H:%M:%S",
            )

            return parsed.replace(tzinfo=timezone.utc)

        if isinstance(
            value,
            pd.Timestamp,
        ):
            value = value.to_pydatetime()

        if isinstance(
            value,
            np.datetime64,
        ):
            value = pd.Timestamp(value).to_pydatetime()

        if not isinstance(
            value,
            datetime,
        ):
            raise TypeError(
                "UTC values must be strings, datetime "
                "objects, pandas timestamps, or NumPy "
                "datetime64 values."
            )

        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)

        return value.astimezone(timezone.utc)

    utc_datetimes = [parse_utc(value) for value in utc_array]

    def to_julian_day(
        value: datetime,
    ) -> float:
        """Convert a UTC datetime to a Julian Day value."""
        year = value.year
        month = value.month
        day = value.day

        hour = (
            value.hour
            + value.minute / 60.0
            + value.second / 3600.0
            + value.microsecond / 3_600_000_000.0
        )

        if month <= 2:
            year -= 1
            month += 12

        century = math.floor(year / 100)

        correction = 2 - century + math.floor(century / 4)

        julian_day_zero = (
            math.floor(365.25 * (year + 4716))
            + math.floor(30.6001 * (month + 1))
            + day
            + correction
            - 1524.5
        )

        return julian_day_zero + hour / 24.0

    julian_day = np.array(
        [to_julian_day(value) for value in utc_datetimes],
        dtype=float,
    )

    days_since_epoch = julian_day - 2451543.5

    perihelion_longitude = 282.9404 + 4.70935e-5 * days_since_epoch

    eccentricity = 0.016709 - 1.151e-9 * days_since_epoch

    mean_anomaly = np.mod(
        356.0470 + 0.9856002585 * days_since_epoch,
        360.0,
    )

    mean_longitude = perihelion_longitude + mean_anomaly

    obliquity = 23.4393 - 3.563e-7 * days_since_epoch

    mean_anomaly_radians = np.deg2rad(mean_anomaly)

    eccentric_anomaly = mean_anomaly + (180.0 / np.pi) * eccentricity * np.sin(
        mean_anomaly_radians
    ) * (1.0 + eccentricity * np.cos(mean_anomaly_radians))

    eccentric_anomaly_radians = np.deg2rad(eccentric_anomaly)

    x_ecliptic_orbit = np.cos(eccentric_anomaly_radians) - eccentricity

    y_ecliptic_orbit = np.sin(eccentric_anomaly_radians) * np.sqrt(
        1.0 - eccentricity**2
    )

    distance_au = np.hypot(
        x_ecliptic_orbit,
        y_ecliptic_orbit,
    )

    true_anomaly = np.rad2deg(
        np.arctan2(
            y_ecliptic_orbit,
            x_ecliptic_orbit,
        )
    )

    solar_longitude = true_anomaly + perihelion_longitude

    solar_longitude_radians = np.deg2rad(solar_longitude)

    x_ecliptic = distance_au * np.cos(solar_longitude_radians)

    y_ecliptic = distance_au * np.sin(solar_longitude_radians)

    obliquity_radians = np.deg2rad(obliquity)

    x_equatorial = x_ecliptic

    y_equatorial = y_ecliptic * np.cos(obliquity_radians)

    z_equatorial = y_ecliptic * np.sin(obliquity_radians)

    corrected_distance_au = distance_au - altitude / 149_598_000.0

    right_ascension = np.rad2deg(
        np.arctan2(
            y_equatorial,
            x_equatorial,
        )
    )

    declination = np.rad2deg(np.arcsin(z_equatorial / corrected_distance_au))

    utc_hours = np.array(
        [
            (
                value.hour
                + value.minute / 60.0
                + value.second / 3600.0
                + value.microsecond / 3_600_000_000.0
            )
            for value in utc_datetimes
        ],
        dtype=float,
    )

    greenwich_sidereal_time = (
        np.mod(
            mean_longitude + 180.0,
            360.0,
        )
        / 15.0
    )

    local_sidereal_time = greenwich_sidereal_time + utc_hours + longitude / 15.0

    hour_angle = local_sidereal_time * 15.0 - right_ascension

    hour_angle_radians = np.deg2rad(hour_angle)

    declination_radians = np.deg2rad(declination)

    latitude_radians = np.deg2rad(latitude)

    x_horizon_initial = np.cos(hour_angle_radians) * np.cos(declination_radians)

    y_horizon = np.sin(hour_angle_radians) * np.cos(declination_radians)

    z_horizon_initial = np.sin(declination_radians)

    x_horizon = x_horizon_initial * np.cos(
        np.pi / 2.0 - latitude_radians
    ) - z_horizon_initial * np.sin(np.pi / 2.0 - latitude_radians)

    z_horizon = x_horizon_initial * np.sin(
        np.pi / 2.0 - latitude_radians
    ) + z_horizon_initial * np.cos(np.pi / 2.0 - latitude_radians)

    azimuth = np.mod(
        np.rad2deg(
            np.arctan2(
                y_horizon,
                x_horizon,
            )
        )
        + 180.0,
        360.0,
    )

    elevation = np.rad2deg(np.arcsin(z_horizon))

    return azimuth, elevation
