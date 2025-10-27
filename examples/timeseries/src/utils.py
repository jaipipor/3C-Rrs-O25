import math
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd


def load_jetty_data(filename, skiprows=0, encoding="utf-8", parse_datetime=True):
    """
    Load ragged "jetty" CSV where columns are:
      instrument, variable, datetime, l_1, l_end, delta_l, <spectral values...>

    Returns:
      instrument : np.ndarray[str]   shape (n_rows,)
      variable   : np.ndarray[str]
      times      : np.ndarray (pd.Timestamp where possible, else str) shape (n_rows,)
      l1         : np.ndarray[float]
      l_end      : np.ndarray[float]
      delta_l    : np.ndarray[float]
      spec       : np.ndarray[float]  shape (n_rows, max_len) padded with np.nan
    """
    inst_list = []
    var_list = []
    time_list = []
    l1_list = []
    lend_list = []
    dlt_list = []
    spec_rows = []

    def _to_float(x):
        try:
            return float(x)
        except Exception:
            return np.nan

    with open(filename, "r", encoding=encoding, errors="replace") as fh:
        for i, raw in enumerate(fh):
            if i < skiprows:
                continue
            line = raw.rstrip("\n\r")
            if not line:
                # skip empty lines
                continue

            # split into 7 parts: first 6 metadata fields + rest (spectral payload)
            parts = line.split(",", 6)
            if len(parts) < 7:
                warnings.warn(
                    f"Line {i+1}: expected >=7 fields, got {len(parts)}. Line skipped.",
                    UserWarning,
                )
                continue

            inst, var, dt_str, l1_s, lend_s, dlt_s, payload = parts
            inst_list.append(inst.strip())
            var_list.append(var.strip())

            # parse datetime if requested (best-effort)
            if parse_datetime:
                try:
                    time_list.append(pd.to_datetime(dt_str.strip()))
                except Exception:
                    time_list.append(dt_str.strip())
            else:
                time_list.append(dt_str.strip())

            # parse numeric metadata
            l1_list.append(_to_float(l1_s.strip()))
            lend_list.append(_to_float(lend_s.strip()))
            dlt_list.append(_to_float(dlt_s.strip()))

            # parse spectral payload (comma-separated floats). Allow empty entries -> NaN
            # split payload into individual items (there may be trailing commas)
            vals = [v.strip() for v in payload.split(",")]
            row = np.array(
                [_to_float(v) if v != "" else np.nan for v in vals], dtype=float
            )
            spec_rows.append(row)

    # convert metadata lists to numpy arrays
    instrument = np.array(inst_list, dtype=object)
    variable = np.array(var_list, dtype=object)
    times = np.array(
        time_list, dtype=object
    )  # contains pd.Timestamp where parse succeeded
    l1 = np.array(l1_list, dtype=float)
    l_end = np.array(lend_list, dtype=float)
    delta_l = np.array(dlt_list, dtype=float)

    # pad spec rows to a rectangular 2-D array with NaN
    if spec_rows:
        maxlen = max(len(r) for r in spec_rows)
        nrows = len(spec_rows)
        spec = np.full((nrows, maxlen), np.nan, dtype=float)
        for idx, row in enumerate(spec_rows):
            spec[idx, : len(row)] = row
    else:
        spec = np.empty((0, 0), dtype=float)

    return instrument, variable, times, l1, l_end, delta_l, spec


def flags_jetty(wl: np.ndarray, Es: np.ndarray):
    """
    Simple QC flags for jetty above‐water spectra.

    Parameters
    ----------
    wl : array_like
        Wavelengths in nm.
    Es : array_like
        Measured downwelling irradiance spectrum at those wavelengths.

    Returns
    -------
    dark : bool
        True if Es(480) < 20.
    red : bool
        True if Es(680) > Es(470)
    anom : bool or np.nan
        True if Es(370)/Es(940) > 3.5, np.nan if 940 not in `l`.
    """
    wl = np.asarray(wl)
    Es = np.asarray(Es)

    # find indices (exact match)
    def idx(w):
        # if you worry about float rounding, replace == with np.isclose
        w_idx = np.where(wl == w)[0]
        return w_idx[0] if w_idx.size else None

    i470 = idx(470)
    i480 = idx(480)
    i680 = idx(680)
    i370 = idx(370)
    i940 = idx(940)

    # dark‐flag
    if i480 is None:
        raise ValueError("480 nm not found in wavelength array")
    dark = Es[i480] < 20

    # red‐flag
    if i470 is None or i680 is None:
        raise ValueError("470 or 680 nm not found in wavelength array")
    red = Es[i680] > Es[i470]

    # anomaly‐flag
    if i370 is None or i940 is None:
        anom = np.nan
    else:
        anom = (Es[i370] / Es[i940]) > 3.5

    return dark, red, anom


def solar_az_el(utc, lat, lon, alt_km=0.0):
    """
    Compute solar azimuth and elevation angles for given UTC time(s) and site.

    Parameters
    ----------
    utc : str or datetime or array-like
        UTC time(s). If str: format 'YYYY/MM/DD HH:MM:SS'. If datetime, must be timezone-aware or assumed UTC.
        Can also be a 1D array of either strings or datetimes.
    lat : float or array-like
        Latitude in degrees (-90 to +90; South negative).
    lon : float or array-like
        Longitude in degrees (-180 to +180; West negative).
    alt_km : float or array-like, optional
        Site altitude above sea level in kilometers.

    Returns
    -------
    az : ndarray
        Solar azimuth angle in degrees (0–360, with 0 = north, increasing eastward).
    el : ndarray
        Solar elevation angle in degrees above the horizon.
    """
    # --- ensure arrays ---
    utc_arr = np.atleast_1d(utc)
    lat = np.atleast_1d(lat).astype(float)
    lon = np.atleast_1d(lon).astype(float)
    alt = np.atleast_1d(alt_km).astype(float)

    # parse UTC into Julian date
    def to_jd(dt_str_or_dt):
        if isinstance(dt_str_or_dt, str):
            dt = datetime.strptime(dt_str_or_dt, "%Y/%m/%d %H:%M:%S")
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt_str_or_dt
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        # Julian Day
        year, month, day = dt.year, dt.month, dt.day
        hour = dt.hour + dt.minute / 60 + dt.second / 3600
        if month <= 2:
            year -= 1
            month += 12
        A = math.floor(year / 100)
        B = 2 - A + math.floor(A / 4)
        jd0 = (
            math.floor(365.25 * (year + 4716))
            + math.floor(30.6001 * (month + 1))
            + day
            + B
            - 1524.5
        )
        return jd0 + hour / 24.0

    jd = np.array([to_jd(u) for u in utc_arr])
    d = jd - 2451543.5

    # Keplerian elements
    w = 282.9404 + 4.70935e-5 * d  # perihelion longitude
    e = 0.016709 - 1.151e-9 * d  # eccentricity
    M = np.mod(356.0470 + 0.9856002585 * d, 360)  # mean anomaly
    L = w + M  # mean longitude
    oblecl = 23.4393 - 3.563e-7 * d  # obliquity

    # auxiliary equation of center
    Mrad = np.deg2rad(M)
    E = M + (180 / np.pi) * e * np.sin(Mrad) * (1 + e * np.cos(Mrad))

    # ecliptic rectangular coordinates
    Erad = np.deg2rad(E)
    x = np.cos(Erad) - e
    y = np.sin(Erad) * np.sqrt(1 - e**2)
    r = np.hypot(x, y)
    v = np.rad2deg(np.arctan2(y, x))  # true anomaly
    lon_sun = v + w

    # position in ecliptic coords
    lon_rad = np.deg2rad(lon_sun)
    xe = r * np.cos(lon_rad)
    ye = r * np.sin(lon_rad)
    ze = 0.0

    # rotate to equatorial coords
    obl_rad = np.deg2rad(oblecl)
    xeq = xe
    yeq = ye * np.cos(obl_rad) + ze * np.sin(obl_rad)
    zeq = ye * np.sin(obl_rad) + ze * np.cos(obl_rad)

    # RA and declination
    # correct r for altitude (AU → km conversion: 1 AU ≈ 149 598 000 km)
    r_corr = r - alt / 149598000.0
    RA = np.rad2deg(np.arctan2(yeq, xeq))
    delta = np.rad2deg(np.arcsin(zeq / r_corr))

    # UTC in fractional hours
    def frac_hour(dt):
        return dt.hour + dt.minute / 60 + dt.second / 3600

    uth = np.array(
        [
            frac_hour(
                u
                if not isinstance(u, str)
                else datetime.strptime(u, "%Y/%m/%d %H:%M:%S")
            )
            for u in utc_arr
        ]
    )

    # Local sidereal time
    GMST0 = np.mod(L + 180, 360) / 15
    LST = GMST0 + uth + lon / 15

    # Hour angle
    HA = (LST * 15) - RA

    # convert to horizon coords
    HA_rad = np.deg2rad(HA)
    delta_rad = np.deg2rad(delta)
    lat_rad = np.deg2rad(lat)

    xh = np.cos(HA_rad) * np.cos(delta_rad)
    yh = np.sin(HA_rad) * np.cos(delta_rad)
    zh = np.sin(delta_rad)

    # rotate for latitude
    xhor = xh * np.cos(np.pi / 2 - lat_rad) - zh * np.sin(np.pi / 2 - lat_rad)
    yhor = yh
    zhor = xh * np.sin(np.pi / 2 - lat_rad) + zh * np.cos(np.pi / 2 - lat_rad)

    az = np.mod(np.rad2deg(np.arctan2(yhor, xhor)) + 180, 360)
    el = np.rad2deg(np.arcsin(zhor))

    return az, el
