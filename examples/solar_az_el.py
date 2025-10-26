import math
from datetime import datetime, timezone

import numpy as np


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
