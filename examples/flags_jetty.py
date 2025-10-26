import numpy as np


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
