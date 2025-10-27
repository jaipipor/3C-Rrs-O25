import warnings

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
