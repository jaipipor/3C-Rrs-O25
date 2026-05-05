"""
Command-line runner for the 3C-O25 Rrs model.

Example
-------
PowerShell, from the repository root:

    & .\.venv\Scripts\python.exe .\scripts\run_3c.py `
        --input .\examples\example_single_spectrum.csv `
        --theta-s 59 `
        --theta-v 35 `
        --phi 100 `
        --am 4 `
        --rh 60 `
        --pressure 1013.25
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lmfit as lm
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Make sure the local package in ./src is importable when running
# this script directly from the repository.
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rrs3c.model import rrs_model_3C_O25  # noqa: E402


def build_default_params() -> lm.Parameters:
    """Return the default parameter set used by the example script."""
    params = lm.Parameters()
    params.add_many(
        ("C", 5.0, True, 0.1, 50.0, None),
        ("N", 1.0, True, 0.01, 100.0, None),
        ("Y", 0.5, True, 0.01, 5.0, None),
        ("SNAP", 0.015, True, 0.005, 0.03, None),
        ("Sg", 0.015, True, 0.005, 0.03, None),
        ("rho", 0.02, False, 0.0, 0.03, None),
        ("rho_d", 0.0, True, 0.0, 10.0, None),
        ("rho_s", 0.0, True, -0.01, 0.01, None),
        ("alpha", 0.2, True, 0.0, 2.0, None),
        ("beta", 0.05, True, 0.0, 1.0, None),
    )
    return params


def update_params_from_dict(params: lm.Parameters, param_dict: dict) -> lm.Parameters:
    """Update an existing lmfit.Parameters object from a plain dictionary."""
    for name, spec in param_dict.items():
        if name not in params:
            raise ValueError(f"Unknown parameter: {name}")

        if not isinstance(spec, dict):
            raise ValueError(f"Parameter '{name}' must map to a dictionary")

        par = params[name]

        if "value" in spec:
            par.set(value=spec["value"])
        if "vary" in spec:
            par.set(vary=spec["vary"])
        if "min" in spec:
            par.set(min=spec["min"])
        if "max" in spec:
            par.set(max=spec["max"])
        if "expr" in spec:
            par.set(expr=spec["expr"])

    return params


def load_params(args: argparse.Namespace) -> lm.Parameters:
    """Build default parameters and optionally override them from JSON."""
    params = build_default_params()

    if args.params_file is not None and args.params_json is not None:
        raise ValueError("Use only one of --params-file or --params-json")

    if args.params_file is not None:
        if not args.params_file.is_file():
            raise FileNotFoundError(f"Parameter file not found: {args.params_file}")

        with open(args.params_file, "r", encoding="utf-8") as f:
            param_dict = json.load(f)

        return update_params_from_dict(params, param_dict)

    if args.params_json is not None:
        param_dict = json.loads(args.params_json)
        return update_params_from_dict(params, param_dict)

    return params


def build_default_weights(wl: np.ndarray) -> np.ndarray:
    """Return the same default weighting scheme used in model.py."""
    return np.where(
        (wl >= 760) & (wl <= 765),
        0.0,  # remove the H2O band feature
        np.where(wl < 400, 1.0, np.where(wl > 800, 5.0, 1.0)),
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the 3C-O25 Rrs model on a single input spectrum CSV."
    )

    # Input / file layout
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--data-folder",
        default=None,
        type=Path,
        help="Path to the auxiliary data folder. Defaults to the repository data folder.",
    )
    parser.add_argument(
        "--skiprows",
        default=15,
        type=int,
        help="Number of rows to skip when reading the CSV (default: 15).",
    )
    parser.add_argument(
        "--index-col",
        default=0,
        type=int,
        help="Column index to use as wavelength index in the CSV (default: 0).",
    )
    parser.add_argument(
        "--li-col",
        default=0,
        type=int,
        help="Zero-based column index (after index_col is removed) for Li (default: 0).",
    )
    parser.add_argument(
        "--lt-col",
        default=1,
        type=int,
        help="Zero-based column index (after index_col is removed) for Lt (default: 1).",
    )
    parser.add_argument(
        "--es-col",
        default=2,
        type=int,
        help="Zero-based column index (after index_col is removed) for Es (default: 2).",
    )

    # Geometry
    parser.add_argument(
        "--theta-s", required=True, type=float, help="Solar zenith angle (deg)."
    )
    parser.add_argument(
        "--theta-v", required=True, type=float, help="Sensor zenith angle (deg)."
    )
    parser.add_argument(
        "--phi", required=True, type=float, help="Relative azimuth angle (deg)."
    )

    # Ancillary
    parser.add_argument("--am", required=True, type=float, help="Air mass.")
    parser.add_argument(
        "--rh", required=True, type=float, help="Relative humidity (%)."
    )
    parser.add_argument("--pressure", required=True, type=float, help="Pressure (hPa).")

    # Optional behavior
    parser.add_argument(
        "--method",
        default="leastsq",
        type=str,
        help="Optimization method passed to lmfit.minimize (default: leastsq).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress fit summary printing.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not show the output plot.",
    )
    parser.add_argument(
        "--save-csv",
        default=None,
        type=Path,
        help="Optional path to save output spectra as CSV.",
    )

    parser.add_argument(
        "--params-file",
        default=None,
        type=Path,
        help="Path to a JSON file defining lmfit parameters.",
    )

    parser.add_argument(
        "--params-json",
        default=None,
        type=str,
        help="Inline JSON string defining lmfit parameters.",
    )

    return parser.parse_args()


def load_input_spectrum(
    input_path: Path,
    skiprows: int,
    index_col: int,
    li_col: int,
    lt_col: int,
    es_col: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load wavelength, Li, Lt, Es from the input CSV."""
    data = pd.read_csv(
        input_path,
        index_col=index_col,
        skiprows=skiprows,
    )

    wl = data.index.to_numpy(dtype=float)
    Li = data.iloc[:, li_col].to_numpy(dtype=float)
    Lt = data.iloc[:, lt_col].to_numpy(dtype=float)
    Es = data.iloc[:, es_col].to_numpy(dtype=float)

    return wl, Li, Lt, Es


def save_output_csv(
    output_path: Path,
    wl: np.ndarray,
    LtEs_measured: np.ndarray,
    Rrs_mod: np.ndarray,
    Rg: np.ndarray,
) -> None:
    """Save measured and modeled spectra to CSV."""
    df = pd.DataFrame(
        {
            "wl_nm": wl,
            "LtEs_measured": LtEs_measured,
            "LtEs_modeled": Rrs_mod + Rg,
            "Rrs_modeled": Rrs_mod,
            "Rg_modeled": Rg,
            "Rrs_output": LtEs_measured - Rg,
        }
    )
    df.to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    # Resolve default data folder if not provided
    data_folder = (
        args.data_folder if args.data_folder is not None else (REPO_ROOT / "data")
    )

    # Load input spectrum
    wl, Li, Lt, Es = load_input_spectrum(
        input_path=args.input,
        skiprows=args.skiprows,
        index_col=args.index_col,
        li_col=args.li_col,
        lt_col=args.lt_col,
        es_col=args.es_col,
    )

    # Prepare model inputs
    LiEs = Li / Es
    LtEs = Lt / Es
    geom = (args.theta_s, args.theta_v, args.phi)
    anc = (args.am, args.rh, args.pressure)

    params = load_params(args)
    weights = build_default_weights(wl)

    model = rrs_model_3C_O25(data_folder=data_folder)

    out, Rrs_mod, Rg = model.fit_LtEs(
        wl=wl,
        LiEs=LiEs,
        LtEs=LtEs,
        params=params,
        weights=weights,
        geom=geom,
        anc=anc,
        method=args.method,
        verbose=not args.quiet,
    )

    # Always print a very compact final line even in quiet mode
    print("\nFit finished.")
    if hasattr(out, "success"):
        print(f"success: {out.success}")
    if hasattr(out, "message"):
        print(f"message: {out.message}")

    # Optional CSV output
    if args.save_csv is not None:
        save_output_csv(args.save_csv, wl, LtEs, Rrs_mod, Rg)
        print(f"Saved output CSV to: {args.save_csv}")

    # Optional plotting
    if not args.no_plot:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.grid(True)
        plt.plot(wl, LtEs, label="L_t/E_s, measured")
        plt.plot(wl, Rrs_mod + Rg, label="L_t/E_s, modeled")
        plt.plot(wl, Rrs_mod, label="R_rs, modeled")
        plt.plot(wl, Rg, label="R_g, modeled")
        plt.plot(wl, LtEs - Rg, label="R_rs, output")
        plt.xlabel("wavelength (nm)")
        plt.ylabel("reflectance (sr$^{-1}$)")
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
