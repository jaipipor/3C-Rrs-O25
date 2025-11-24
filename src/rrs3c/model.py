"""
model.py

Optimized analytical 3C remote-sensing reflectance (Rrs) model.

This module provides the class `rrs_model_3C` which implements an
O25-parameterisation of the 3C Rrs model and exposes a fast
`fit_LtEs` method intended for repeated fitting calls.

Key features:
- Loads required auxiliary tables (aph templates, water IOPs, G-tables)
  from a configurable `data_folder`.
- Uses small LRU caches to avoid repeated expensive interpolations
  when the same wavelength grids are used repeatedly.
- Uses SciPy's RegularGridInterpolator for G-function evaluations.

Public API
----------
rrs_model_3C(data_folder)
    Create a model object. Files expected in `data_folder`:
    - vars_aph_v2.npz (contains `l_int`, `aph_norm_55`, `aph670_bounds`)
    - abs_scat_seawater_20d_35PSU_20230922_short.txt (aw, bbw table)
    - G0w.txt, G1w.txt, G0p.txt, G1p.txt (G-function tables)

fit_LtEs(wl, LiEs, LtEs, params, weights, geom, anc, method='leastsq', verbose=True)
    Fit Lt/Es spectra returning a tuple: (result_object, Rrs_model, Rg).

Notes
-----
The class intentionally keeps a small surface API and several helper
methods/attributes are implementation details (prefixed with an underscore)
and may change in future versions.

License: GPL-3.0 (inherit from project)
Author: Jaime Pitarch (refactor for repository)
"""

import hashlib
import os
import warnings
from collections import OrderedDict
from pathlib import Path

import lmfit as lm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# suppress a noisy warning from the `uncertainties` package triggered indirectly
# by lmfit when parameters have zero std_dev. This warning is harmless here but
# pollutes output; if you prefer to keep it visible, remove the next line.
warnings.filterwarnings(
    "ignore",
    message="Using UFloat objects with std_dev==0 may give unexpected results.",
    category=UserWarning,
    module="uncertainties.core",
)


# ---------------------------------------------------------------------------
# Core model class
# ---------------------------------------------------------------------------
class rrs_model_3C(object):
    """Class implementing the optimized 3C Rrs forward model and fitting.

    Notes:
    - The constructor loads auxiliary files (aph templates, water IOPs, G-tables)
      from `data_folder` and prepares small LRU caches used by the forward
      model to avoid repeated interpolation work when the same wavelength
      grids are evaluated repeatedly.
    - The public methods are `model_3C()` (returns the forward callable) and
      `fit_LtEs(...)` which performs a fit of measured Lt/Es using lmfit.
    """

    def __init__(self, data_folder=None):

        if data_folder is None or not os.path.isdir(data_folder):
            repo_root = Path(__file__).parent.parent.parent
            data_folder = repo_root / "data"

        # store the path where auxiliary data files are expected
        self.data_folder = Path(data_folder)  # convert to Path object
        # precompute water IOP file content (raw array)
        self._load_water_iops()
        # load G tables into memory (but don't create RegularGridInterpolator)
        self._load_G_tables()
        # load aph LUT
        self._load_aph()

        # caches for per-wavelength computations (LRU)
        # _wl_cache: maps a hash of the wl bytes -> (aw, bbw) for that wl grid
        # _aph_interp_cache: maps (wl_hash, aph_bin) -> interpolated aph template
        self._wl_cache = OrderedDict()
        self._aph_interp_cache = OrderedDict()
        self._wl_cache_max = 8
        self._aph_interp_cache_max = 64

        # keep the compiled model function (callable) for fast repeated use
        self.model = self.model_3C()

    def _load_water_iops(self):
        # Load water absorption (aw) and pure-water backscatter (bbw) table.
        # Filename is historical; the file is expected to have three columns:
        # wavelength, aw, bbw and some header lines (skipped with skiprows=8).
        water_IOPs_pathfilename = (
            self.data_folder / "abs_scat_seawater_20d_35PSU_20230922_short.txt"
        )
        # aw: WOPP v3. bbw: Zhang 2009
        raw = np.loadtxt(water_IOPs_pathfilename, skiprows=8)
        # if file had extra trailing row handling in original code
        raw = raw[:-1] if raw.shape[1] >= 3 else raw
        # store the raw table for interpolation in the forward model
        self.lw_aw_bw = raw  # columns: wavelength, aw, bbw

    def _load_G_tables(self):
        # The G coefficients are stored in text files.
        # Reshape transpose them to match axes order expected by the evaluator.
        az = np.arange(0, 181, 15)
        tv = np.concatenate([np.arange(0, 81, 10), [87.5]])
        ts = tv.copy()

        # helper to load and reshape
        def ld3(fname):
            path = os.path.join(self.data_folder, fname)
            m = np.loadtxt(path)
            arr = m.reshape(len(az), len(ts), len(tv)).transpose(1, 2, 0)
            return arr

        # store grid axes (useful for debugging or alternative interpolation)
        self._G_ts = ts
        self._G_tv = tv
        self._G_az = az
        # load the 4 G tables as arrays
        self._G0w = ld3("G0w.txt")
        self._G1w = ld3("G1w.txt")
        self._G0p = ld3("G0p.txt")
        self._G1p = ld3("G1p.txt")
        # build scipy RegularGridInterpolator objects (fast C evaluation)
        from scipy.interpolate import RegularGridInterpolator

        # The interpolators are created using axes order (ts, tv, az)
        self._G_i0 = RegularGridInterpolator((ts, tv, az), self._G0w)
        self._G_i1 = RegularGridInterpolator((ts, tv, az), self._G1w)
        self._G_i2 = RegularGridInterpolator((ts, tv, az), self._G0p)
        self._G_i3 = RegularGridInterpolator((ts, tv, az), self._G1p)

    def _load_aph(self):
        # --- load aph templates  ---
        aph_pathfilename = self.data_folder / "vars_aph_v2.npz"
        aph_data = np.load(aph_pathfilename)
        # wavelength axis (squeezed in case it was saved as (nw,) or (1,nw))
        self.l_int = aph_data["l_int"].squeeze()
        # 'aph_norm_55' is expected to contain the 55 templates already normalized
        # by their value around 670 nm (i.e. my_spec / normF from the original v7).
        # vars_aph.npz is expected to contain 'aph_norm_55' and 'aph670_bounds'. Use them directly
        self._aph_norm_55 = aph_data["aph_norm_55"]  # shape (55, nw)
        self._aph670_bounds = aph_data["aph670_bounds"].squeeze()

    def _G_eval(self, s, v, a):
        """Evaluate G-functions using the prebuilt RegularGridInterpolator objects.

        Parameters
        ----------
        s : solar zenith (degrees)
        v : sensor zenith (degrees)
        a : relative azimuth (degrees)

        Returns
        -------
        (G0w, G1w, G0p, G1p) : tuple of floats evaluated at the given geometry
        """
        # ensure absolute positive value for azimuth
        a = abs(float(a))
        pt = (float(s), float(v), a)
        # use the interpolator objects created in _load_G_tables
        return (
            float(self._G_i0(pt)),
            float(self._G_i1(pt)),
            float(self._G_i2(pt)),
            float(self._G_i3(pt)),
        )

    def model_3C(self):
        # cache references locally for speed (these variables are closed over by f)
        # lw_aw_bw = None  # will be loaded inside f to ensure up-to-date
        l_int = self.l_int
        aph670_bounds = self._aph670_bounds
        aph_norm_55 = self._aph_norm_55

        # The forward model is returned as a callable `f` so that repeated calls
        # execute the inner implementation without attribute lookups on the class.
        def f(wl, beta, alpha, C, N, Y, SNAP, Sg, geom, anc, LiEs, rho, rho_d, rho_s):
            # short local refs avoid repeated attribute lookup
            lw_aw_bw = self.lw_aw_bw
            G_eval = self._G_eval
            preG = getattr(self, "_G_precomputed", None)

            # unpack ancillary parameters and geometry
            (am, rh, pressure) = anc
            (theta_s, theta_v, phi) = geom
            cosths = np.cos(np.deg2rad(theta_s))

            # Atmospheric partition (Gregg and Carder 1990)
            z3 = -0.1417 * alpha + 0.82
            z2 = 0.65 if alpha > 1.2 else z3
            z1 = 0.82 if alpha < 0 else z2
            B3 = np.log(1 - z1)
            B2 = B3 * (0.0783 + B3 * (-0.3824 - 0.5874 * B3))
            B1 = B3 * (1.459 + B3 * (0.1595 + 0.4129 * B3))
            Fa = 1 - 0.5 * np.exp((B1 + B2 * cosths) * cosths)
            wl_a = 550.0
            omega_a = (-0.0032 * am + 0.972) * np.exp(3.06e-4 * rh)
            tau_a = beta * (wl / wl_a) ** (-alpha)
            M = 1.0 / (cosths + 0.50572 * (90.0 + 6.07995 - theta_s) ** (-1.6364))
            M_ = M * pressure / 1013.25
            # Rayleigh transmittance term (Tr)
            Tr = np.exp(
                -M_ / (115.6406 * (wl / 1000.0) ** 4 - 1.335 * (wl / 1000.0) ** 2)
            )
            # aerosol transmittance term (Tas)
            Tas = np.exp(-omega_a * tau_a * M)
            Esd = Tr * Tas
            # Ess term accounts for the scattered skylight partition
            Ess = 0.5 * (1 - Tr**0.95) + Tr**1.5 * (1 - Tas) * Fa
            EsdEs = Esd / (Esd + Ess)
            EssEs = 1 - EsdEs

            # Surface reflection Rg: combination of instrument sky term and
            # partitioned atmospheric terms weighted by rho coefficients
            Rg = rho * LiEs + rho_d * EsdEs / np.pi + rho_s * EssEs / np.pi

            # Create a hash key for the current wavelength grid
            wl_key = hashlib.sha1(wl.tobytes()).hexdigest()
            cached_wl = self._wl_cache.get(wl_key)
            if cached_wl is None:
                # interpolate aw and bbw to the requested wavelength grid
                aw = np.interp(wl, lw_aw_bw[:, 0], lw_aw_bw[:, 1])
                bbw = np.interp(wl, lw_aw_bw[:, 0], lw_aw_bw[:, 2]) / 2.0
                # store in LRU cache (pop oldest if at capacity)
                if len(self._wl_cache) >= self._wl_cache_max:
                    self._wl_cache.popitem(last=False)
                self._wl_cache[wl_key] = (aw, bbw)
            else:
                aw, bbw = cached_wl

            # phytoplankton absorption: select template by C bin
            aph670 = 0.019092732293411 * C**0.955677209349669
            bin_idx = np.searchsorted(aph670_bounds, aph670)

            tpl_key = f"{wl_key}:bin{bin_idx}"
            base_interp = self._aph_interp_cache.get(tpl_key)
            if base_interp is None:
                # interpolate the stored aph template to current wl using the
                # internal template wavelength axis l_int, ignoring NaNs
                base_spec = aph_norm_55[bin_idx]
                valid = ~np.isnan(base_spec)
                base_interp = np.interp(wl, l_int[valid], base_spec[valid])
                # store in LRU cache
                if len(self._aph_interp_cache) >= self._aph_interp_cache_max:
                    self._aph_interp_cache.popitem(last=False)
                self._aph_interp_cache[tpl_key] = base_interp

            # scale template to absolute aph using aph670 estimate
            aph = base_interp * aph670
            aph = np.maximum(aph, 0.0)

            # NAP absorption (non-algal particles)
            laNAPstar440 = -0.1886 * np.exp(-1.0551 * np.log10(C / N)) - 1.27
            laNAPstar440 = np.clip(laNAPstar440, -3, -0.5)
            aNAP440 = (10**laNAPstar440) * N
            aNAP = aNAP440 * np.exp(np.clip(-SNAP * (wl - 440), -700, 700))

            # CDOM absorption (exponential slope Sg)
            ag = Y * np.exp(np.clip(-Sg * (wl - 440), -700, 700))

            # Particulate backscattering model (power-law)
            T = N + 0.07 * C  # Brando and Dekker (2003)
            bbp560 = (
                0.013207124962878 * T**0.814981148538268
            )  # power law derived from PB24
            eta = 0.8624 - 0.2147 * np.log10(T) - 0.06528 * C / T  # Derived from PB24
            bbp = bbp560 * (560 / wl) ** eta
            bb = bbw + bbp

            a = aw + aph + aNAP + ag
            omw = bbw / (a + bb)
            omp = (bb - bbw) / (a + bb)

            # use fast G eval: either reuse precomputed tuple or interpolate
            if preG is not None:
                G0w, G1w, G0p, G1p = preG
            else:
                G0w, G1w, G0p, G1p = G_eval(theta_s, theta_v, phi)
            # final modeled remote sensing reflectance (unit: sr^-1)
            Rrs = (G0w + G1w * omw) * omw + (G0p + G1p * omp) * omp

            return Rrs, Rg

        return f

    def fit_LtEs(
        self, wl, LiEs, LtEs, params, weights, geom, anc, method="leastsq", verbose=True
    ):
        """
        Fit Lt/Es by minimizing residuals. Returns (out, Rrs_mod, Rg)

        Notes:
        - `params` is expected to be an lmfit.Parameters instance with the
          parameter names used inside the model (beta, alpha, C, N, Y, SNAP,
          Sg, rho, rho_d, rho_s). The code calls p.valuesdict() to obtain
          numeric values for each evaluation.
        - The residual function returns a concatenated vector with a data
          residual part and a penalty part to discourage negative Rg.
        """
        # ensure numpy arrays
        wl = np.asarray(wl)
        LiEs = np.asarray(LiEs)
        LtEs = np.asarray(LtEs)
        weights = np.asarray(weights)

        def resid(p):
            # fetch current parameter values from lmfit's Parameters container
            pv = p.valuesdict()
            # evaluate model using current parameter vector
            Rrs_mod, Rg = self.model(
                wl,
                pv["beta"],
                pv["alpha"],
                pv["C"],
                pv["N"],
                pv["Y"],
                pv["SNAP"],
                pv["Sg"],
                geom,
                anc,
                LiEs,
                pv["rho"],
                pv["rho_d"],
                pv["rho_s"],
            )
            # return vector residuals scaled by sqrt(weights) so least-squares objective
            # equals sum(weights * (LtEs - Rrs_mod - Rg)**2)

            res_data = (LtEs - Rrs_mod - Rg) * np.sqrt(weights)

            # per-wavelength penalty: neg = max(0, -Rg)
            neg = np.clip(-Rg, 0.0, None)  # same shape as wl
            penalty_weight = 1e6
            res_pen = (
                np.sqrt(penalty_weight) * neg
            )  # linear in neg -> objective ~ penalty_weight * neg^2

            return np.concatenate([res_data, res_pen])  # fixed size: 2*nw

        # Precompute G functions if geom is constant for this fit
        # _G_precomputed is used by model_3C to bypass interpolator calls
        self._G_precomputed = self._G_eval(*geom)
        try:
            out = lm.minimize(resid, params, method=method)
            # Print common attributes if present
            for attr in (
                "message",
                "ier",
                "nfev",
                "nit",
                "success",
                "status",
                "x",
                "cost",
                "fun",
            ):
                if hasattr(out, attr):
                    print(f"{attr}: {getattr(out, attr)}")

            # If this was an lmfit / MINPACK run, interpret the `ier` code (1..5)
            if hasattr(out, "ier") and getattr(out, "ier") is not None:
                try:
                    ier = int(getattr(out, "ier"))
                except Exception:
                    ier = None
                if ier is not None:
                    minpack_reasons = {
                        1: "converged: ftol satisfied (small relative reduction in sum-of-squares).",
                        2: "converged: xtol satisfied (small change in parameters).",
                        3: "converged: ftol and xtol both satisfied.",
                        4: "converged: gtol satisfied (orthogonality/gradient condition).",
                        5: "stopped: maxfev reached (maximum function evaluations).",
                    }
                    print(
                        "MINPACK ier:",
                        ier,
                        "-",
                        minpack_reasons.get(ier, "unknown / user-requested stop."),
                    )

            # For scipy.optimize.OptimizeResult (least_squares) the message field is usually authoritative:
            if hasattr(out, "message") and not hasattr(out, "ier"):
                print("Optimizer message:", getattr(out, "message"))

            pv = out.params.valuesdict()
            # Re-evaluate model with optimized parameters to return final Rrs and Rg
            Rrs_mod, Rg = self.model(
                wl,
                pv["beta"],
                pv["alpha"],
                pv["C"],
                pv["N"],
                pv["Y"],
                pv["SNAP"],
                pv["Sg"],
                geom,
                anc,
                LiEs,
                pv["rho"],
                pv["rho_d"],
                pv["rho_s"],
            )

            # --- print optimized parameters ---
            print("\nOptimized parameters:")
            # lmfit result: print value, stderr (if present), bounds and whether it varied
            for name, par in out.params.items():
                stderr = getattr(par, "stderr", None)
                stderr_str = f" ± {stderr:.3g}" if stderr not in (None, 0) else ""
                print(
                    f"{name:12s} value={par.value:.6g}{stderr_str}   min={par.min}   max={par.max}   vary={par.vary}"
                )

        finally:
            # ensure we remove the precomputed G to avoid stale cached values
            if hasattr(self, "_G_precomputed"):
                del self._G_precomputed
        return out, Rrs_mod, Rg


if __name__ == "__main__":
    # Example / diagnostic run when the module is executed directly.
    # Adjust `data_folder` to your local path if needed.
    repo_root = Path(__file__).parent.parent.parent
    data_folder = repo_root / "data"
    examples_folder = repo_root / "examples"
    data = pd.read_csv(
        os.path.join(examples_folder, "example_single_spectrum.csv"),
        index_col=0,
        skiprows=15,
    )
    # extract wavelength and columns: Li, Lt, Es
    wl = data.index.values
    Li = data.iloc[:, 0].values
    Lt = data.iloc[:, 1].values
    Es = data.iloc[:, 2].values
    geom = (59, 35, 100)  # Jetty 2
    am = 4
    rh = 60
    pressure = 1013.25
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

    # weights: remove H2O band, downweight UV, upweight NIR
    weights = np.where(
        (wl >= 760) & (wl <= 765),
        0,  # Remove the ugly H2O feature
        np.where(
            wl < 400, 1, np.where(wl > 800, 5, 1)  # Some potential troubles at the UV
        ),
    )  # Let's power-up the NIR

    # optional profiling block to inspect performance
    import cProfile
    import pstats

    pr = cProfile.Profile()
    pr.enable()
    try:
        reg, R_rs_mod, Rg = model.fit_LtEs(
            wl, Li / Es, Lt / Es, params, weights, geom, anc=(am, rh, pressure)
        )
    except Exception as e:
        # Ensure profiler is disabled on error and provide diagnostics
        pr.disable()
        print("Error during model.fit_LtEs:", repr(e))
        # Re-raise so the calling environment is aware (remove the raise if you prefer to continue)
        raise
    else:
        pr.disable()
        ps = pstats.Stats(pr).sort_stats("cumtime")
        ps.print_stats(30)

    # run a second fit (often used to inspect repeatability / cached behaviour)
    reg, R_rs_mod, Rg = model.fit_LtEs(
        wl, Li / Es, Lt / Es, params, weights, geom, anc=(am, rh, pressure)
    )

    # Quick plotting of measured vs modeled quantities
    plt.figure()
    plt.grid(True)
    plt.plot(wl, Lt / Es, label="L_t/E_s, measured")
    plt.plot(wl, R_rs_mod + Rg, label="L_t/E_s, modeled")
    plt.plot(wl, R_rs_mod, label=" R_rs, modeled")
    plt.plot(wl, Rg, label="R_g, 3C output")
    plt.plot(wl, Lt / Es - Rg, label="R_rs, 3C output")
    plt.xlabel("wavelength (nm)")
    plt.ylabel("Various reflectances (sr^(-1))")
    plt.legend()
    plt.tight_layout()
    plt.show()
