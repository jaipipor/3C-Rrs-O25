# 3C-Rrs-O25: Analytical 3-component Rrs model 🌊

3C-Rrs-O25 is a Python implementation of the **3C-O25 model** for processing above-water radiometry. It fits measured `Lt/Es` spectra and separates:

- `Rrs`: remote-sensing reflectance
- `Rg`: modeled surface-reflection term

The core implementation is in `src/rrs3c/model.py`. A command-line runner, examples, notebooks, and demonstration datasets are also included.

## Features ✨

- Analytical **3C-O25** forward model for `Lt/Es`
- `lmfit`-based inversion through `fit_LtEs(...)`
- Packaged ancillary scientific resources
- Optional alternative ancillary-data directory
- LRU caching for repeated wavelength grids
- Validation of spectral inputs, wavelengths, weights, geometry, and fitting parameters
- Single-spectrum command-line runner
- JSON parameter overrides
- Example time-series workflow and Jupyter notebooks
- Automated tests, including numerical regression
- Cross-platform support for Windows, Linux, and macOS

## Repository layout 📁

```text
3C-Rrs-O25/
├── data/                  # authoritative ancillary scientific data
├── examples/              # example inputs and workflows
├── notebooks/             # interactive tutorials
├── params/                # example parameter configuration
├── scripts/
│   └── run_3C.py          # single-spectrum command-line runner
├── src/
│   └── rrs3c/
│       ├── __init__.py
│       └── model.py       # core 3C-O25 implementation
├── tests/                 # automated tests
├── tools/                 # setup and release helpers
├── CHANGELOG.md
├── CITATION.cff
├── DEVELOPMENT.md
├── LICENSE
├── README.md
├── pyproject.toml
└── requirements.txt
```

## Installation 🚀

Python 3.10 or later is required. A virtual environment is recommended.

### Install from the repository

#### Windows PowerShell 🪟

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

If PowerShell blocks activation scripts, allow locally created scripts for the current session:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1
```

#### Linux and macOS 🐧🍎

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Verify the installation:

```bash
python -c "from rrs3c import rrs_model_3C_O25; print('rrs3c import successful')"
```

Development installation, testing, formatting, packaging, and release instructions are provided in [`DEVELOPMENT.md`](DEVELOPMENT.md).

### Install from a wheel ☸️

On Windows PowerShell:

```powershell
python -m pip install ".\dist\rrs3c-1.0.4-py3-none-any.whl"
```

On Linux or macOS:

```bash
python -m pip install "./dist/rrs3c-1.0.4-py3-none-any.whl"
```

The wheel includes the ancillary scientific resources required by the default model constructor.

## Ancillary data 🗂️

The model requires scientific ancillary resources containing:

- water absorption and backscattering data
- phytoplankton absorption templates
- G-function lookup tables

The authoritative source files are stored in the repository-level `data/` directory. The resources required by the core model are also included in the installed package.

The default constructor therefore works without an explicit data path:

```python
from rrs3c import rrs_model_3C_O25

model = rrs_model_3C_O25()
```

If required, an alternative ancillary-data directory can be supplied explicitly:

```python
model = rrs_model_3C_O25(
    data_folder="path/to/alternative/data",
)
```

The explicit `data_folder` argument takes precedence over the resources distributed with the package.

Do not edit or replace the scientific ancillary resources without documenting their provenance and validating the resulting model output.

## Python API 🐍

Import and construct the model:

```python
from rrs3c import rrs_model_3C_O25

model = rrs_model_3C_O25()
```

Fit a measured spectrum:

```python
out, Rrs_mod, Rg = model.fit_LtEs(
    wl=wl,
    LiEs=LiEs,
    LtEs=LtEs,
    params=params,
    weights=weights,
    geom=(theta_s, theta_v, phi),
    anc=(am, rh, pressure),
)
```

The method returns:

- `out`: `lmfit` optimization result
- `Rrs_mod`: modeled remote-sensing reflectance spectrum
- `Rg`: modeled surface-reflection spectrum

Input geometry is expressed as:

```text
(theta_s, theta_v, phi)
```

The three values are the solar zenith, viewing zenith, and relative azimuth angles in degrees.

Atmospheric ancillary inputs are expressed as:

```text
(am, rh, pressure)
```

Here, `am` is air mass, `rh` is relative humidity, and `pressure` is atmospheric pressure.

## Single-spectrum runner ⚡

Run the supplied example from the repository root:

```powershell
python ".\scripts\run_3C.py" --input ".\examples\example_single_spectrum.csv" --theta-s 59 --theta-v 35 --phi 100 --am 4 --rh 60 --pressure 1013.25
```

To run without opening a plot window:

```powershell
python ".\scripts\run_3C.py" --input ".\examples\example_single_spectrum.csv" --theta-s 59 --theta-v 35 --phi 100 --am 4 --rh 60 --pressure 1013.25 --no-plot
```

To override the default fitting parameters:

```powershell
python ".\scripts\run_3C.py" --input ".\examples\example_single_spectrum.csv" --theta-s 59 --theta-v 35 --phi 100 --am 4 --rh 60 --pressure 1013.25 --params-file ".\params\params.json" --no-plot
```

A successful fit should finish without a traceback and report:

```text
success: True
message: Fit succeeded.
```

Use the direct Python API for integration into operational processing chains. The command-line runner is intended for single-spectrum execution, testing, and reproducible examples.

## Examples and notebooks 💡

The repository includes:

- `notebooks/01_QuickStart_Example.ipynb`
- `notebooks/02_TimeSeries_Processing.ipynb`
- a command-line single-spectrum example
- an example time-series workflow

The time-series workflow demonstrates application of the core model to the supplied multi-instrument input format. Its input parsing, metadata handling, geometry calculation, and quality controls are specific to the example.

Install the time-series and notebook dependencies:

```bash
python -m pip install -e ".[timeseries,notebooks]"
```

Open the notebooks with Jupyter Lab:

```bash
python -m jupyter lab
```

If needed, register the virtual environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name 3c-rrs-o25 --display-name "3C-Rrs-O25 (.venv)"
```

Select **3C-Rrs-O25 (.venv)** from the notebook kernel menu.

## Tests and code quality 🧪

Run the test suite from the repository root:

```bash
python -m pytest -q
```

The automated tests cover:

- package import and model construction
- packaged ancillary-data discovery
- ancillary scientific data integrity
- spectral input validation
- wavelength and geometry validation
- G-function table evaluation
- the established single-spectrum numerical regression

Run all configured pre-commit checks:

```bash
python -m pre_commit run --all-files
```

For the complete development workflow, see [`DEVELOPMENT.md`](DEVELOPMENT.md).

## Troubleshooting 🧐

### `ModuleNotFoundError: No module named 'rrs3c'`

Activate the intended virtual environment and install the project from the repository:

```bash
python -m pip install -e .
```

Alternatively, install the built wheel:

```bash
python -m pip install "./dist/rrs3c-1.0.4-py3-none-any.whl"
```

### Model construction cannot find the ancillary data

Check the data directory used by the model:

```bash
python -c "from rrs3c import rrs_model_3C_O25; model = rrs_model_3C_O25(); print(model.data_folder)"
```

For an editable installation, this normally points to the repository-level `data/` directory.

For a wheel installation, this normally points to:

```text
site-packages/rrs3c/_data
```

If an explicit `data_folder` is supplied, confirm that it contains all six required ancillary files.

### PowerShell refuses to activate the environment

Allow locally created scripts for the current PowerShell session:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1
```

### A pre-commit check modifies files

Review the modifications and run the checks again:

```bash
python -m pre_commit run --all-files
```

Formatting and file-hygiene hooks may update files automatically.

## Contributions 🤝

Contributions are welcome.

Before opening a pull request:

1. Install the project in editable mode.
2. Implement and document one focused change.
3. Run the automated tests.
4. Run all pre-commit checks.
5. Run any affected examples or notebooks.
6. Describe any effect on scientific or numerical output.

Please describe the purpose of the change and any effect on model output or the public API.

## Support 📨

For reproducible problems, open a GitHub issue and include:

- the command or Python code that failed
- the complete traceback
- the operating system
- the output of `python --version`
- the output of `python -m pip check`
- whether the package was installed in editable mode or from a wheel
- whether the default packaged data or an explicit `data_folder` was used

For scientific discrepancies, also include the wavelength range, observation geometry, atmospheric ancillary values, fitting parameters, bounds, and spectral weights.

Do not include confidential data, credentials, or access tokens.

## Citation 📜

If you use this software in scientific work, please cite:

> Pitarch, J. A general model for sun and sky glint removal in above-water optical radiometry: mathematical description and Python code. *Earth Science Informatics* 19, 78 (2026).
> [https://doi.org/10.1007/s12145-026-02114-w](https://doi.org/10.1007/s12145-026-02114-w)

Machine-readable citation metadata are provided in `CITATION.cff`.

## License ©️

This project is distributed under the terms specified in the `LICENSE` file.
