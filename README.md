# 3C-Rrs-O25: Analytical 3-component Rrs model 🌊

3C-Rrs-O25 is a Python implementation of the **3C-O25 model** for processing above-water radiometry. It fits measured `Lt/Es` spectra and separates:

- `Rrs`: remote-sensing reflectance
- `Rg`: modeled surface-reflection term

The core implementation is in `src/rrs3c/model.py`. A command-line runner, examples, notebooks, and a small demonstration dataset are also included.

## Features ✨

- Analytical **3C-O25** forward model for `Lt/Es`
- `lmfit`-based inversion through `fit_LtEs(...)`
- Configurable ancillary-data directory
- LRU caching for repeated wavelength grids
- Single-spectrum command-line runner
- JSON parameter overrides
- Example time-series workflows and Jupyter notebooks
- Cross-platform support for Windows, Linux, and macOS

## Repository layout 📁

```text
3C-Rrs-O25/
├── data/                  # ancillary model data and lookup tables
├── examples/              # example inputs and workflows
├── notebooks/             # interactive tutorials
├── params/                # example parameter configurations
├── scripts/
│   └── run_3C.py          # single-spectrum command-line runner
├── src/
│   └── rrs3c/
│       ├── __init__.py
│       └── model.py       # core 3C-O25 implementation
├── tests/                 # automated tests
├── tools/                 # setup and release helpers
├── DEVELOPMENT.md
├── LICENSE
├── README.md
├── pyproject.toml
└── requirements.txt
```

## Installation 🚀

Python 3.10 or later is required. A virtual environment is recommended.

### Windows PowerShell 🪟

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

If PowerShell blocks activation scripts, allow them for the current session:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

### Linux and macOS 🐧🍎

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

Development instructions, tests, formatting, and release checks are described in [`DEVELOPMENT.md`](DEVELOPMENT.md).

## Ancillary data 🗂️

The model requires the scientific files in the repository-level `data/` directory, including:

- water absorption and backscattering data
- phytoplankton absorption templates
- G-function lookup tables

These files are managed with Git LFS. After cloning the repository, retrieve and verify them with:

```bash
git lfs pull
git lfs fsck
```

If required, an alternative data directory can be supplied explicitly:

```python
model = rrs_model_3C_O25(data_folder="path/to/data")
```

## Python API 🐍

Import and construct the model:

```python
from rrs3c import rrs_model_3C_O25

model = rrs_model_3C_O25(data_folder="data")
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

where the three values are the solar zenith, viewing zenith, and relative azimuth angles in degrees.

Atmospheric ancillary inputs are expressed as:

```text
(am, rh, pressure)
```

where `am` is air mass, `rh` is relative humidity, and `pressure` is atmospheric pressure.

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

- a quick-start notebook
- a time-series processing example

Open the notebooks with Jupyter Lab:

```bash
python -m pip install jupyterlab ipykernel
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

Run all configured pre-commit checks:

```bash
python -m pre_commit run --all-files
```

For the complete development workflow, see [`DEVELOPMENT.md`](DEVELOPMENT.md).

## Troubleshooting 🧐

### `ModuleNotFoundError: No module named 'rrs3c'`

Activate the virtual environment and install the project in editable mode:

```bash
python -m pip install -e .
```

### PowerShell refuses to activate the environment

Allow local scripts for the current PowerShell session:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

### A pre-commit check modifies files

Review the modifications, stage the files again, and repeat the commit. Formatting tools may update files automatically.

## Contributions 🤝

Contributions are welcome.

Before opening a pull request:

1. Install the project in editable mode.
2. Implement and document the change.
3. Run the test suite.
4. Run the pre-commit checks.
5. Confirm that the single-spectrum example still succeeds.

Please describe the purpose of the change and any effect on model output or the public API.

## Support 📨

For reproducible problems, open a GitHub issue and include:

- the command or Python code that failed
- the complete traceback
- your operating system
- the output of `python --version`
- the output of `python -m pip list`
- whether the ancillary files were retrieved through Git LFS

Do not include confidential data, credentials, or access tokens.

## Citation 📜

If you use this software in scientific work, please cite:

> Pitarch, J. A general model for sun and sky glint removal in above-water optical radiometry: mathematical description and Python code. *Earth Science Informatics* 19, 78 (2026).
> https://doi.org/10.1007/s12145-026-02114-w

Machine-readable citation metadata are provided in CITATION.cff.

## License ©️

This project is distributed under the terms specified in the LICENSE file.
