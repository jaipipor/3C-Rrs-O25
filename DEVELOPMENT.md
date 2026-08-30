# Development Guide

This guide describes how to set up, develop, test, and package
**3C-Rrs-O25**.

## Repository layout

```text
3C-Rrs-O25/
├── data/                     # Authoritative ancillary scientific data
├── examples/                 # Example inputs and workflows
├── notebooks/                # Interactive examples
├── params/                   # Example parameter configuration
├── scripts/                  # Command-line utilities
├── src/
│   └── rrs3c/
│       ├── __init__.py
│       └── model.py          # Core model
├── tests/                    # Automated tests
├── tools/                    # Setup and release helpers
├── CHANGELOG.md
├── CITATION.cff
├── DEVELOPMENT.md
├── LICENSE
├── README.md
├── pyproject.toml
└── requirements.txt
```

The recommended workflow is:

1. Clone the repository.
2. Create a virtual environment.
3. Install the project in editable mode.
4. Create a focused development branch.
5. Implement and document the change.
6. Run tests and quality checks.
7. Commit, push, and open a pull request.

---

## 1. Requirements

The project requires:

- Python 3.10 or later
- pip
- Git, GitHub Desktop, or another Git client

Check available Python versions on Windows:

```powershell
py --list
```

Check Python on Linux or macOS:

```bash
python3 --version
```

Git LFS is not required. The ancillary scientific files are stored directly
in Git.

---

## 2. Obtain the repository

Clone with Git:

```bash
git clone https://github.com/jaipipor/3C-Rrs-O25.git
cd 3C-Rrs-O25
```

Alternatively, clone `jaipipor/3C-Rrs-O25` with GitHub Desktop and use
**Repository > Open in Visual Studio Code**.

---

## 3. Create a virtual environment

### Windows PowerShell

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Python 3.11, 3.12, or 3.13 may be used instead.

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1
```

### Linux and macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

---

## 4. Install the project

Install the complete development environment:

```bash
python -m pip install -e ".[timeseries,notebooks,dev]"
```

The equivalent convenience command is:

```bash
python -m pip install -r requirements.txt
```

Other installation choices are:

```bash
python -m pip install -e .
python -m pip install -e ".[examples]"
python -m pip install -e ".[timeseries]"
python -m pip install -e ".[notebooks]"
```

The optional groups provide:

- `examples`: pandas and Matplotlib
- `timeseries`: pandas, Matplotlib, and xarray
- `notebooks`: Jupyter and notebook execution tools
- `dev`: pytest, Ruff, Black, pre-commit, build, and Twine

Check dependency consistency:

```bash
python -m pip check
```

Expected output:

```text
No broken requirements found.
```

---

## 5. Ancillary scientific data

The authoritative ancillary files are stored under `data/`:

```text
data/G0p.txt
data/G0w.txt
data/G1p.txt
data/G1w.txt
data/abs_scat_seawater_20d_35PSU_20230922_short.txt
data/vars_aph_v2.npz
```

During installation, these files are mapped to the private package
`rrs3c._data`. Default construction therefore requires no explicit path:

```python
from rrs3c import rrs_model_3C_O25

model = rrs_model_3C_O25()
```

A different ancillary-data directory may be selected explicitly:

```python
model = rrs_model_3C_O25(
    data_folder="path/to/alternative/data",
)
```

Do not modify the scientific ancillary files without documenting their
provenance and validating the resulting model output.

---

## 6. Verify the installation

Check the active interpreter:

```bash
python -c "import sys; print(sys.executable)"
```

Check the installed package:

```bash
python -c "import rrs3c; print(rrs3c.__file__)"
```

Check model construction and data discovery:

```bash
python -c "from rrs3c import rrs_model_3C_O25; model = rrs_model_3C_O25(); print(model.data_folder)"
```

For an editable installation, the data path normally points to the
repository-level `data/` directory.

For a wheel installation, the path normally points to:

```text
site-packages/rrs3c/_data
```

---

## 7. Create a development branch

Do not make substantive changes directly on `main`.

Suggested branch names include:

```text
fix/input-validation
fix/package-metadata
test/model-regression
docs/install-instructions
feature/model-enhancement
```

Use GitHub Desktop to create and publish the branch if preferred.

Keep each branch and commit focused on one clear purpose.

---

## 8. Development principles

Prioritize:

1. correctness
2. scientific rigor
3. reproducibility
4. maintainability
5. performance

Keep the code concise and readable. Avoid unnecessary abstractions and
defensive handling of unlikely cases.

Separate scientific-model changes from packaging, testing, documentation,
and formatting changes.

Do not change model equations, parameter bounds, optimization behavior, or
ancillary data without explaining and validating the numerical consequences.

---

## 9. Run tests and quality checks

Run the complete test suite:

```bash
python -m pytest -q
```

Run a focused test file when appropriate:

```powershell
python -m pytest ".\tests\test_model_validation.py" -v
```

Run Ruff:

```bash
python -m ruff check .
```

Run Black in verification mode:

```bash
python -m black --check .
```

Run all pre-commit hooks:

```bash
python -m pre_commit run --all-files
```

Install the hooks locally:

```bash
python -m pre_commit install
```

If a hook modifies a file, review the change and rerun the complete command.

Do not update numerical regression references merely to make a test pass.
Any intentional numerical change must be scientifically justified and
documented.

---

## 10. Run the examples

### Single-spectrum example

```powershell
python ".\scripts\run_3C.py" `
    --input ".\examples\example_single_spectrum.csv" `
    --theta-s 59 `
    --theta-v 35 `
    --phi 100 `
    --am 4 `
    --rh 60 `
    --pressure 1013.25 `
    --no-plot
```

To use the example JSON parameter configuration, add:

```powershell
--params-file ".\params\params.json"
```

### Time-series example

Display the available options:

```powershell
python ".\examples\timeseries\src\run_timeseries.py" --help
```

A typical execution is:

```powershell
python ".\examples\timeseries\src\run_timeseries.py" `
    --input-folder ".\examples\timeseries\data" `
    --output-folder ".\examples\timeseries\output" `
    --date 20200530
```

The time-series workflow is specific to the supplied example input format.
Changes to this workflow do not necessarily change the core model.

---

## 11. Run the notebooks

Install the required dependencies:

```bash
python -m pip install -e ".[timeseries,notebooks]"
```

Start Jupyter Lab:

```bash
python -m jupyter lab
```

The repository includes:

```text
notebooks/01_QuickStart_Example.ipynb
notebooks/02_TimeSeries_Processing.ipynb
```

In Visual Studio Code, select the interpreter from the project `.venv` as
the notebook kernel.

If necessary, register the environment:

```bash
python -m ipykernel install --user --name 3c-rrs-o25 --display-name "3C-Rrs-O25 (.venv)"
```

Before committing notebook changes, run all cells in order and inspect their
outputs.

---

## 12. Build and check the distributions

Remove old build artifacts.

### Windows PowerShell

```powershell
Remove-Item -Recurse -Force ".\build", ".\dist" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force ".\src\rrs3c.egg-info" -ErrorAction SilentlyContinue
```

### Linux and macOS

```bash
rm -rf build dist src/rrs3c.egg-info
```

Build the source distribution and wheel:

```bash
python -m build
```

Check their metadata:

```bash
python -m twine check dist/*
```

The Windows release helper performs these operations:

```powershell
.\tools\make_release.ps1 -Version 1.0.4
```

Use the appropriate version for future releases.

Do not commit `build/`, `dist/`, or `*.egg-info/`.

---

## 13. Verify the wheel

Test the built wheel outside the repository.

### Windows PowerShell

```powershell
$wheelTest = Join-Path $env:TEMP "rrs3c-wheel-test"

Remove-Item -Recurse -Force $wheelTest -ErrorAction SilentlyContinue

py -3.13 -m venv $wheelTest

$wheel = Get-ChildItem ".\dist\rrs3c-*.whl" |
    Select-Object -First 1

& "$wheelTest\Scripts\python.exe" `
    -m pip install `
    $wheel.FullName

& "$wheelTest\Scripts\python.exe" -m pip check
```

Move outside the repository:

```powershell
Push-Location $env:TEMP
```

Verify the imported package:

```powershell
& "$wheelTest\Scripts\python.exe" -c "import rrs3c; print('Imported package:', rrs3c.__file__)"
```

Verify model construction:

```powershell
& "$wheelTest\Scripts\python.exe" -c "from rrs3c import rrs_model_3C_O25; model = rrs_model_3C_O25(); print('Data folder:', model.data_folder); print('Water table shape:', model.lw_aw_bw.shape); print('Wheel construction passed')"
```

The data folder should point inside:

```text
site-packages/rrs3c/_data
```

Return to the repository:

```powershell
Pop-Location
```

Remove the temporary environment when finished:

```powershell
Remove-Item -Recurse -Force $wheelTest
```

---

## 14. Commit and open a pull request

Before committing, run:

```bash
python -m pip check
python -m pytest -q
python -m ruff check .
python -m black --check .
python -m pre_commit run --all-files
```

When affected, also run:

- the single-spectrum example
- the time-series example
- both notebooks
- the distribution build
- `twine check`
- the clean wheel test

Do not commit generated files such as:

```text
build/
dist/
*.egg-info/
.pytest_cache/
.ruff_cache/
__pycache__/
.ipynb_checkpoints/
examples/timeseries/output/
```

Use a concise commit title, for example:

```text
fix: validate wavelength ordering
test: add numerical regression coverage
build: package ancillary model data
docs: update installation guidance
```

In the pull request, describe:

- the purpose of the change
- the files or behavior changed
- the scientific impact
- the validation performed
- any effect on numerical output

Do not merge while required CI checks are failing.

---

## 15. Prepare a release

Before releasing a new version:

1. update the version in `pyproject.toml`;
2. update `CHANGELOG.md`;
3. update `CITATION.cff`;
4. confirm that release dates agree;
5. run all tests and quality checks;
6. run affected examples and notebooks;
7. build the wheel and source distribution;
8. run `twine check`;
9. test the wheel outside the repository;
10. confirm that CI passes.

The release wheel must contain the required resources under:

```text
rrs3c/_data/
```

Publish only artifacts built from the final reviewed release commit.

---

## 16. Report problems

For reproducible problems, open a GitHub issue and include:

- the failing command or Python code
- the complete traceback
- the operating system
- the Python version
- the output of `python -m pip check`
- the installation method
- whether editable or wheel installation was used
- whether default packaged data or an explicit `data_folder` was used

For scientific discrepancies, also include:

- the wavelength range
- observation geometry
- atmospheric ancillary values
- parameters and bounds
- spectral weights
- expected and observed results

Do not include confidential data, credentials, or access tokens.
