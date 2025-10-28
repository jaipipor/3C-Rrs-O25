# 3C-Rrs-O25 — Analytical 3-component Rrs model (O25 variant)

Analytical 3-component remote sensing reflectance (Rrs) model — O25 variant.
This repository contains an optimized NumPy implementation of the 3C Rrs model, small example scripts, Jupyter notebooks, and a tiny demo dataset so users can run and test the code quickly.

---

## Repository layout 📁

```
3C-Rrs-O25/
├─ README.md
├─ LICENSE
├─ data/                     # ancillary files
├─ examples/                 # runnable scripts & small loaders (importable)
├─ notebooks/                # interactive tutorials and demos
│  ├─ 01_QuickStart_Example.ipynb
│  ├─ 02_TimeSeries_Processing.ipynb
│  └─ 03_Parameter_Analysis.ipynb
├─ src/
│  └─ rrs3c/
│     └─ model.py            # core model implementation
├─ tests/
├─ tools/                    # helper scripts for setup (PowerShell, etc.)
├─ .pre-commit-config.yaml
├─ requirements.txt
└─ pyproject.toml / setup.cfg
```

---

## Two recommended ways to run the code

You can run the examples either:

* **A — without installing the package** (quick exploration): run directly from the cloned repo by making `src/` visible to Python (set `PYTHONPATH` or use supplied notebook helper cells).
* **B — editable install** (development): `pip install -e .` in a venv so `import rrs3c` works normally.

Both are shown below for Windows, Linux and macOS.

---

## 1. Create and activate a virtual environment 🚀

### Windows 🪟 (PowerShell)

```powershell
# from the repository root
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks scripts:

```powershell
# allow scripts for this session
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

Or run the setup script with bypass:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\setup_project.ps1
```

### Linux 🐧 / macOS 🍎 (bash / zsh)

```bash
# from the repository root
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

(If using conda, create a conda env and then `pip install -r requirements.txt`.)

---

## 2A. Quickstart — run without installing (recommended for quick tests) ⚡

This workflow keeps the repository self-contained. It requires the venv from above.

### Windows PowerShell 

```powershell
& .\.venv\Scripts\Activate.ps1
# ensure Python can import the local package code from src/
$env:PYTHONPATH = "$PWD\src"

# Start Jupyter Lab (recommended)
jupyter lab

# OR run the example script directly
python examples\run_timeseries.py --input-folder data --input-file example_time_series_data.csv --output-folder examples --date 20200530 --plot
```

### Linux / macOS

```bash
. .venv/bin/activate
export PYTHONPATH="$PWD/src"
jupyter lab

# OR run the example script
python examples/run_timeseries.py --input-folder data --input-file example_time_series_data.csv --output-folder examples --date 20200530 --plot
```

**Notes:**

* The example CLI `examples/run_timeseries.py` accepts `--input-folder`/`--input-file`/`--output-folder` options and `--plot`. Run `python examples/run_timeseries.py --help` to see available flags.
* Notebooks include a helper first cell that attempts to find the repository root and insert `src/` into `sys.path` so `from rrs3c.model import rrs_model_3C` works even if Jupyter was started from a subfolder (e.g. `notebooks/`). If you prefer, start Jupyter from the repository root and set `PYTHONPATH` as above.

---

## 2B. Editable install (recommended for development)🧰

If you will modify `src/` and want the package importable as `rrs3c`:

```bash
# Windows PowerShell
& .\.venv\Scripts\Activate.ps1
python -m pip install -e .

# Linux / macOS (bash)
. .venv/bin/activate
python -m pip install -e .
```

This installs the package in "editable" mode — changes to `src/` are reflected immediately.

Verify install:

```bash
python -c "from rrs3c.model import rrs_model_3C; print('rrs3c import OK')"
```

---

## 3. Jupyter notebooks📓

* Open the notebooks found in `notebooks/` with Jupyter Lab/Notebook.
* If the kernel cannot import `rrs3c`, run the first notebook cell that inserts the repo `src/` directory into `sys.path`. This cell walks up parent directories and finds `src/rrs3c`; it is safe and included in the provided notebooks.
* To ensure the kernel uses the venv Python, install the venv kernel:

```bash
# with venv active
python -m pip install ipykernel
python -m ipykernel install --user --name 3c-rrs-o25 --display-name "3C-Rrs-O25 (.venv)"
```

Select the `3C-Rrs-O25 (.venv)` kernel in the notebook UI.

---

## 4. Example script usage💡

From the repository root (with venv active):

```bash
python examples/run_timeseries.py \
  --input-folder data \
  --input-file example_time_series_data.csv \
  --output-folder examples \
  --date 20200530 \
  --plot
```

Common options:

* `--input-folder`, `-i` : folder with input CSV files (default `data`)
* `--input-file`, `-f`   : filename in the input folder (default `example_time_series_data.csv`)
* `--output-folder`, `-o`: folder to write outputs (default `examples`)
* `--date`, `-d`         : optional date string for naming outputs
* `--plot`               : show interactive plots (if supported by the script)

---

## 5. Troubleshooting — common issues 🧐

### `ModuleNotFoundError: No module named 'rrs3c'`

* You can either:

  * set `PYTHONPATH` to the repository `src/` directory before launching Python/Jupyter, or
  * install the package in editable mode: `pip install -e .` in your activated venv.

Example (Linux/Mac):

```bash
export PYTHONPATH="$PWD/src"
jupyter lab
```

PowerShell:

```powershell
$env:PYTHONPATH = "$PWD\src"
jupyter lab
```

### Notebook import errors (kernel cwd ≠ repo root)

* Run the first helper cell in the notebook (it attempts to discover the repo root and add `src/` to `sys.path`).
* Alternatively, start Jupyter from the repo root and set `PYTHONPATH` as shown above.

### Pre-commit shows "fixable"

* Some linters auto-fix code (black/isort). When they modify files, re-stage the modified files (`git add`) and commit again.

### PowerShell script execution policy

If PowerShell refuses to run a script, either:

```powershell
# allow scripts in current session
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

or run the script with bypass:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\setup_project.ps1
```

---

## 6. Contribution workflow 🤝

1. Fork repository → create a feature branch.
2. Implement changes, run `pre-commit run --all-files` and `pytest`.
3. Commit and push; open a PR describing the changes and rationale.

Suggested commit messages:

* `docs: improve README with cross-platform quickstart`
* `tests: add conftest to support in-repo imports`
* `examples: argparse interface for run_timeseries.py`
* `fix: notebook import when kernel cwd != repo root`

---


## 7. Contact & support 📨

If you encounter reproducible errors:

1. Confirm you are using the repository `python` environment (.venv) and that `src/` is on `PYTHONPATH` or the package is installed editable.
2. Confirm the exact command you ran, the full traceback, the output of `python -V` and `pip list`.
3. Open an issue on GitHub with that information.

---
## 8. Developer notes 🛠️

If you want to contribute, run tests, or edit the code, read `DEVELOPMENT.md` (at the repository root).
It explains how to create a virtual environment, install the package in editable mode, run pre-commit hooks (black/isort/ruff), and run the test suite.

---
## 9. Citation / Acknowledgement 📜

This code implements research in progress. When publishing results obtained with this code, please cite the peer-reviewed paper when available. Current placeholder citation:

> Jaime Pitarch (submitted), A general model for sun and sky glint removal in above-water optical radiometry: mathematical description and Python code.

---
## 10. License ©️

See the repository `LICENSE` file for licensing terms (BSD-3-Clause or as included).

---
