# DEVELOPMENT — quick reference

This document contains the minimal, pragmatic instructions you need to develop, test and contribute to this repository on **Windows**, **macOS** and **Linux**. It focuses on the tasks that caused friction earlier: virtual environments, editable install, pre‑commit hooks, linting fixes and running examples.

> **Goal:** make it easy to run examples locally *without* installing the package system‑wide, keep consistent formatting with pre‑commit, and explain how to recover when hooks modify files.

---

## 1. Create and activate a virtual environment

> **Why:** isolate dependencies and ensure the same tools (black, ruff, pytest) run from the project venv.

### Windows (PowerShell)

```powershell
cd C:\path\to\3C-Rrs-O25
python -m venv .venv
# activate
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS / Linux (bash)

```bash
cd /path/to/3C-Rrs-O25
python -m venv .venv
# activate
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Tip:** if the repository provides a `tools/setup_project.ps1` script, it may create and populate the venv automatically — read that script before running it.

---

## 2. Install the package in editable mode (recommended)

Installing editable allows `import rrs3c` to work everywhere without fiddling with `PYTHONPATH`.

```bash
# from repo root, with venv active
pip install -e .
```

After this, `python -c "from rrs3c.model import rrs_model_3C; print('ok')"` should print `ok`.

---

## 3. Pre-commit: install and run hooks

Pre-commit enforces consistent formatting (black, isort) and linting (ruff). Install the git hooks and run them.

```bash
# install hooks (one-time)
python -m pre_commit install
# run all hooks on all files (recommended before committing)
python -m pre_commit run --all-files
```

### What happens when a hook *changes* files?

1. A hook like `black` or `isort` may automatically rewrite files during the commit attempt.
2. The commit will *fail* and report the modified files.
3. You must add the modified files and re-run the commit, e.g.:

```bash
git add <files changed by pre-commit>
git commit -m "chore: format via pre-commit"
```

If pre-commit stashes unstaged changes and fails with `Stashed changes conflicted with hook auto-fixes... Rolling back fixes...`, ensure you:

* `git status` to see unstaged and stashed files
* `git add` the modified files and re-commit

**Last resort:** bypass hooks for a single commit (not recommended):

```bash
git commit --no-verify -m "temporary bypass"
```

---

## 4. Ruff / E402 situation (imports after `sys.path` changes)

Ruff flags `E402` when an import isn't at the top of the file. In example scripts we intentionally add `src/` to `sys.path` before importing the package.

**Best practice:** add an inline ignore on the specific import line:

```python
from rrs3c.model import rrs_model_3C  # noqa: E402
```

This documents the reason and silences Ruff for that line only. Don’t globally ignore `E402` unless you’re certain.

---

## 5. Running the examples

There are two ways:

### A — Without installing (uses the script's `sys.path` shim)

Run the example script from its folder so it can find the repository `src/` folder automatically.

```bash
cd examples/timeseries/src
python run_timeseries.py --input-file ../example_time_series_data.csv --output-folder ../output --plot --verbose
```

### B — After editable install

```bash
# from anywhere
python examples/timeseries/src/run_timeseries.py --input-file examples/timeseries/example_time_series_data.csv
```

If you still get `ModuleNotFoundError: No module named 'rrs3c'`, either run from the example folder (A) or set `PYTHONPATH`:

**Windows PowerShell**

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python examples\timeseries\src\run_timeseries.py --input-file examples\timeseries\example_time_series_data.csv
```

**Unix**

```bash
PYTHONPATH=$(pwd)/src python examples/timeseries/src/run_timeseries.py --input-file examples/timeseries/example_time_series_data.csv
```

---

## 6. Running linters and auto-fixes locally

If a tool in `.venv` is not on your PATH (e.g. `ruff`), call it via the Python module:

```bash
# fixable ruff issues (uses .venv Python)
python -m ruff check . --fix
# format with black
python -m black .
# sort imports with isort
python -m isort .
```

On Windows, replace `python` with the exact Python from the venv if needed: `.venv\Scripts\python.exe`.

---

## 7. Tests

Run the test suite with pytest:

```bash
pytest -q
```

If a test fails, paste the full traceback into issues so it’s reproducible.

---

## 8. CI hints

A simple GitHub Actions workflow (`.github/workflows/ci.yml`) should:

1. Checkout
2. Setup Python
3. Install requirements
4. Run `pre-commit run --all-files`
5. Run `pytest`

This helps catching formatting or import regressions on PRs.

---

## 9. Common problems & quick fixes

* **Hook modifies files but commit fails:** `git add` the modified files and re-run `git commit`.
* **`pre-commit` fetch errors when installing hooks:** network issue or transient remote tag. Re-run `pre-commit install` later; ensure your Git client can fetch.
* **`ruff`: "F821 Undefined name 'rrs3c'"**: Usually indicates tests import incorrectly. Edit the test to `from rrs3c.model import rrs_model_3C` and use it directly.
* **Interpolation ValueError in example:** check raw CSV rows for malformed lengths or missing values; the loader `load_jetty_data` should validate row lengths.

---

## 10. If you want me to make the edits for you

I can generate the exact file contents (pyproject.toml, CI file, DEVELOPMENT.md) and prepare a patch you can `git apply`. Tell me which files you want automated and I will produce them.

---

*End of DEVELOPMENT.md*
