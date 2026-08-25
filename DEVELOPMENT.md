# Development Guide

This document explains how to set up, develop, test, and contribute to **3C-Rrs-O25** on Windows, Linux, and macOS.

The project uses a `src` layout:

```text
3C-Rrs-O25/
├── data/                     # Required ancillary model data
├── examples/                 # Example input data and workflows
├── notebooks/                # Interactive examples
├── scripts/                  # Command-line utilities
├── src/
│   └── rrs3c/
│       ├── __init__.py
│       └── model.py          # Core 3C-O25 implementation
├── tests/                    # Automated tests
├── tools/                    # Development helper scripts
├── CHANGELOG.md
├── LICENSE
├── README.md
├── pyproject.toml
└── requirements.txt
```

The recommended development workflow is:

1. Obtain the complete repository.
2. Verify that the required ancillary files are present in `data/`.
3. Create a virtual environment.
4. Install the project in editable mode.
5. Create a dedicated branch for each change.
6. Run tests and quality checks before committing.
7. Open a pull request for review.

---

## 1. Requirements

The project requires:

- Python 3.10 or later
- pip
- Git, GitHub Desktop, or another Git client
- the complete repository, including the ancillary files under `data/`

Recommended development tools include:

- Visual Studio Code
- GitHub Desktop
- PowerShell on Windows
- Bash or Zsh on Linux and macOS

Check the installed Python versions on Windows:

```powershell
py --list
```

Check the default Python version on Linux or macOS:

```bash
python3 --version
```

---

## 2. Obtain the repository

You may use command-line Git or GitHub Desktop.

### 2.1 Command-line Git

Clone the repository:

```bash
git clone https://github.com/jaipipor/3C-Rrs-O25.git
cd 3C-Rrs-O25
```

If the repository uses Git LFS, make sure Git LFS is installed and retrieve the actual ancillary files:

```bash
git lfs install
git lfs pull
```

Check the repository status:

```bash
git status
```

### 2.2 GitHub Desktop

1. Open GitHub Desktop.
2. Select **File > Clone repository**.
3. Open the **GitHub.com** tab.
4. Select `jaipipor/3C-Rrs-O25`.
5. Choose a local folder.
6. Select **Clone**.
7. After cloning, select **Fetch origin**.
8. If GitHub Desktop offers **Pull origin**, select it.
9. Use **Repository > Open in Visual Studio Code** to open the project.

GitHub Desktop replaces command-line Git operations such as cloning, fetching, pulling, branching, committing, and pushing. Python environment and package installation commands must still be run in a terminal.

---

## 3. Verify the ancillary data

The core model currently expects its required ancillary resources in the repository-level `data/` directory.

The required files include:

```text
data/G0p.txt
data/G0w.txt
data/G1p.txt
data/G1w.txt
data/abs_scat_seawater_20d_35PSU_20230922_short.txt
data/vars_aph_v2.npz
```

Check that the text files contain numerical data and are not unresolved Git LFS pointers.

An unresolved Git LFS pointer starts with text similar to:

```text
version https://git-lfs.github.com/spec/v1
```

The file `vars_aph_v2.npz` should be a binary NumPy archive with a realistic file size. A file of approximately 130 bytes is probably an unresolved Git LFS pointer.

If the files are unresolved pointers, retrieve them with Git LFS:

```bash
git lfs pull
```

Users of GitHub Desktop should verify that Git LFS is installed and restart GitHub Desktop if necessary.

Do not edit, regenerate, or replace the scientific ancillary files without documenting their provenance and validating the resulting model output.

---

## 4. Create a virtual environment

A virtual environment keeps the project dependencies isolated from other Python installations.

### 4.1 Windows PowerShell

From the repository root:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

You may replace `3.10` with another supported version, such as `3.11`, `3.12`, or `3.13`.

If PowerShell blocks the activation script, allow locally created scripts for the current PowerShell session:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1
```

Alternatively, use the virtual environment interpreter directly without activation:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

### 4.2 Linux and macOS

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

---

## 5. Install the project for development

The recommended development installation is editable:

```bash
python -m pip install -e ".[timeseries,dev]"
```

On Windows PowerShell, use the same command:

```powershell
python -m pip install -e ".[timeseries,dev]"
```

Editable installation means that Python imports `rrs3c` from the repository's `src/` directory. Changes made to `src/rrs3c/model.py` are therefore available immediately without rebuilding or reinstalling the package.

The installation groups are defined in `pyproject.toml`.

### Core installation

Install only the core package and its runtime dependencies:

```bash
python -m pip install -e .
```

This installs the core requirements, including:

- NumPy
- SciPy
- lmfit

### Installation with example dependencies

```bash
python -m pip install -e ".[examples]"
```

This additionally installs packages used by examples and plotting, including:

- pandas
- Matplotlib

### Installation with time-series dependencies

```bash
python -m pip install -e ".[timeseries]"
```

This additionally installs:

- pandas
- Matplotlib
- xarray

### Full development installation

```bash
python -m pip install -e ".[timeseries,dev]"
```

This additionally installs development tools such as:

- pytest
- Ruff
- Black
- pre-commit
- build

If `requirements.txt` contains the editable development installation, the equivalent convenience command is:

```bash
python -m pip install -r requirements.txt
```

---

## 6. Verify the installation

Check that the package imports successfully:

```bash
python -c "import rrs3c; print(rrs3c.__file__)"
```

Check the public model class:

```bash
python -c "from rrs3c import rrs_model_3C_O25; print(rrs_model_3C_O25)"
```

Check that the model can load the repository ancillary files:

```bash
python -c "from rrs3c import rrs_model_3C_O25; model = rrs_model_3C_O25(); print(model.data_folder)"
```

The reported data path should point to the repository-level `data/` directory.

If model construction fails, verify:

1. The command is being run from the complete repository checkout.
2. The project is installed in editable mode.
3. The required ancillary files exist.
4. The ancillary files are real data and not Git LFS pointers.
5. The active Python interpreter belongs to the intended virtual environment.

---

## 7. Create a development branch

Do not make substantive changes directly on `main`.

Use one branch for each focused correction or feature.

Suggested branch names include:

```text
fix/input-validation
fix/package-metadata
test/model-regression
docs/install-instructions
feature/model
