# 3C-Rrs-O25

Analytical 3C remote-sensing reflectance model (O25 variant).

This repository contains the optimized core model, example wrappers, small
scaffolding scripts and CI configuration to reproduce and run the model on
example data.

## Quickstart (Windows)

1. Clone / create the repo locally (GitHub Desktop recommended).
2. Copy the scaffold files into the repo (or download them from this canvas).
3. Open **PowerShell** in the repository root and run:

```powershell
# create venv and install deps, configure git-lfs and pre-commit
.\tools\setup_project.ps1
```

4. Activate the virtual environment (if not automatically active):
```powershell
& ".\.venv\Scripts\Activate.ps1"
```

5. Run the example (adjust `data_folder` inside `examples/run_timeseries.py` if needed):
```powershell
python examples\run_timeseries.py --data-folder data
```

## Contents

- `src/rrs3c/model.py` — core `rrs_model_3C` implementation (exactly as provided).
- `examples/run_timeseries.py` — wrapper script to run the model on example data.
- `data/` — small demo data and instructions (do not commit large ancillary files).
- `tools/` — PowerShell automation scripts to set up the project and make releases.
- `.github/workflows/ci.yml` — CI pipeline (lint + tests).
- `.pre-commit-config.yaml` — pre-commit hooks (black, ruff, isort).

## Data & ancillary files

Large files (G-tables, vars_aph_v2.npz, full IOP data) must be downloaded separately.
See `data/README.md` for exact filenames and checksums.

## License

GPL-3.0 (see LICENSE)
