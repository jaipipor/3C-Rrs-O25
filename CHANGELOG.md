# Changelog

All notable changes to 3C-Rrs-O25 are documented in this file.

The format is based on Keep a Changelog,
and the project follows https://semver.org/ where practical.

## [Unreleased]

### Planned

- Expand automated coverage of the forward model and inversion workflow.
- Improve validation of model inputs and ancillary data.
- Continue reviewing numerical robustness and scientific boundary conditions.

## [1.0.2] - 2026-08-25

### Fixed

- Corrected package discovery so built wheels include the `rrs3c` Python package.
- Corrected runtime dependency metadata in built distributions.
- Removed conflicting package metadata previously maintained in `setup.cfg`.
- Corrected the `run_3C.py` module docstring to avoid invalid escape-sequence warnings on Windows.
- Enabled inclusion of Git LFS objects in GitHub-generated source archives.

### Changed

- Consolidated package metadata, dependency declarations, and package discovery in `pyproject.toml`.
- Separated core, command-line, time-series, and development dependencies.
- Updated the development installation through `requirements.txt`.
- Simplified and reorganized `README.md`.
- Standardized references to the command-line runner as `scripts/run_3C.py`.
- Clarified that the complete repository includes the ancillary model files stored in `data/`.

### Removed

- Removed the obsolete `setup.cfg` packaging configuration.

### Verified

- Confirmed that the wheel contains `rrs3c/__init__.py` and `rrs3c/model.py`.
- Confirmed installation of the wheel in a clean Python 3.13 virtual environment.
- Confirmed automatic installation of NumPy, SciPy, and `lmfit`.
- Confirmed that `pip check` reports no broken requirements.
- Confirmed that `twine check` passes for the built distribution.
- Confirmed that GitHub source archives contain the actual ancillary data instead of Git LFS pointer files.
- Confirmed that the automated test suite passes.
- Confirmed successful end-to-end execution of `scripts/run_3C.py` using the example spectrum.

### Notes

- The ancillary model files remain in the repository-level `data/` directory.
- The recommended workflow is to obtain the complete repository, create a virtual environment, and install the project in editable mode.
- The scientific formulation of the 3C-O25 model has not been intentionally changed in this release.

## [1.0.1] - 2026-08-24

### Changed

- Improved and clarified the project documentation.
- Corrected references to the `run_3C.py` command-line runner.

### Notes

- This release contains no substantive changes to the scientific model relative to version 1.0.0.

## [1.0.0] - 2026-04-27

### Added

- First versioned public release of 3C-Rrs-O25.
- Added the analytical 3C-O25 forward model in `src/rrs3c/model.py`.
- Added `lmfit`-based inversion through `fit_LtEs(...)`.
- Added ancillary water-optics, phytoplankton-absorption, and G-function data.
- Added a command-line workflow for processing a single spectrum.
- Added example spectra, parameter configurations, and Jupyter notebooks.
- Added caching of wavelength-dependent model calculations.
- Added initial automated tests and continuous-integration configuration.
- Added project documentation, citation metadata, and GPL licensing information.

## [0.2.0] - 2025-11-24

### Changed

- Replaced selected string-based paths with `pathlib.Path` objects.

### Fixed

- Corrected `tests/test_model_import.py` to support package import and model construction during testing.

## [0.1.0] - 2025-11-24

### Added

- Added the initial changelog.
- Added a single-spectrum example dataset.
- Added the initial executable example in `src/rrs3c/model.py`.

### Changed

- Improved comments and documentation in `src/rrs3c/model.py`.
- Renamed selected variables for clarity.

### Fixed

- Improved input and output handling in the core model and time-series example.

### Removed

- Removed redundant defensive conditions from the core model.
