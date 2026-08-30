# Changelog

All notable changes to 3C-Rrs-O25 are documented in this file.

The format is based on Keep a Changelog,
and the project follows https://semver.org/ where practical.

## [1.0.4] - 2026-08-29

### Added

- Added automated testing on Linux and Windows with Python 3.10,
  3.11, 3.12, and 3.13.
- Added dedicated tests for:
  - public package import and model construction
  - model input validation
  - ancillary scientific data integrity
  - G-function lookup-table loading and interpolation
  - single-spectrum numerical regression
- Added an optional `notebooks` dependency group containing the packages
  required for interactive and automated notebook execution.
- Added a named internal constant for the negative modeled-glint penalty.

### Changed

- Expanded the continuous-integration workflow with:
  - dependency consistency checks
  - Python compilation checks
  - Ruff linting
  - Black formatting validation
  - automated tests
  - a single-spectrum smoke test
  - source-distribution and wheel building
- Improved model input handling by consistently converting spectral inputs
  to floating-point NumPy arrays.
- Reordered model validation so dimensions, shapes, finite values,
  wavelength ordering, and weights are checked before numerical operations.
- Added explicit validation that wavelengths are strictly increasing.
- Added explicit validation of solar zenith, viewing zenith, and relative
  azimuth against the domain of the G-function lookup tables.
- Added explicit validation that all required fitting parameters are present.
- Improved the loading of the pure-water optical-property table by:
  - checking its three-column structure
  - removing the final row only when it matches the documented sentinel
  - checking for finite values
  - checking for a strictly increasing wavelength coordinate
- Renamed internal G-function loaders and interpolators to identify their
  corresponding water and particle coefficients more clearly.
- Loaded the phytoplankton absorption archive using a context manager.
- Applied consistent Black formatting to Python files and notebook code cells.
- Removed the duplicated executable example, plotting, and profiling code
  from `src/rrs3c/model.py`; the maintained single-spectrum interfaces are
  now `scripts/run_3C.py` and the quick-start notebook.

### Fixed

- Fixed support for array-like model inputs, including Python lists used
  for spectral weights.
- Prevented the square root of invalid negative weights from being evaluated
  before input validation.
- Removed the runtime warning previously produced while testing negative
  spectral weights.
- Corrected internal handling and reporting of invalid model geometry.
- Corrected several stale or inconsistent notebook checks related to the
  time-series wavelength grid.
- Replaced fixed assumptions

### Packaging

- Included the required ancillary scientific resources in the wheel while
  retaining their authoritative source files in the repository-level
  `data/` directory.
- Made the default `rrs_model_3C_O25()` constructor operational after a
  wheel-only installation, without requiring a repository checkout.
- Preserved the explicit `data_folder` argument for selecting an alternative
  or independently managed ancillary dataset.
- Added clean-environment validation that installs the wheel, constructs the
  model outside the repository, and loads the packaged ancillary resources.
- Excluded demonstration datasets from the installed package because they
  are not required by the core model.

## [1.0.3] - 2026-08-27

### Improved

- Improved the Windows setup, example, and release helper scripts.
- Added comprehensive documentation for the supplied examples.
- Rebuilt the quick-start and time-series notebooks with consistent instructions for Visual Studio Code and Jupyter Notebook.
- Corrected the handling of wavelength metadata in the supplied time-series example.
- Extended the time-series example grid to 350–940 nm at 1 nm resolution.
- Improved validation and interpretation of the example NetCDF output.

### Removed

- Removed Git LFS tracking and stored the ancillary model resources directly in Git.
- Removed obsolete debugging and repository-initialization helpers.

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
