# Changelog

All notable changes to this project will be documented in this file.

The format is loosely based on “Keep a Changelog”:
https://keepachangelog.com (human-readable, release-oriented)

## [Unreleased]

- Initial repository, no version number
- Core model implementation in `src/rrs3c/model.py`.
- Example time-series processing script in `examples/timeseries/...`.
- Demo data: `examples/timeseries/example_time_series_data.csv`.
- Jupyter notebooks in `notebooks/` (QuickStart, TimeSeries, Parameter analysis).
- Pre-commit hooks (black, ruff, isort) configured.

---

## [0.1.0] - 2025-11-24
Initial public release.

### Added
- This CHANGELOG.md file
- A single spectrum processing example at eexamples/data/example_time_series.csv,
  used when calling model.py without arguments

### Fixed
- Minor I/O robustness improvements in src/rrs3/model.py and examples/timeseries/src/run_timeseries.py
- Improved model.py commenting

### Changed
- A few variable remanings

### Removed
- A few unnecessary defensive conditions here and there in model.py

## [0.2.0] - 2025-11-24
Minor build/test fixes.

### Fixed
- Fixed the /tests/test_model_import.py code for proper build and import

---

<!--
  How to use:
  - Keep top section "Unreleased" for changes not yet in a released tag.
  - When you cut a release:
      1) replace "Unreleased" with a version heading and date (e.g. [0.2.0] - 2025-11-24)
      2) add a new empty "Unreleased" header above it for future changes.
  - Follow semantic versioning informally:
      - bump MAJOR for incompatible API changes,
      - bump MINOR for new functionality (backwards-compatible),
      - bump PATCH for bug fixes/backwards-compatible changes.
-->
