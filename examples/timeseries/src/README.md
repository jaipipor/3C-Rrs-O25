# README for `/examples/timeseries/src`

This directory contains a runnable example demonstrating how to use the **3C Remote Sensing Reflectance (Rrs)** analytical model implemented in the `rrs3c` package.

---

## 📁 Directory structure

```
/examples/timeseries/src/
│
├── run_timeseries.py     # Main script to run the time-series example
├── utils.py              # Helper routines (solar geometry, data flags, etc.)
└── debug_import_cli.py   # Optional debugging script for import issues
```

---

## 🚀 Purpose

`run_timeseries.py` reads above-water spectral measurements (e.g. from a fixed station), processes them using the 3C model implemented in `src/rrs3c/model.py`, and optionally produces plots and NetCDF outputs.

It demonstrates how to:

* Import and call the analytical 3C model (`rrs_model_3C`).
* Handle time-series of spectral data.
* Use solar geometry and quality flags (`utils.py`).
* Write outputs (NetCDF, PNG diagnostics).

---

## 🧭 How to run

You can execute the script directly without installing the package, thanks to a built‑in `sys.path` adjustment that finds the `src/` folder relative to this script.

### **Windows PowerShell**

```powershell
cd C:\Users\Jaime\Documents\GitHub\3C-Rrs-O25
& .\.venv\Scripts\Activate.ps1
cd examples\timeseries\src
python run_timeseries.py --input-file ..\example_time_series_data.csv --output-folder ..\output --plot --verbose
```

### **macOS / Linux**

```bash
cd /path/to/3C-Rrs-O25
source .venv/bin/activate
cd examples/timeseries/src
python run_timeseries.py --input-file ../example_time_series_data.csv --output-folder ../output --plot --verbose
```

If the script cannot find the module `rrs3c`, ensure you are running it from within the example directory, or set:

```bash
export PYTHONPATH=$(pwd)/src
```

---


## 🧩 Example command-line interface

```
usage: run_timeseries.py [-h] [--input-file INPUT_FILE] [--input-folder INPUT_FOLDER]
                         [--output-folder OUTPUT_FOLDER] [--date DATE] [--plot] [--verbose]
```

Options:

* `--input-file` : CSV with spectral time series (default: `example_time_series_data.csv`)
* `--input-folder` : Folder with the input file (default: script folder)
* `--output-folder` : Folder for results (default: `../output`)
* `--plot` : Save PNG diagnostics
* `--verbose` : Enable detailed logging

---

**End of README — /examples/timeseries/src**
