# Time-series example

This directory contains a runnable example showing how to apply the **3C-O25 Remote Sensing Reflectance model** to a time series of above-water radiometric measurements.

The example uses the model implemented in:

```text
src/rrs3c/model.py
```

The current model class is:

```python
rrs_model_3C_O25
```

## 📁 Directory structure

```text
examples/
└── timeseries/
    ├── data/
    │   ├── example_time_series_data.csv
    │   └── example_time_series_data_long.csv
    ├── output/
    └── src/
        ├── __init__.py
        ├── README.md
        ├── run_timeseries.py
        └── utils.py
```

The files and directories have the following purposes:

- `run_timeseries.py` runs the time-series processing workflow.
- `utils.py` provides data-loading, solar-geometry, and quality-flag helpers.
- `../data/` contains the example time-series input files.
- `../output/` is the default destination for generated results.

## 🚀 Purpose

`run_timeseries.py` reads above-water spectral measurements, groups the observations by acquisition time, and processes the available radiometric measurements using `rrs_model_3C_O25`.

The example demonstrates how to:

- import and instantiate `rrs_model_3C_O25`
- load a sequence of spectral radiometric measurements
- combine irradiance, sky-radiance, and surface-radiance observations
- calculate solar geometry
- apply basic data-quality flags
- fit the 3C-O25 model repeatedly
- retrieve modeled remote-sensing reflectance, `Rrs`
- retrieve the modeled surface-reflection contribution, `Rg`
- save results as NetCDF files
- optionally generate diagnostic plots

The script represents an example research workflow. Users should review the geometry, quality controls, parameter values, and assumptions before applying the workflow to other instruments or datasets.

## ⚙️ Installation

Run the example from a complete repository checkout.

From the repository root, create a virtual environment and install the project with the time-series dependencies.

### Windows PowerShell

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e ".[timeseries]"
```

A different supported Python version, such as Python 3.11, 3.12, or 3.13, may be used.

If PowerShell blocks the activation script:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1
```

Alternatively, use the virtual-environment interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[timeseries]"
```

### Linux and macOS

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[timeseries]"
```

Editable installation is recommended because the example uses the source package and the repository-level ancillary model data.

## 🧭 Run from the repository root

Running from the repository root keeps all paths explicit and is the recommended method.

### Windows PowerShell

```powershell
python ".\examples\timeseries\src\run_timeseries.py" `
    --input-file "example_time_series_data.csv" `
    --input-folder ".\examples\timeseries\data" `
    --output-folder ".\examples\timeseries\output" `
    --verbose
```

To generate diagnostic plots:

```powershell
python ".\examples\timeseries\src\run_timeseries.py" `
    --input-file "example_time_series_data.csv" `
    --input-folder ".\examples\timeseries\data" `
    --output-folder ".\examples\timeseries\output" `
    --plot `
    --verbose
```

### Linux and macOS

```bash
python examples/timeseries/src/run_timeseries.py \
    --input-file example_time_series_data.csv \
    --input-folder examples/timeseries/data \
    --output-folder examples/timeseries/output \
    --verbose
```

To generate diagnostic plots:

```bash
python examples/timeseries/src/run_timeseries.py \
    --input-file example_time_series_data.csv \
    --input-folder examples/timeseries/data \
    --output-folder examples/timeseries/output \
    --plot \
    --verbose
```

## 📂 Run from the script directory

The example may also be run from:

```text
examples/timeseries/src/
```

### Windows PowerShell

```powershell
Set-Location ".\examples\timeseries\src"

python ".\run_timeseries.py" `
    --input-file "example_time_series_data.csv" `
    --input-folder "..\data" `
    --output-folder "..\output" `
    --verbose
```

### Linux and macOS

```bash
cd examples/timeseries/src

python run_timeseries.py \
    --input-file example_time_series_data.csv \
    --input-folder ../data \
    --output-folder ../output \
    --verbose
```

The input file is located under `../data`, not directly under the parent `timeseries` directory.

## 🧩 Command-line interface

Display the authoritative command-line help with:

```bash
python examples/timeseries/src/run_timeseries.py --help
```

On Windows PowerShell:

```powershell
python ".\examples\timeseries\src\run_timeseries.py" --help
```

The interface is expected to provide options similar to:

```text
usage: run_timeseries.py [-h]
                         [--input-file INPUT_FILE]
                         [--input-folder INPUT_FOLDER]
                         [--output-folder OUTPUT_FOLDER]
                         [--date DATE]
                         [--plot]
                         [--verbose]
```

### Options

- `--input-file` specifies the CSV filename.
- `--input-folder` specifies the directory containing the input CSV.
- `--output-folder` specifies the directory used for NetCDF and plot outputs.
- `--date` supplies an optional date label for output products.
- `--plot` enables generation of diagnostic PNG plots.
- `--verbose` enables detailed logging.

The output produced by `--help` should be treated as authoritative if the command-line interface changes.

## 📥 Input data

The default short example is:

```text
examples/timeseries/data/example_time_series_data.csv
```

A longer example may also be
