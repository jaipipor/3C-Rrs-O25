# Time-series example 📈

This example applies the **3C-O25 model** to a sequence of above-water radiometric measurements.

The workflow:

1. reads the time-series radiometric data
2. selects and interpolates the required spectra
3. calculates the observation geometry
4. fits the 3C-O25 model
5. derives the remote-sensing reflectance
6. saves the processed results

## 📁 Contents

```text
examples/timeseries/
├── data/                   # Example time-series input
├── output/                 # Generated results
└── src/
    ├── run_timeseries.py   # Time-series processing workflow
    └── utils.py            # Input, geometry, and quality-control utilities
```

## ⚡ Running the example

Run the script from the **repository root**:

```powershell
python ".\examples\timeseries\src\run_timeseries.py" --help
```

The help message lists the available input, output, date-selection, and plotting options.

A typical PowerShell command is:

```powershell
python ".\examples\timeseries\src\run_timeseries.py" `
    --input-folder ".\examples\timeseries\data" `
    --output-folder ".\examples\timeseries\output" `
    --date 20200530
```

Add the plotting option if an interactive diagnostic plot is wanted:

```text
--plot
```

Use the exact option names displayed by `--help`.

## 📊 Output

The processed results are written to:

```text
examples/timeseries/output/
```

The workflow stores the retrieved reflectance spectra and associated processing information in a NetCDF file.

If plotting is enabled, diagnostic figures are also written to the output directory.

## 🧭 Processing notes

- The script processes the available east- and west-looking radiometric observations.
- The observation geometry is calculated for each measurement time.
- Measurements that do not satisfy the processing or quality conditions are skipped.
- The model is imported as:

```python
from rrs3c.model import rrs_model_3C_O25
```

- The required ancillary model resources are read from the repository-level `data/` directory.

## 💡 Adapting the workflow

To process another dataset, check the following parts of the example:

- input file structure and metadata
- instrument and variable names
- wavelength range and interpolation grid
- station coordinates
- viewing geometry
- quality-control criteria
- model parameters and parameter bounds
- output filename and destination

The helper functions used for input parsing, solar geometry, and quality checks are located in:

```text
examples/timeseries/src/utils.py
```

## 📚 Further information

See the [main project README](../../README.md) for installation instructions, API usage, testing,information.
