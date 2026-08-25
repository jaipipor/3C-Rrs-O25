# Run the time-series example with the project virtual environment
param(
    [string]$DataFolder = "data"
)

& ".\.venv\Scripts\python.exe" ".\examples\run_timeseries.py" `
    --data-folder
