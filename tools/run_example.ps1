\
        # Simple helper to run the example script using the venv
        param(
          [string]$dataFolder = "data"
        )

        & ".\.venv\Scripts\Activate.ps1"
        python examples\run_timeseries.py --data-folder $dataFolder
