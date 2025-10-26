\
        <#
        tools\setup_project.ps1

        PowerShell script to set up the development environment on Windows:
        - creates a virtual environment `.venv`
        - installs requirements
        - installs Git LFS and tracks `data/*`
        - installs pre-commit and registers hooks

        Run in PowerShell from the repository root:
            .\tools\setup_project.ps1
        #>

        param(
            [string]$venvName = ".venv"
        )

        # create virtual environment
        python -m venv $venvName

        # activate venv (PowerShell)
        & "$PWD\$venvName\Scripts\Activate.ps1"

        # upgrade pip and install requirements
        python -m pip install --upgrade pip
        pip install -r requirements.txt

        # git lfs installation (if git-lfs is available on PATH)
        if (Get-Command git-lfs -ErrorAction SilentlyContinue) {
            git lfs install
            git lfs track "data/*"
            git add .gitattributes
            try { git commit -m "chore: track data with git-lfs" --no-verify } catch { }
        } else {
            Write-Host "git-lfs not found on PATH. Install Git LFS if you plan to track large data files." -ForegroundColor Yellow
        }

        # install pre-commit (and register hooks)
        pip install pre-commit
        pre-commit install

        Write-Host "Setup complete. Activate venv with: & \"$PWD\\$venvName\\Scripts\\Activate.ps1\"" -ForegroundColor Green
