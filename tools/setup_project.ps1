<#
Set up the Windows development environment.

Run from the repository root:

    .\tools\setup_project.ps1
#>

param(
    [string]$VenvName = ".venv"
)

$ErrorActionPreference = "Stop"
$Python = ".\$VenvName\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    python -m venv $VenvName
}

& $Python -m pip install --upgrade pip
& $Python -m pip install -e ".[examples,timeseries,dev]"

if (Get-Command git-lfs -ErrorAction SilentlyContinue) {
    git lfs install
    git lfs pull
} else {
    Write-Host "Git LFS not found. Install it to retrieve the model data." -ForegroundColor Yellow
}

& $Python -m pre_commit install

Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Activate with: .\$VenvName\Scripts\Activate.ps1"
