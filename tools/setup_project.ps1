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
& $Python -m pip install -e ".[timeseries,notebooks,dev]"

& $Python -m pre_commit install

Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Activate with: .\$VenvName\Scripts\Activate.ps1"
