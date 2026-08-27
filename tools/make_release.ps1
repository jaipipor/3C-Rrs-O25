param(
    [string]$Version = "1.0.3"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Virtual environment not found. Create .venv and install the development dependencies first."
}

Set-Location $RepoRoot

Write-Host "Cleaning previous build artifacts..." -ForegroundColor Cyan

Remove-Item -LiteralPath ".\build" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath ".\dist" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath ".\src\rrs3c.egg-info" -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "Building rrs3c $Version..." -ForegroundColor Cyan

& $Python -m build

$Wheel = ".\dist\rrs3c-$Version-py3-none-any.whl"
$Source = ".\dist\rrs3c-$Version.tar.gz"

if (-not (Test-Path -LiteralPath $Wheel)) {
    throw "Expected wheel not found: $Wheel"
}

if (-not (Test-Path -LiteralPath $Source)) {
    throw "Expected source distribution not found: $Source"
}

Write-Host "Checking distribution metadata..." -ForegroundColor Cyan

& $Python -m twine check $Wheel $Source

Write-Host ""
Write-Host "Release files created successfully:" -ForegroundColor Green

Get-Item -LiteralPath $Wheel, $Source |
    Select-Object Name, Length, LastWriteTime
