Param(
  [switch]$SkipInstall,
  [switch]$SkipInstaller
)

$ErrorActionPreference = "Stop"

Write-Host "[Scorpipe] Windows build" -ForegroundColor Cyan

if (-not $SkipInstall) {
  python -m pip install -U pip
  pip install -e ".[gui]" 
  pip install -U pyinstaller
}

Write-Host "[Scorpipe] PyInstaller..." -ForegroundColor Cyan
pyinstaller packaging\windows\scorpipe.spec

if (-not $SkipInstaller) {
  Write-Host "[Scorpipe] Inno Setup (setup.exe)..." -ForegroundColor Cyan
  $iscc = Get-Command iscc.exe -ErrorAction SilentlyContinue
  if (-not $iscc) {
    Write-Host "iscc.exe not found. Install Inno Setup 6 and add iscc.exe to PATH." -ForegroundColor Yellow
    exit 2
  }
  iscc.exe packaging\windows\scorpipe.iss
  Write-Host "Built: packaging\\windows\\Output\\setup.exe" -ForegroundColor Green
}

Write-Host "Done." -ForegroundColor Green
