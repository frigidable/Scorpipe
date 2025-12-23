Param(
  [switch]$SkipInstall,
  [switch]$SkipInstaller
)

$ErrorActionPreference = "Stop"

Write-Host "[Scorpipe] Windows build" -ForegroundColor Cyan

if (-not $SkipInstall) {
  python -m pip install -U pip
  pip install -e ".[gui]"
  python -m pip install -U pyinstaller
}

Write-Host "[Scorpipe] PyInstaller..." -ForegroundColor Cyan
pyinstaller packaging\windows\scorpipe.spec
if ($LASTEXITCODE -ne 0) {
  throw "PyInstaller failed, exit code: $LASTEXITCODE"
}

if (-not $SkipInstaller) {
  Write-Host "[Scorpipe] Inno Setup (setup.exe)..." -ForegroundColor Cyan

  $iscc = (Get-Command iscc.exe -ErrorAction SilentlyContinue).Source
  if (-not $iscc) { $iscc = (Get-Command ISCC.exe -ErrorAction SilentlyContinue).Source }

  if (-not $iscc) {
    $candidates = @(
      (Join-Path ${env:ProgramFiles(x86)} 'Inno Setup 6\ISCC.exe'),
      (Join-Path $env:ProgramFiles 'Inno Setup 6\ISCC.exe'),
      'C:\Program Files (x86)\Inno Setup 6\ISCC.exe',
      'C:\Program Files\Inno Setup 6\ISCC.exe'
    ) | Where-Object { $_ -and (Test-Path $_) }

    if ($candidates.Count -gt 0) { $iscc = $candidates[0] }
  }

  if (-not $iscc) {
    Write-Host "ISCC.exe not found. Install Inno Setup 6 (e.g. 'choco install innosetup -y') and retry." -ForegroundColor Yellow
    exit 2
  }

  & $iscc packaging\windows\scorpipe.iss
  if ($LASTEXITCODE -ne 0) {
    throw "Inno Setup (ISCC) failed, exit code: $LASTEXITCODE"
  }
  Write-Host "Built: packaging\\windows\\Output\\setup.exe" -ForegroundColor Green
}

Write-Host "Done." -ForegroundColor Green
