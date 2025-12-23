Param(
  [switch]$SkipInstall,
  [switch]$SkipInstaller
)

$ErrorActionPreference = "Stop"

function Invoke-Logged {
  param(
    [string]$FilePath,
    [string[]]$ArgumentList,
    [string]$LogPath,
    [string]$StepName
  )

  $logDir = Split-Path -Parent $LogPath
  if ($logDir -and -not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
  }

  Write-Host "[$StepName] $FilePath $($ArgumentList -join ' ')" -ForegroundColor DarkCyan

  # NOTE:
  # Start-Process forbids redirecting stdout and stderr to the same file.
  # This caused GitHub Actions to fail before even starting PyInstaller/ISCC.
  $errLog = "$LogPath.err"

  $p = Start-Process -FilePath $FilePath -ArgumentList $ArgumentList -NoNewWindow -PassThru -Wait `
        -RedirectStandardOutput $LogPath -RedirectStandardError $errLog

  if ($p.ExitCode -ne 0) {
    Write-Host "---- $StepName stdout tail ----" -ForegroundColor Yellow
    if (Test-Path $LogPath) {
      Get-Content $LogPath -Tail 60
    } else {
      Write-Host "(no stdout log produced: $LogPath)" -ForegroundColor DarkYellow
    }

    Write-Host "---- $StepName stderr tail ----" -ForegroundColor Yellow
    if (Test-Path $errLog) {
      Get-Content $errLog -Tail 60
    } else {
      Write-Host "(no stderr log produced: $errLog)" -ForegroundColor DarkYellow
    }

    throw "$StepName failed, exit code: $($p.ExitCode). See logs: $LogPath and $errLog"
  }
}

Write-Host "[Scorpipe] Windows build" -ForegroundColor Cyan

if (-not $SkipInstall) {
  python -m pip install -U pip
  pip install -e ".[gui]"
  python -m pip install -U pyinstaller
}

New-Item -ItemType Directory -Force -Path packaging\windows\Output | Out-Null

Write-Host "[Scorpipe] PyInstaller..." -ForegroundColor Cyan
Invoke-Logged -FilePath "python" -ArgumentList @(
  "-m", "PyInstaller", "packaging\windows\scorpipe.spec"
) -LogPath "packaging\windows\Output\pyinstaller.log" -StepName "PyInstaller"

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
    throw "ISCC.exe not found. Install Inno Setup 6 (e.g. choco install innosetup -y) and retry."
  }

  Invoke-Logged -FilePath $iscc -ArgumentList @(
    "packaging\windows\scorpipe.iss"
  ) -LogPath "packaging\windows\Output\iscc.log" -StepName "ISCC"

  if (-not (Test-Path packaging\windows\Output\setup.exe)) {
    throw "setup.exe was not produced (expected: packaging\\windows\\Output\\setup.exe). Check packaging\\windows\\Output\\iscc.log"
  }

  Write-Host "Built: packaging\\windows\\Output\\setup.exe" -ForegroundColor Green
}

Write-Host "Done." -ForegroundColor Green
