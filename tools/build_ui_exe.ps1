param(
  [string]$Name = "scorpipe",
  [string]$PythonPath = "",
  [string]$ProjectRoot = ""
)

$ErrorActionPreference = "Stop"

function Assert-LastExitOk([string]$step) {
  if ($LASTEXITCODE -ne 0) {
    throw "Step failed ($step), exit code: $LASTEXITCODE"
  }
}

Write-Host "Building GUI .exe: $Name" -ForegroundColor Cyan

$py = $PythonPath
if (-not $py) { $py = "python" }

# IMPORTANT:
# PyInstaller stores --add-data entries inside the generated .spec.
# When the .spec is written to a subfolder (e.g. build\pyinstaller),
# RELATIVE paths inside datas are resolved relative to that folder.
# Поэтому здесь используем абсолютные пути для datas.

if (-not $ProjectRoot) {
  # tools/.. -> project root
  $ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot ".."))
} else {
  $ProjectRoot = (Resolve-Path $ProjectRoot)
}

$resDir = Join-Path $ProjectRoot "src\scorpio_pipe\resources"
if (-not (Test-Path $resDir)) {
  throw "resources folder not found: $resDir"
}

$entry = Join-Path $ProjectRoot "tools\run_ui.py"
if (-not (Test-Path $entry)) {
  throw "entry script not found: $entry"
}

Push-Location $ProjectRoot
try {
  # Ensure pyinstaller exists in the selected interpreter
  & $py -m pip install -U pip
  Assert-LastExitOk "pip upgrade"
  & $py -m pip install -U pyinstaller
  Assert-LastExitOk "pyinstaller install"

  # Clean old
  if (Test-Path "$Name.exe") { Remove-Item -Force "$Name.exe" }
  if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
  if (Test-Path "$Name.spec") { Remove-Item -Force "$Name.spec" }

  # NOTE:
  # - --noconsole: no terminal window
  # - --onefile: single exe
  # - --paths src: so imports work in editable layout
  # - hidden imports for QtPdf (vector) and PyMuPDF (fallback renderer)

  $dataArg = "$resDir;scorpio_pipe\resources"

  & $py -m PyInstaller `
    --noconsole --onefile `
    --name $Name `
    --distpath "." `
    --workpath "build\pyinstaller" `
    --specpath "build\pyinstaller" `
    --paths "src" `
    --add-data $dataArg `
    --hidden-import "PySide6.QtPdf" `
    --hidden-import "PySide6.QtPdfWidgets" `
    --hidden-import "pymupdf" `
    --hidden-import "fitz" `
    --hidden-import "pyqtgraph" `
    $entry

  Assert-LastExitOk "PyInstaller"

  if (-not (Test-Path "$Name.exe")) {
    throw "PyInstaller finished but $Name.exe not found in project root."
  }

  Write-Host "Done. EXE is in .\$Name.exe" -ForegroundColor Green
} finally {
  Pop-Location
}
