param(
  [string]$Name = "ScorpioPipeUI"
)

$ErrorActionPreference = "Stop"

Write-Host "Building GUI .exe: $Name" -ForegroundColor Cyan

# Ensure pyinstaller exists in the active venv
python -m pip install -U pip
python -m pip install -U pyinstaller

# Clean old
if (Test-Path "$Name.exe") { Remove-Item -Force "$Name.exe" }
if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
if (Test-Path "$Name.spec") { Remove-Item -Force "$Name.spec" }

# NOTE:
# - --noconsole: no terminal window
# - --onefile: single exe
# - --paths src: so imports work in editable layout
# - hidden imports for QtPdf (vector) and PyMuPDF (fallback renderer)

pyinstaller `
  --noconsole --onefile `
  --name $Name `
  --distpath "." `
  --workpath "build\pyinstaller" `
  --specpath "build\pyinstaller" `
  --paths "src" `
  --hidden-import "PySide6.QtPdf" `
  --hidden-import "PySide6.QtPdfWidgets" `
  --hidden-import "pymupdf" `
  "tools\run_ui.py"

Write-Host "Done. EXE is in .\$Name.exe" -ForegroundColor Green
