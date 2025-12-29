Param(
  [string]$ProjectRoot = ""
)

$ErrorActionPreference = "Stop"

if (-not $ProjectRoot) {
  $ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\.."))
} else {
  $ProjectRoot = (Resolve-Path $ProjectRoot)
}

$ver = (python -c "from scorpio_pipe.version import __version__; print(__version__)").Trim()

$setup = Join-Path $ProjectRoot "packaging\windows\Output\ScorpioPipe-Setup-x64-$ver.exe"
if (-not (Test-Path $setup)) {
  throw "Installer not found: $setup (run: setup.bat --installer)"
}

$bundleName = "Scorpipe-Windows-x64-$ver"
$bundleRoot = Join-Path $ProjectRoot "release\$bundleName"
$docsDir    = Join-Path $bundleRoot "docs"

if (Test-Path $bundleRoot) { Remove-Item $bundleRoot -Recurse -Force }
New-Item -ItemType Directory -Force -Path $docsDir | Out-Null

Copy-Item $setup (Join-Path $bundleRoot (Split-Path $setup -Leaf)) -Force
Copy-Item (Join-Path $ProjectRoot "INSTALL.md") (Join-Path $bundleRoot "INSTALL.md") -Force

# docs
$docList = @(
  "docs\MANUAL.md",
  "docs\RUNBOOK.md",
  "docs\SETUP_TROUBLESHOOTING.md",
  "docs\GET_SETUP_EXE.md"
)

foreach ($rel in $docList) {
  $src = Join-Path $ProjectRoot $rel
  if (Test-Path $src) {
    Copy-Item $src (Join-Path $docsDir (Split-Path $rel -Leaf)) -Force
  }
}

# checksums
$sumPath = Join-Path $bundleRoot "SHA256SUMS.txt"
"SHA256 checksums for $bundleName" | Out-File -Encoding utf8 $sumPath
"" | Out-File -Append -Encoding utf8 $sumPath

Get-ChildItem -File -Recurse $bundleRoot | ForEach-Object {
  $relPath = $_.FullName.Substring($bundleRoot.Length + 1).Replace("\","/")
  $hash = (Get-FileHash -Algorithm SHA256 $_.FullName).Hash
  "$hash  $relPath" | Out-File -Append -Encoding utf8 $sumPath
}

$zip = Join-Path $ProjectRoot "$bundleName.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }

Compress-Archive -Path $bundleRoot -DestinationPath $zip -Force
Write-Host "Built: $zip" -ForegroundColor Green
