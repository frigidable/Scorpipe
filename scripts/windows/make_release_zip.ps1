Param(
  [string]$ProjectRoot = ""
)

$ErrorActionPreference = "Stop"

if (-not $ProjectRoot) {
  $ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\.."))
} else {
  $ProjectRoot = (Resolve-Path $ProjectRoot)
}

$setup = Join-Path $ProjectRoot "packaging\windows\Output\setup.exe"
if (-not (Test-Path $setup)) {
  throw "setup.exe not found: $setup (run: setup.bat --installer)"
}

$releaseDir = Join-Path $ProjectRoot "release"
New-Item -ItemType Directory -Force -Path $releaseDir | Out-Null

Copy-Item $setup (Join-Path $releaseDir "setup.exe") -Force
Copy-Item (Join-Path $ProjectRoot "INSTALL.md") (Join-Path $releaseDir "INSTALL.md") -Force

$zip = Join-Path $ProjectRoot "Scorpipe-Windows-x64.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }

Compress-Archive -Path (Join-Path $releaseDir "*") -DestinationPath $zip -Force
Write-Host "Built: $zip" -ForegroundColor Green
