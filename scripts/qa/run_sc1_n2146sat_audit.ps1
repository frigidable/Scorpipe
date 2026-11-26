<#
.SYNOPSIS
  Run header and native RAW pixel QA on an existing N2146sat workspace.

.DESCRIPTION
  Strictly read-only with respect to observed RAW, saved config and existing
  stage products. Writes two new JSON reports outside RAW/work.
  This is NOT first-build authorization or scientific calibration.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $ConfigPath,

    [Parameter(Mandatory = $true)]
    [string] $ReportDir,

    [string] $PythonExe = "python"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
$config = (Resolve-Path -LiteralPath $ConfigPath).Path

$previousPythonPath = [Environment]::GetEnvironmentVariable("PYTHONPATH", "Process")
$previousNoBytecode = [Environment]::GetEnvironmentVariable("PYTHONDONTWRITEBYTECODE", "Process")

try {
    $env:PYTHONPATH = "$root\src;$root\scripts\qa"
    if ($previousPythonPath) {
        $env:PYTHONPATH += ";$previousPythonPath"
    }
    $env:PYTHONDONTWRITEBYTECODE = "1"

    # Feed Python source through stdin. Windows PowerShell 5.1 can mangle
    # nested quoting in python -c when invoking a native Windows program.
    $probe = @'
import numpy
import astropy
import yaml
from scorpio_pipe.frame_signature import FrameSignature
print("PYTHON DEPENDENCIES OK")
'@
    $probe | & $PythonExe -
    if ($LASTEXITCODE -ne 0) {
        throw "Python dependencies unavailable in selected interpreter: $PythonExe"
    }

    $stamp = Get-Date -Format "yyyyMMdd_HHmmss_fff"
    $preflight = Join-Path $ReportDir "$($stamp)_header_existing.json"
    $inventory = Join-Path $ReportDir "$($stamp)_raw_inventory.json"

    Write-Host "=== 1. EXISTING WORKSPACE HEADER AUDIT (READ-ONLY) ==="
    & $PythonExe (Join-Path $root "scripts\qa\sc1_first_masterbias_gate.py") --audit-existing $config $preflight
    if ($LASTEXITCODE -ne 0) {
        throw "Header audit failed; RAW pixel inventory was NOT started."
    }

    Write-Host "=== 2. NATIVE RAW PIXEL INVENTORY (READ-ONLY) ==="
    & $PythonExe (Join-Path $root "scripts\qa\sc1_raw_pixel_readability_qa.py") --inspect $config $inventory
    if ($LASTEXITCODE -ne 0) {
        throw "RAW pixel inventory failed; inspect errors above."
    }

    $header = Get-Content -LiteralPath $preflight -Raw -Encoding UTF8 |
        ConvertFrom-Json
    $result = Get-Content -LiteralPath $inventory -Raw -Encoding UTF8 |
        ConvertFrom-Json

    if ($header.mode -ne "audit_existing_readonly" -or
        $header.stage_execution_authorized -ne $false -or
        $result.all_selected_primary_arrays_readable_and_finite -ne $true -or
        $result.input_files.Count -ne 32) {
        throw "Unexpected output schema or observation counts; do not certify."
    }

    Write-Host "=== READ-ONLY RAW AUDIT COMPLETED ===" -ForegroundColor Green
    Write-Host "Header mode: $($header.mode)"
    Write-Host "Existing stage products: $($header.existing_stage_products.Count)"
    Write-Host "Frames: $($result.input_files.Count)"
    Write-Host "Manifest SHA-256: $($result.input_manifest_sha256)"
    Write-Host "Header report: $preflight"
    Write-Host "Pixel report: $inventory"
    Write-Host "Scientific E2E qualification: HOLD"
}
finally {
    [Environment]::SetEnvironmentVariable("PYTHONPATH", $previousPythonPath, "Process")
    [Environment]::SetEnvironmentVariable("PYTHONDONTWRITEBYTECODE", $previousNoBytecode, "Process")
}
