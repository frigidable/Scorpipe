@echo off
setlocal EnableExtensions
set "HERE=%~dp0"
set "ROOT=%HERE%..\..\"
pushd "%ROOT%" >nul

REM Prefer the built one-file .exe (created by setup.bat)
if exist "scorpipe.exe" (
  start "" "scorpipe.exe"
  exit /b 0
)

if not exist ".venv\Scripts\scorpipe.exe" (
  echo [ERROR] UI launcher not found.
  echo Run setup.bat first.
  pause
  exit /b 1
)

start "" ".venv\Scripts\scorpipe.exe"
