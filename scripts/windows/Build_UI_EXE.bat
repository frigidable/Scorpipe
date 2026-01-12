@echo off
setlocal EnableExtensions
set "HERE=%~dp0"
set "ROOT=%HERE%..\..\"
pushd "%ROOT%" >nul

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Virtual environment not found.
  echo Run setup.bat first.
  pause
  exit /b 1
)

call ".venv\Scripts\activate.bat"

REM Build onefile GUI EXE in the project root:
powershell -NoProfile -ExecutionPolicy Bypass -File "tools\build_ui_exe.ps1" -ProjectRoot "%ROOT%"
pause
