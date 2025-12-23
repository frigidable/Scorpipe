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
python -m scorpio_pipe.ui.launcher_app
