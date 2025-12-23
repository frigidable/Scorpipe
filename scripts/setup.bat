@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ------------------------------------------------------------
REM Scorpio Pipe - one-click setup (install/update + build .exe + run)
REM
REM Usage:
REM   setup.bat            -> install/update, build scorpipe.exe, then launch UI
REM   setup.bat --run      -> launch UI (no reinstall, no .exe rebuild)
REM   setup.bat --build    -> build scorpipe.exe only (PyInstaller)
REM ------------------------------------------------------------

set "HERE=%~dp0"
set "ROOT=%HERE%..\..\"
pushd "%ROOT%" >nul

set "MODE=%~1"
set "EXE=%ROOT%scorpipe.exe"

REM Fast path: if the .exe already exists, allow launching without Python.
if /i "%MODE%"=="--run" (
  if exist "%EXE%" (
    echo Launching %EXE% ...
    start "" "%EXE%"
    exit /b 0
  )
)

REM If Python is missing but the .exe is present, we can still run the UI.
where python >nul 2>nul
if errorlevel 1 (
  if exist "%EXE%" (
    echo Python not found, but scorpipe.exe exists.
    echo Launching %EXE% ...
    start "" "%EXE%"
    exit /b 0
  )
)

REM Otherwise, we need Python for install/build.
where python >nul 2>nul
if errorlevel 1 (
  echo.
  echo [ERROR] Python not found in PATH.
  echo Install Python 3.10+ and tick "Add python.exe to PATH", then run this file again.
  echo.
  pause
  exit /b 1
)

if /i "%MODE%"=="--build" goto BUILD

REM --- Create/activate venv ---
if not exist ".venv\Scripts\python.exe" (
  echo [1/3] Creating virtual environment...
  python -m venv .venv
)

call ".venv\Scripts\activate.bat"

if /i "%MODE%"=="--run" goto RUN

echo [2/4] Installing / updating Scorpio Pipe...
python -m pip install --upgrade pip >nul
python -m pip install -e .

echo [3/4] Building scorpipe.exe...
python -m pip install pyinstaller >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "tools\build_ui_exe.ps1"

echo.
echo Built: scorpipe.exe (project root)

:RUN
echo.
echo [4/4] Launching UI...
if exist "%EXE%" (
  start "" "%EXE%"
) else (
  python -m scorpio_pipe.ui.launcher_app
)

echo.
echo Done.
exit /b 0

:BUILD
echo.
echo Building scorpipe.exe (this may take a while)...

REM Ensure venv exists
if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  python -m venv .venv
)
call ".venv\Scripts\activate.bat"

python -m pip install --upgrade pip >nul
python -m pip install -e .
python -m pip install pyinstaller >nul

powershell -NoProfile -ExecutionPolicy Bypass -File "tools\build_ui_exe.ps1"
echo.
pause
exit /b 0
