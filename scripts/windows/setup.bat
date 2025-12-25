@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ------------------------------------------------------------
REM Scorpipe - one-click setup (Windows)
REM
REM Designed to work even when:
REM - "python" in PATH points to Microsoft Store alias
REM - project path contains spaces
REM - script is launched from any working directory
REM
REM Usage:
REM   setup.bat              -> install/update, build scorpipe.exe, then launch UI
REM   setup.bat --run        -> launch UI (no reinstall, no .exe rebuild)
REM   setup.bat --build      -> build scorpipe.exe only (PyInstaller)
REM   setup.bat --installer  -> build setup.exe (Inno Setup) after building scorpipe.exe
REM ------------------------------------------------------------

set "MODE=%~1"

set "HERE=%~dp0"
for %%I in ("%HERE%..\..") do set "ROOT=%%~fI\"

if not exist "%ROOT%pyproject.toml" (
  echo.
  echo [ERROR] Cannot locate project root from:
  echo   %HERE%
  echo Expected to find: %ROOT%pyproject.toml
  echo.
  goto :FAIL
)

pushd "%ROOT%" >nul

REM Resolve ROOT to an absolute path (important if the project folder has spaces)
set "ROOT=%CD%\"
if errorlevel 1 (
  echo.
  echo [ERROR] Cannot change directory to project root:
  echo   %ROOT%
  echo.
  goto :FAIL
)

set "EXE=%ROOT%scorpipe.exe"

REM Choose Python launcher: prefer 'py -3' when available (avoids Microsoft Store python alias)
set "PY=python"
where py >nul 2>nul
if not errorlevel 1 (
  set "PY=py -3"
)

set "VENV_PY=%ROOT%.venv\Scripts\python.exe"
set "BUILD_PS=%ROOT%tools\build_ui_exe.ps1"
set "BUILD_INSTALLER_PS=%ROOT%packaging\windows\build.ps1"

echo [Scorpipe] Project root: "%ROOT%"

REM Fast path: if the .exe already exists, allow launching without Python.
if /i "%MODE%"=="--run" (
  if exist "%EXE%" (
    echo Launching "%EXE%" ...
    start "" "%EXE%"
    goto :OK
  )
)

REM If Python is missing but the .exe is present, we can still run the UI.
where python >nul 2>nul
if errorlevel 1 (
  where py >nul 2>nul
  if errorlevel 1 (
    if exist "%EXE%" (
      echo Python not found, but scorpipe.exe exists.
      echo Launching "%EXE%" ...
      start "" "%EXE%"
      goto :OK
    )
  )
)

REM Prefer an existing venv interpreter if present.
if exist "%VENV_PY%" (
  set "PY_FOR_VENV=%VENV_PY%"
  goto :HAVE_VENV
)

call :DETECT_PY
if errorlevel 1 goto :FAIL

echo [1/4] Creating virtual environment...
%PY_CMD% -m venv ".venv"
if errorlevel 1 (
  echo.
  echo [ERROR] Failed to create .venv.
  echo If you see a Microsoft Store message, install Python from python.org OR enable the "py" launcher.
  echo.
  goto :FAIL
)

:HAVE_VENV
set "PY_FOR_VENV=%VENV_PY%"
if not exist "%PY_FOR_VENV%" (
  echo.
  echo [ERROR] venv python not found:
  echo   "%PY_FOR_VENV%"
  echo.
  goto :FAIL
)

echo [2/4] Installing / updating Scorpipe (editable + GUI extras)...
"%PY_FOR_VENV%" -m pip install -U pip
if errorlevel 1 goto :FAIL
"%PY_FOR_VENV%" -m pip install -e ".[gui]"
if errorlevel 1 goto :FAIL

if /i "%MODE%"=="--run" goto :RUN_ONLY

echo [3/4] Building scorpipe.exe (PyInstaller)...
if not exist "%BUILD_PS%" (
  echo.
  echo [ERROR] Build script not found:
  echo   "%BUILD_PS%"
  echo.
  goto :FAIL
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%BUILD_PS%" -PythonPath "%PY_FOR_VENV%" -ProjectRoot "%ROOT%"
if errorlevel 1 (
  echo.
  echo [ERROR] EXE build failed.
  echo.
  goto :FAIL
)

if not exist "%EXE%" (
  echo.
  echo [ERROR] scorpipe.exe was not created in project root:
  echo   "%EXE%"
  echo.
  goto :FAIL
)

echo Built: "%EXE%"
if /i "%MODE%"=="--build" goto :OK

if /i "%MODE%"=="--installer" (
  echo [4/4] Building setup.exe (Inno Setup)...
  if not exist "%BUILD_INSTALLER_PS%" (
    echo.
    echo [ERROR] Installer build script not found:
    echo   "%BUILD_INSTALLER_PS%"
    echo.
    goto :FAIL
  )
  powershell -NoProfile -ExecutionPolicy Bypass -File "%BUILD_INSTALLER_PS%" -SkipInstall
  if errorlevel 1 (
    echo.
    echo [ERROR] Installer build failed.
    echo.
    goto :FAIL
  )
  if not exist "%ROOT%packaging\windows\Output\setup.exe" (
    echo.
    echo [ERROR] setup.exe was not created:
    echo   "%ROOT%packaging\windows\Output\setup.exe"
    echo.
    goto :FAIL
  )
  echo Built: "%ROOT%packaging\windows\Output\setup.exe"
  goto :OK
)

:RUN_ONLY
echo [4/4] Launching UI...
if exist "%EXE%" (
  start "" "%EXE%"
) else (
  "%PY_FOR_VENV%" -m scorpio_pipe.ui.launcher_app
)

goto :OK

:DETECT_PY
set "PY_CMD="

where py >nul 2>nul
if not errorlevel 1 (
  set "PY_CMD=py -3"
  %PY_CMD% -c "import sys; print(sys.version)" >nul 2>nul
  if not errorlevel 1 exit /b 0
)

where python >nul 2>nul
if errorlevel 1 (
  echo.
  echo [ERROR] Python not found.
  echo Install Python 3.10+ from python.org and tick "Add python.exe to PATH".
  echo.
  exit /b 1
)

for /f "delims=" %%P in ('where python') do (
  set "PY_PATH=%%P"
  goto :CHECK_PY
)

:CHECK_PY
echo "%PY_PATH%" | findstr /i "WindowsApps" >nul
if not errorlevel 1 (
  echo.
  echo [ERROR] Your 'python' points to Microsoft Store alias:
  echo   "%PY_PATH%"
  echo Install Python from python.org (recommended) OR enable the 'py' launcher.
  echo.
  exit /b 1
)

set "PY_CMD=python"
python -c "import sys; print(sys.version)" >nul 2>nul
if errorlevel 1 (
  echo.
  echo [ERROR] 'python' exists but cannot run.
  echo.
  exit /b 1
)
exit /b 0

:FAIL
echo.
echo Setup failed.
popd >nul 2>nul
pause
exit /b 1

:OK
popd >nul 2>nul
exit /b 0
