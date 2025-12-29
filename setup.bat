@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ------------------------------------------------------------
REM Scorpipe developer helper (project root)
REM
REM Creates/updates .venv, installs editable deps, and optionally:
REM   --run        run GUI (default)
REM   --build      build onefile GUI EXE (tools/build_ui_exe.ps1)
REM   --installer  build Windows installer (packaging/windows/build.ps1)
REM
REM Tip: if the window closes instantly, use setup_debug.bat (writes setup_debug.log).
REM ------------------------------------------------------------

cd /d "%~dp0"

set "DO_RUN=1"
set "DO_BUILD=0"
set "DO_INSTALLER=0"

:parse
if "%~1"=="" goto :parsed
if /i "%~1"=="--run"       (set "DO_RUN=1" & shift & goto :parse)
if /i "%~1"=="--no-run"    (set "DO_RUN=0" & shift & goto :parse)
if /i "%~1"=="--build"     (set "DO_BUILD=1" & set "DO_RUN=0" & shift & goto :parse)
if /i "%~1"=="--installer" (set "DO_INSTALLER=1" & set "DO_RUN=0" & shift & goto :parse)
if /i "%~1"=="--help"      goto :help
shift
goto :parse

:parsed

REM Pause on exit if double-clicked (unless SCORPIPE_NO_PAUSE=1)
set "PAUSE_ON_EXIT=0"
echo %cmdcmdline% | findstr /i "/c" >nul && set "PAUSE_ON_EXIT=1"
if defined SCORPIPE_NO_PAUSE set "PAUSE_ON_EXIT=0"

REM Ensure venv
if not exist ".venv\Scripts\python.exe" (
  echo [Scorpipe] Creating .venv ...
  python -m venv .venv
  if errorlevel 1 (
    echo [ERROR] Failed to create venv. Install Python and ensure 'python' is on PATH.
    exit /b 1
  )
)

call ".venv\Scripts\activate.bat"

echo [Scorpipe] Updating pip...
python -m pip install -U pip
if errorlevel 1 exit /b 1

echo [Scorpipe] Installing Scorpipe (editable)...
pip install -e ".[science,gui]"
if errorlevel 1 exit /b 1

if "%DO_BUILD%"=="1" (
  call "scripts\windows\Build_UI_EXE.bat"
  set "EC=%ERRORLEVEL%"
  goto :done
)

if "%DO_INSTALLER%"=="1" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "packaging\windows\build.ps1"
  set "EC=%ERRORLEVEL%"
  goto :done
)

if "%DO_RUN%"=="1" (
  echo [Scorpipe] Launching GUI...
  python -m scorpio_pipe.ui.launcher_app
  set "EC=%ERRORLEVEL%"
  goto :done
)

set "EC=0"

:done
if "%PAUSE_ON_EXIT%"=="1" (
  echo.
  echo [Scorpipe] Exit code: %EC%
  echo Press any key to close...
  pause >nul
)
exit /b %EC%

:help
echo.
echo Scorpipe setup helper (run from project root):
echo   setup.bat             ^(install/update deps + run GUI^)
echo   setup.bat --build      ^(build onefile GUI EXE^)
echo   setup.bat --installer  ^(build Windows installer^)
echo   setup.bat --run        ^(run GUI^)
echo.
exit /b 0

