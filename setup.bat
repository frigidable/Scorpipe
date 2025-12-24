@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ------------------------------------------------------------
REM Scorpipe setup wrapper (project root)
REM
REM Normal users should install via setup.exe from GitHub Releases.
REM This .bat is kept as a developer helper.
REM
REM - If setup.exe is present next to this file, it is launched.
REM - Otherwise, set SCORPIPE_DEV_SETUP=1 to run the dev installer.
REM ------------------------------------------------------------

cd /d "%~dp0"

if exist "%~dp0setup.exe" (
  echo [Scorpipe] Found setup.exe, launching...
  start "" "%~dp0setup.exe"
  exit /b 0
)

if not defined SCORPIPE_DEV_SETUP (
  echo.
  echo [Scorpipe] This setup.bat is a developer helper.
  echo [Scorpipe] For normal installation, download and run setup.exe from GitHub Releases.
  echo [Scorpipe] If you know what you are doing: set SCORPIPE_DEV_SETUP=1 and re-run.
  echo.
  pause
  exit /b 0
)

set "PAUSE_ON_EXIT=0"
if defined cmdcmdline (
  echo %cmdcmdline% | findstr /i "/c" >nul && set "PAUSE_ON_EXIT=1"
)
if defined SCORPIPE_NO_PAUSE set "PAUSE_ON_EXIT=0"

call "%~dp0scripts\windows\setup.bat" %*
set "EC=%ERRORLEVEL%"

if "%PAUSE_ON_EXIT%"=="1" (
  echo.
  echo [Scorpipe] Exit code: %EC%
  echo Press any key to close...
  pause >nul
)

exit /b %EC%
