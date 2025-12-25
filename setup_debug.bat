@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"
set "LOG=%~dp0setup_debug.log"

if exist "%~dp0setup.exe" (
  echo [Scorpipe] Found setup.exe, launching...
  start "" "%~dp0setup.exe"
  exit /b 0
)

if not defined SCORPIPE_DEV_SETUP (
  echo.
  echo [Scorpipe] setup_debug.bat is a developer helper.
  echo [Scorpipe] For normal installation, download and run setup.exe from GitHub Releases.
  echo [Scorpipe] If you know what you are doing: set SCORPIPE_DEV_SETUP=1 and re-run.
  echo.
  pause
  exit /b 0
)

echo ============================================================ > "%LOG%"
echo [Scorpipe] Setup debug log                             >> "%LOG%"
echo Date: %DATE%  Time: %TIME%                              >> "%LOG%"
echo CWD:  %CD%                                              >> "%LOG%"
echo ============================================================ >> "%LOG%"

call "%~dp0scripts\windows\setup.bat" %* >> "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"

echo.>> "%LOG%"
echo [Scorpipe] Exit code: %EC% >> "%LOG%"

echo.
findstr /n "." "%LOG%"
echo.
echo [Scorpipe] Exit code: %EC%
echo Log saved to: "%LOG%"
pause
exit /b %EC%
