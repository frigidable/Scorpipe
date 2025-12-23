@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ------------------------------------------------------------
REM Scorpipe setup wrapper (project root)
REM Keeps the window open when launched by double-click.
REM Set SCORPIPE_NO_PAUSE=1 to disable pausing.
REM ------------------------------------------------------------

cd /d "%~dp0"

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
