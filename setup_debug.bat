@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"
set "LOG=%~dp0setup_debug.log"

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
