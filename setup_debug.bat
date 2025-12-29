@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"
set "LOG=%~dp0setup_debug.log"

echo ============================================================ > "%LOG%"
echo [Scorpipe] setup_debug.bat started at %DATE% %TIME% >> "%LOG%"
echo ============================================================ >> "%LOG%"
echo. >> "%LOG%"

call "%~dp0setup.bat" %* >> "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"

echo.>> "%LOG%"
echo [Scorpipe] Exit code: %EC% >> "%LOG%"

echo.
echo [Scorpipe] Exit code: %EC%
echo Log saved to: "%LOG%"
echo.
findstr /n "." "%LOG%"
echo.
pause
exit /b %EC%

