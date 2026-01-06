@echo off
setlocal EnableExtensions
set "HERE=%~dp0"
set "ROOT=%HERE%. .\..\"
pushd "%ROOT%" >nul

REM Use the debug . exe with console output (created by Build_UI_EXE.bat)
if exist "scorpipe-debug.exe" (
  echo [Scorpipe] Launching GUI (debug mode with console)...
  start "" "scorpipe-debug. exe"
  exit /b 0
)

echo [ERROR] Debug executable not found:  scorpipe-debug.exe
echo Run:  setup.bat --build
pause
exit /b 1