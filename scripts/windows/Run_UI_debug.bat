@echo off
REM Запуск debug версии scorpipe с консолью для отладки

setlocal EnableExtensions
set "HERE=%~dp0"
set "ROOT=%HERE%. .\..\"
pushd "%ROOT%" >nul

REM Предпочитаем встроенную debug версию
if exist "scorpipe-debug.exe" (
  echo [Scorpipe] Launching debug GUI (with console)...
  echo [Scorpipe] Errors will be printed below: 
  echo. 
  "scorpipe-debug.exe"
  exit /b %ERRORLEVEL%
)

echo [ERROR] Debug executable not found:  scorpipe-debug.exe
echo. 
echo Please run: setup. bat --build
echo This will create scorpipe-debug.exe for troubleshooting.
echo. 
pause
exit /b 1