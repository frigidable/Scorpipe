@echo off
setlocal EnableExtensions
call "%~dp0..\..\setup_debug.bat" %*
exit /b %ERRORLEVEL%
