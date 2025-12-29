@echo off
setlocal EnableExtensions
call "%~dp0..\..\setup.bat" %*
exit /b %ERRORLEVEL%
