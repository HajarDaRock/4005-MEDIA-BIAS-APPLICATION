@echo off
REM 
setlocal
set SCRIPT_DIR=%~dp0
powershell -ExecutionPolicy Bypass -File "%SCRIPT_DIR%quickstart.ps1"
endlocal
