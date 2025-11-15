@echo off
REM Runs the PowerShell quickstart with execution policy bypass.
setlocal
set SCRIPT_DIR=%~dp0
powershell -ExecutionPolicy Bypass -File "%SCRIPT_DIR%quickstart.ps1"
endlocal
