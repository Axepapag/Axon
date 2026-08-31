@echo off
setlocal
set "AXON_ROOT=%~dp0"
start "Axon Kaggle Control Center" powershell.exe -NoExit -ExecutionPolicy Bypass -File "%AXON_ROOT%scripts\axon_kaggle_control.ps1"
endlocal
