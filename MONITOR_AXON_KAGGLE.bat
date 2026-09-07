@echo off
setlocal
set "AXON_ROOT=%~dp0"
start "Axon Kaggle Live Monitor" powershell.exe -NoExit -ExecutionPolicy Bypass -File "%AXON_ROOT%scripts\axon_kaggle_monitor.ps1" %*
endlocal
