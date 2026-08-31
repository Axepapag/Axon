@echo off
setlocal
set "AXON_ROOT=%~dp0"
set "AXON_JOB=%~1"
if not defined AXON_JOB set /p "AXON_JOB=Paste the Axon cloud job ID: "
if not defined AXON_JOB (
  echo A job ID is required.
  pause
  exit /b 2
)
start "Axon Kaggle Live Monitor" powershell.exe -NoExit -ExecutionPolicy Bypass -File "%AXON_ROOT%scripts\axon_kaggle_monitor.ps1" -JobId "%AXON_JOB%"
endlocal
