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
set "AXON_PY=C:\Users\axema\AppData\Local\Programs\Python\Python312\python.exe"
if not exist "%AXON_PY%" set "AXON_PY=python"
echo Axon Training Watch - live loss/accuracy/Q-A dashboard
echo Job: %AXON_JOB%
echo Closing this window never stops Kaggle training.
echo.
"%AXON_PY%" "%AXON_ROOT%scripts\axon_training_watch.py" %AXON_JOB% --qa
echo.
echo The dashboard stream ended. The cloud job itself may still be running.
pause
endlocal