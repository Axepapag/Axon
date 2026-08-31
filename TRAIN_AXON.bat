@echo off
setlocal
set "AXON_ROOT=%~dp0"
python "%AXON_ROOT%scripts\axon_trainer.py" status --state-root "%AXON_ROOT%State"
if errorlevel 1 (
  echo.
  echo Axon Trainer status failed. Review the error above.
)
echo.
echo Training mutation controls remain fail-closed until their governed handlers are connected.
pause
endlocal
