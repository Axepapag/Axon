@echo off
setlocal
cd /d D:\Axon

echo [BLOCKED] Direct-record D64 training is retired from the active Axon path.
echo.
echo Jeffrey's canonical-anatomy boundary requires training to run through a
echo State\training branch using the same canonical field/compiler/delta interfaces
echo as runtime. The preserved V6 JSON-record trainer remains available only for
echo explicit legacy mechanism work via --legacy-record-direct.
echo.
echo No model, checkpoint, curriculum, or State will be changed by this launcher.
echo See docs\CANONICAL_STATE_RECONCILIATION.md.
exit /b 3
