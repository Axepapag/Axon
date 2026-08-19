@echo off
setlocal
cd /d D:\axon
set "AXON_PY=C:\Users\axema\AppData\Local\Programs\Python\Python312\python.exe"
set "AXON_DATA=State\private_curriculum\complete_field_r0"
set "AXON_RUN=runs\complete_field_64d_r0_200k"

if not exist "%AXON_DATA%\train.jsonl" (
  "%AXON_PY%" training\build_complete_field_r0_curriculum.py --output-dir "%AXON_DATA%"
  if errorlevel 1 exit /b %errorlevel%
)

"%AXON_PY%" training\train_complete_field_64d.py ^
  --train "%AXON_DATA%\train.jsonl" ^
  --eval "%AXON_DATA%\dev.jsonl" ^
  --run-dir "%AXON_RUN%" ^
  --device cuda ^
  --steps 200000 ^
  --lr 2e-4 ^
  --page-size 256 ^
  --max-output-chars 512 ^
  --checkpoint-every 250 ^
  --sample-every 250 ^
  --eval-every 1000 ^
  --keep-checkpoints 3 ^
  --resume

endlocal
