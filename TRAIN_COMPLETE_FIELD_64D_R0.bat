@echo off
setlocal
cd /d D:\axon
set "AXON_PY=C:\Users\axema\AppData\Local\Programs\Python\Python312\python.exe"
set "AXON_DATA=State\private_curriculum\complete_field_r0"
set "AXON_RUN=runs\complete_field_64d_r0_counterfactual_200k"

"%AXON_PY%" training\build_complete_field_r0_curriculum.py --output-dir "%AXON_DATA%"
if errorlevel 1 exit /b %errorlevel%

rem Stage 1 is a hard promotion gate. A failed gate exits before the 200k continuation.
"%AXON_PY%" training\train_complete_field_64d.py ^
  --train "%AXON_DATA%\train.jsonl" ^
  --eval "%AXON_DATA%\dev.jsonl" ^
  --run-dir "%AXON_RUN%" ^
  --device cuda ^
  --steps 5000 ^
  --lr 2e-4 ^
  --page-size 256 ^
  --max-output-chars 512 ^
  --causal-weight 0.5 ^
  --causal-every 1 ^
  --teacher-forcing-ratio 0.75 ^
  --checkpoint-every 250 ^
  --sample-every 250 ^
  --eval-every 1000 ^
  --keep-checkpoints 3 ^
  --resume-if-available
if not exist "%AXON_RUN%\gate.json" exit /b 2
"%AXON_PY%" -c "import json,pathlib,sys; g=json.loads(pathlib.Path(r'%AXON_RUN%\gate.json').read_text(encoding='utf-8')); sys.exit(0 if g.get('promotion_allowed') is True and g.get('step') == 5000 and g.get('target_step') == 5000 else 2)"
if errorlevel 1 exit /b 2

rem Only a passing 5k gate reaches this continuation. Resume preserves model,
rem optimizer, scaler, global RNG, CUDA RNG, and sampler RNG state.
"%AXON_PY%" training\train_complete_field_64d.py ^
  --train "%AXON_DATA%\train.jsonl" ^
  --eval "%AXON_DATA%\dev.jsonl" ^
  --run-dir "%AXON_RUN%" ^
  --device cuda ^
  --steps 200000 ^
  --lr 2e-4 ^
  --page-size 256 ^
  --max-output-chars 512 ^
  --causal-weight 0.5 ^
  --causal-every 1 ^
  --teacher-forcing-ratio 0.75 ^
  --checkpoint-every 250 ^
  --sample-every 250 ^
  --eval-every 1000 ^
  --keep-checkpoints 3 ^
  --resume

endlocal
