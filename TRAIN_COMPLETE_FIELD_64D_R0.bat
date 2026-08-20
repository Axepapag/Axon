@echo off
setlocal
cd /d D:\axon
set "AXON_PY=C:\Users\axema\AppData\Local\Programs\Python\Python312\python.exe"
set "AXON_DATA=State\private_curriculum\complete_field_r0_v6"
set "AXON_ALIGN=State\private_curriculum\complete_field_r0_v6_alignment"
set "AXON_RUN=runs\complete_field_64d_r0_v6_alignment_smoke"

rem V6 is intentionally bounded. There is no 1k, 5k, or 200k continuation in
rem this launcher. Exact held-out alignment/retrieval must pass first.
"%AXON_PY%" -m training.build_complete_field_r0_curriculum ^
  --output-dir "%AXON_DATA%" ^
  --v6-alignment 256 ^
  --alignment-page-size 64
if errorlevel 1 exit /b %errorlevel%

"%AXON_PY%" -m training.build_r0_v6_alignment_shard ^
  --output-dir "%AXON_ALIGN%" ^
  --count 256 ^
  --seed 70024 ^
  --page-size 64
if errorlevel 1 exit /b %errorlevel%

"%AXON_PY%" -m pytest -q tests\test_complete_field_64d.py
if errorlevel 1 exit /b %errorlevel%

"%AXON_PY%" training\train_complete_field_64d.py ^
  --train "%AXON_ALIGN%\train.jsonl" ^
  --eval "%AXON_ALIGN%\dev.jsonl" ^
  --run-dir "%AXON_RUN%" ^
  --device cuda ^
  --steps 100 ^
  --lr 2e-4 ^
  --page-size 64 ^
  --max-output-chars 128 ^
  --alignment-weight 1.0 ^
  --copy-gate-weight 0.5 ^
  --causal-weight 0.5 ^
  --causal-every 1 ^
  --teacher-forcing-ratio 0.75 ^
  --checkpoint-every 100 ^
  --sample-every 50 ^
  --eval-every 50 ^
  --eval-examples 18 ^
  --v6-eval-examples 18 ^
  --counterfactual-sample-count 18 ^
  --keep-checkpoints 3

rem Exit 0 only when the 100-step smoke itself passes every hard V6 gate.
if not exist "%AXON_RUN%\gate.json" exit /b 2
"%AXON_PY%" -c "import json,pathlib,sys; g=json.loads(pathlib.Path(r'%AXON_RUN%\gate.json').read_text(encoding='utf-8')); sys.exit(0 if g.get('promotion_allowed') is True and g.get('v6_alignment_gate_passed') is True and g.get('step') == 100 and g.get('target_step') == 100 else 2)"
if errorlevel 1 exit /b 2

endlocal
