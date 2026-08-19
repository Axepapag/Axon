@echo off
REM Axon 64D conversational trainer - Bible council mix, GPU.
REM Double-click to resume the overnight run for another 20,000 steps.
REM Checkpoints land in D:\Axon\runs\bible_64D_gpu_overnight (last 10 kept).
title Axon 64D Training
cd /d D:\Axon
"C:\Users\axema\AppData\Local\Programs\Python\Python312\python.exe" training\conversational_cpu_trainer.py --checkpoint runs\bible_64D_gpu_overnight\ckpt_461500.pt --curriculum datasets\bible\bible_council_mix.jsonl --run-dir runs\bible_64D_gpu_overnight --device cuda --resume --steps 20000 --lr 1e-4 --eval-every 1000 --max-eval-examples 100 --checkpoint-every 250 --keep-checkpoints 10 --sample-every 50
pause
