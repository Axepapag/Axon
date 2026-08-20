@echo off
REM Axon Council runtime launcher (see runtime\council\CONTRACT.md)
REM Double-click to start the council server, then open http://127.0.0.1:8788
REM Runs in its own window; independent of any assistant session.
title Axon Council Server
cd /d D:\Axon\runtime\council
"C:\Users\axema\AppData\Local\Programs\Python\Python312\python.exe" -m uvicorn server:app --host 127.0.0.1 --port 8788
pause
