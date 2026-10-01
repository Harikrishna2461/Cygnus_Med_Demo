@echo off
REM Starts the Vein Naming System web app. The local Qwen model server must already be running:
REM   1) double-click start_llama_server.bat   (leave that window open; wait for "model loaded")
REM   2) then run this file, and open http://localhost:7862
REM BioMedParse loads on the first job (~10-15s). Check model-server status at /api/health.
REM
REM HF_HOME defaults to hf_cache\ next to this script, where the BiomedBERT text encoder is
REM already bundled -- the first job needs no internet.
if not defined HF_HOME set HF_HOME=%~dp0hf_cache
echo Starting Vein Naming System (local Qwen edition)...
cd /d "%~dp0backend"
python3 app.py
pause
