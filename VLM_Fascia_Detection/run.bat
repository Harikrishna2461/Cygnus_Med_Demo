@echo off
echo Starting Fascia ^& Vein Detector (Windows Native GPU)...
echo Model weights download on first run (~1-2 GB).
echo Once you see "Model loaded." open http://localhost:5000 in your browser.
echo.
cd /d "%~dp0"
set HF_ENDPOINT=https://hf-mirror.com
python3 app.py
pause
