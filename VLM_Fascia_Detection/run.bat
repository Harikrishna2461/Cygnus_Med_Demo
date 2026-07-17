@echo off
echo Starting Fascia ^& Vein Detector (Windows Native GPU)...
echo BiomedParse + LISA models will load (~30-60 s). Open http://localhost:5000 when you see "Ready."
echo.
cd /d "%~dp0"
py -3.12 app.py
pause
