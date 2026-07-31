@echo off
echo Starting Vein Name Classification System...
echo Loads on first job submission (BioMedParse ~10-15s). Open http://localhost:7862 when ready.
echo.
cd /d "%~dp0backend"
python3 app.py
pause
