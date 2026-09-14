@echo off
echo Starting BiomedParse Vein Segmentation app...
echo Model (~1.8GB) loads on first upload, not at startup. Open http://localhost:5050 when it says "Running on".
echo.
cd /d "%~dp0backend"
py -3.12 app.py
pause
