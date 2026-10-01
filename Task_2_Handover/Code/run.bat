@echo off
setlocal

if not exist ".venv" call setup.bat
if errorlevel 1 exit /b 1

:: Check key
findstr /m "GROQ_API_KEY=" .env >nul 2>&1
for /f "tokens=2 delims==" %%k in ('findstr "GROQ_API_KEY=" .env') do set KEY=%%k
if "%KEY%"=="" (
    echo ERROR: GROQ_API_KEY is not set in Code\.env
    pause & exit /b 1
)

echo Starting Task 2 — Probe Localisation ^& Active Guidance ...
.venv\Scripts\python backend\app.py
endlocal
