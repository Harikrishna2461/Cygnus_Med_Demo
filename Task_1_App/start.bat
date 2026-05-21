@echo off
setlocal EnableDelayedExpansion
title CHIVA Clinical Assistant

echo.
echo  ================================================
echo   CHIVA Clinical Assistant  ^|  Starting...
echo  ================================================
echo.

:: Locate backend directory relative to this script
set "SCRIPT_DIR=%~dp0"
set "BACKEND_DIR=%SCRIPT_DIR%backend"

:: Verify Python
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10+ and add it to PATH.
    pause
    exit /b 1
)

:: Check Python version
for /f "tokens=2" %%V in ('python --version 2^>^&1') do set "PYVER=%%V"
echo [OK] Python %PYVER% found

:: Verify Ollama is running
curl -s -o nul -w "%%{http_code}" http://localhost:11434/api/tags 2>nul | find "200" >nul
if errorlevel 1 (
    echo [WARN] Ollama does not appear to be running on localhost:11434.
    echo        Start Ollama and ensure 'nomic-embed-text' model is pulled:
    echo          ollama serve
    echo          ollama pull nomic-embed-text
    echo.
    echo        The app will start anyway — embeddings will fail until Ollama is running.
    echo.
)

:: Install / verify dependencies
echo [INFO] Checking Python dependencies...
python -m pip install -r "%BACKEND_DIR%\requirements.txt" --quiet --disable-pip-version-check
if errorlevel 1 (
    echo [ERROR] Dependency installation failed. Run manually:
    echo         pip install -r backend\requirements.txt
    pause
    exit /b 1
)
echo [OK] Dependencies verified.
echo.

:: Copy core module from parent project if not present (shipping scenario)
set "CORE_MODULE=%BACKEND_DIR%\shunt_classification_and_ligation_llm.py"
set "PARENT_MODULE=%SCRIPT_DIR%..\backend\shunt_classification_and_ligation_llm.py"
if not exist "%CORE_MODULE%" (
    if exist "%PARENT_MODULE%" (
        echo [INFO] Copying core module from parent backend...
        copy /Y "%PARENT_MODULE%" "%CORE_MODULE%" >nul
        echo [OK] Core module copied.
    ) else (
        echo [WARN] shunt_classification_and_ligation_llm.py not found.
        echo        Place it in the backend\ folder to enable classification.
    )
)

:: Start the Flask app
echo [INFO] Starting CHIVA Clinical Assistant on http://127.0.0.1:7860
echo.
echo  Open your browser at: http://127.0.0.1:7860
echo  Press Ctrl+C to stop the server.
echo.

cd /d "%BACKEND_DIR%"
python app.py

pause
