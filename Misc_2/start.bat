@echo off
REM CHIVA Clinical Assistant — Automated Startup Script
REM This script checks for Python, installs dependencies, and starts the app

echo.
echo ================================================================
echo  CHIVA Clinical Assistant — Starting Up
echo ================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo.
    echo Please install Python from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    echo.
    pause
    exit /b 1
)

echo [OK] Python found
python --version
echo.

REM Check pip is available
pip --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] pip is not available
    echo Please ensure Python was installed correctly with pip support
    pause
    exit /b 1
)

echo [OK] pip is available
echo.

REM Navigate to the backend directory
cd /d "%~dp0cmed_demo\backend"
if errorlevel 1 (
    echo [ERROR] Failed to navigate to backend directory
    pause
    exit /b 1
)

echo [INFO] Current directory: %CD%
echo.

REM Install dependencies
echo [INFO] Installing required Python libraries...
echo This may take a few minutes on first run...
echo.

pip install -q -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    echo Please check your internet connection and try again
    pause
    exit /b 1
)

echo [OK] All dependencies installed successfully
echo.

REM Check if Qdrant collection exists
if not exist "qdrant_storage" (
    echo [WARN] Qdrant vector database not found at: %CD%\qdrant_storage
    echo [INFO] The app will create it on first use, or you may need to ingest knowledge base
    echo.
)

REM Start the Flask app
echo ================================================================
echo [INFO] Starting CHIVA Clinical Assistant...
echo ================================================================
echo.
echo The app will be available at: http://localhost:5000
echo Press Ctrl+C to stop the app
echo.

python app.py

REM If the app exits unexpectedly, show an error message
if errorlevel 1 (
    echo.
    echo [ERROR] The app exited unexpectedly
    echo Please check the error messages above
    echo.
    pause
    exit /b 1
)
