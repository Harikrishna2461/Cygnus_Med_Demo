@echo off
REM Quick Start Script for Clinical Medical Decision Support Demo
REM Windows Version

setlocal enabledelayedexpansion

echo 🏥 Clinical Medical Decision Support System
echo ===========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ✗ Python not found. Please install Python 3.8+
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo ✓ Python found: %PYTHON_VERSION%

REM Check Node
node --version >nul 2>&1
if errorlevel 1 (
    echo ✗ Node.js not found. Please install Node.js 16+
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('node --version') do set NODE_VERSION=%%i
echo ✓ Node.js found: %NODE_VERSION%

echo.
echo 📦 Setting up backend...

cd backend

if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat
echo Installing Python dependencies...
pip install -q -r requirements.txt

if not exist "faiss_index\medical_index.faiss" (
    echo.
    echo 📚 FAISS index not found. Ingesting medical data...
    python ingest.py
    echo ✓ Medical data ingested
) else (
    echo ✓ FAISS index found
)

cd..

echo.
echo ✓ Backend ready!
echo.
echo 📦 Setting up frontend...

cd frontend

if not exist "node_modules" (
    echo Installing Node dependencies...
    call npm install
)

cd..

echo.
echo ✓ Frontend ready!
echo.
echo ===========================================
echo ✅ Setup Complete!
echo ===========================================
echo.
echo 📖 Next steps:
echo.
echo 1. Start Backend (Command Prompt 1):
echo    cd backend
echo    venv\Scripts\activate.bat
echo    python app.py
echo.
echo 2. Start Frontend (Command Prompt 2):
echo    cd frontend
echo    npm start
echo.
echo 3. Open browser:
echo    http://localhost:3000
echo.
echo ===========================================
echo.
pause
