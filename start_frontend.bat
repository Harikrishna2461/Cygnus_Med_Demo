@echo off
REM CHIVA Clinical Assistant — Frontend Startup Script
REM This script starts the React development server

echo.
echo ================================================================
echo  CHIVA Clinical Assistant — Frontend Server
echo ================================================================
echo.

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js is not installed or not in PATH
    echo.
    echo Please install Node.js from: https://nodejs.org/
    echo Download the LTS version and follow the installer
    echo.
    pause
    exit /b 1
)

echo [OK] Node.js found
node --version
echo.

REM Check npm is available
npm --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] npm is not available
    echo Please ensure Node.js was installed correctly with npm support
    pause
    exit /b 1
)

echo [OK] npm is available
echo.

REM Navigate to the frontend directory
cd /d "%~dp0cmed_demo\frontend"
if errorlevel 1 (
    echo [ERROR] Failed to navigate to frontend directory
    pause
    exit /b 1
)

echo [INFO] Current directory: %CD%
echo.

REM Check if node_modules exists
if not exist "node_modules" (
    echo [INFO] Installing npm dependencies...
    echo This may take a minute on first run...
    echo.
    call npm install
    if errorlevel 1 (
        echo [ERROR] Failed to install npm dependencies
        pause
        exit /b 1
    )
    echo [OK] npm dependencies installed
    echo.
)

REM Start the development server
echo ================================================================
echo [INFO] Starting React development server...
echo ================================================================
echo.
echo The frontend will be available at: http://localhost:3000
echo The backend should be running at: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.

call npm start

REM If the server exits unexpectedly, show an error message
if errorlevel 1 (
    echo.
    echo [ERROR] The frontend server exited unexpectedly
    echo Please check the error messages above
    echo.
    pause
    exit /b 1
)
