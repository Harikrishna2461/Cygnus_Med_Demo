@echo off
setlocal EnableDelayedExpansion
title CHIVA Assistant — Build

echo.
echo  ============================================================
echo   CHIVA Clinical Assistant  ^|  Building .exe
echo  ============================================================
echo.

:: ── Verify Python ─────────────────────────────────────────────────────────────
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Add Python to PATH and retry.
    goto :fail
)
for /f "tokens=2" %%V in ('python --version 2^>^&1') do set PYVER=%%V
echo [OK] Python %PYVER%

:: ── Kill any python processes holding the Qdrant .lock file ───────────────────
echo [INFO] Stopping any running Python processes...
taskkill /F /IM python.exe >nul 2>&1
taskkill /F /IM python3.exe >nul 2>&1
taskkill /F /IM pythonw.exe >nul 2>&1
timeout /t 2 /nobreak >nul
echo [OK] Done.

:: ── Install build dependencies ────────────────────────────────────────────────
echo [INFO] Installing build dependencies...
python -m pip install pyinstaller PySide6 --quiet --disable-pip-version-check >> build_log.txt 2>&1
if errorlevel 1 (
    echo [ERROR] Failed to install build dependencies.
    echo         Details in build_log.txt
    goto :fail
)
echo [OK] Build deps ready.
echo.

:: ── Clean previous build ──────────────────────────────────────────────────────
echo [INFO] Cleaning previous build...
if exist build  rmdir /s /q build  >nul 2>&1
if exist dist   rmdir /s /q dist   >nul 2>&1
if exist CHIVA_Assistant.spec del /q CHIVA_Assistant.spec >nul 2>&1
echo [OK] Clean done.
echo.

:: ── Delete the Qdrant .lock file so PyInstaller can copy qdrant_storage ───────
if exist "backend\qdrant_storage\.lock" (
    del /f /q "backend\qdrant_storage\.lock" >nul 2>&1
    echo [INFO] Removed qdrant_storage\.lock
)

:: ── Run PyInstaller (output goes to console AND build_log.txt) ────────────────
echo [INFO] Running PyInstaller — this takes a few minutes...
echo        Follow along in build_log.txt if needed.
echo.

python -m PyInstaller ^
    --name "CHIVA_Assistant" ^
    --onedir ^
    --windowed ^
    --noconfirm ^
    --add-data "backend\qdrant_storage;qdrant_storage" ^
    --add-data "frontend;frontend" ^
    --paths "backend" ^
    --hidden-import "app" ^
    --hidden-import "services" ^
    --hidden-import "config" ^
    --hidden-import "chat_db" ^
    --hidden-import "rag_engine" ^
    --hidden-import "general_chat_engine" ^
    --hidden-import "nl_interpreter" ^
    --hidden-import "shunt_classification_and_ligation_llm" ^
    --hidden-import "routes" ^
    --hidden-import "routes.views" ^
    --hidden-import "routes.status" ^
    --hidden-import "routes.sessions" ^
    --hidden-import "routes.clinical" ^
    --hidden-import "routes.general" ^
    --hidden-import "routes.feedback" ^
    --hidden-import "flask_cors" ^
    --hidden-import "groq" ^
    --hidden-import "qdrant_client" ^
    --hidden-import "qdrant_client.http" ^
    --hidden-import "qdrant_client.http.models" ^
    --hidden-import "qdrant_client.local.local_collection" ^
    --hidden-import "rank_bm25" ^
    --hidden-import "sentence_transformers" ^
    --hidden-import "sklearn" ^
    --hidden-import "sklearn.metrics.pairwise" ^
    --collect-all "sentence_transformers" ^
    --collect-all "qdrant_client" ^
    --collect-all "groq" ^
    --collect-all "certifi" ^
    --collect-all "PySide6" ^
    --exclude-module "matplotlib" ^
    --exclude-module "scipy" ^
    --exclude-module "IPython" ^
    --exclude-module "tkinter" ^
    --exclude-module "PIL" ^
    --exclude-module "cv2" ^
    --exclude-module "PyQt5" ^
    --exclude-module "PyQt6" ^
    launcher.py >> build_log.txt 2>&1

if errorlevel 1 goto :pyinstaller_fail

:: ── Move exe + _internal to root (so the exe is right here, not buried in dist\) ─
echo [INFO] Moving executable to project root...
if exist "CHIVA_Assistant.exe" del /f /q "CHIVA_Assistant.exe"
if exist "_internal" rmdir /s /q "_internal"
move /Y "dist\CHIVA_Assistant\CHIVA_Assistant.exe" "CHIVA_Assistant.exe" >nul
move /Y "dist\CHIVA_Assistant\_internal" "_internal" >nul
rmdir /s /q "dist" >nul 2>&1
rmdir /s /q "build" >nul 2>&1
if exist "CHIVA_Assistant.spec" move /Y "CHIVA_Assistant.spec" "Miscellaneous\build_artifacts\CHIVA_Assistant.spec" >nul 2>&1
if exist "build_log.txt" move /Y "build_log.txt" "Miscellaneous\logs\build_log.txt" >nul 2>&1

:: ── Success ───────────────────────────────────────────────────────────────────
echo.
echo  ============================================================
echo   Build complete!
echo.
echo   Executable: CHIVA_Assistant.exe  (right here in this folder)
echo.
echo   To distribute: zip this entire folder and send it.
echo   Recipient unzips and double-clicks CHIVA_Assistant.exe
echo.
echo   NOTE: First launch downloads the cross-encoder model once
echo   (~80 MB). After that runs fully offline.
echo  ============================================================
echo.
pause
exit /b 0

:: ── PyInstaller failed ────────────────────────────────────────────────────────
:pyinstaller_fail
echo.
echo [ERROR] PyInstaller failed. Last 60 lines of build_log.txt:
echo.
python -c "lines=open('build_log.txt',errors='replace').readlines(); [print(l,end='') for l in lines[-60:]]"
echo.
echo Full log: %CD%\build_log.txt
goto :fail

:fail
echo.
echo Build did not complete. See error above or open build_log.txt.
echo.
pause
exit /b 1
