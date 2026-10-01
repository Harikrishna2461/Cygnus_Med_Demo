@echo off
setlocal

echo === Task 2 Setup ===

:: Find Python 3.10-3.13
set PY=
for %%v in (3.13 3.12 3.11 3.10) do (
    if not defined PY (
        py -%%v --version >nul 2>&1 && set PY=py -%%v
    )
)
if not defined PY (
    echo ERROR: Python 3.10-3.13 not found. Install from python.org.
    pause & exit /b 1
)
echo Using: %PY%

:: Create venv
if not exist ".venv" (
    echo Creating .venv ...
    %PY% -m venv .venv
)

:: Install dependencies
echo Installing requirements (this takes a few minutes on first run) ...
.venv\Scripts\pip install --quiet -r requirements.txt
if errorlevel 1 ( echo ERROR: pip install failed. & pause & exit /b 1 )

:: Create .env from example
if not exist ".env" (
    if exist ".env.example" (
        copy .env.example .env >nul
        echo Created .env from .env.example — add your GROQ_API_KEY to Code\.env
    )
)

echo Setup complete.
endlocal
