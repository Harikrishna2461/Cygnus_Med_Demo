@echo off
setlocal
set ARG=%~1

if not exist ".venv" call setup.bat
if errorlevel 1 exit /b 1

if "%ARG%"=="" (
    echo Running crew smoke test ...
    .venv\Scripts\python -m pytest tests\test_crew_pipeline.py -v
    goto end
)
if "%ARG%"=="scenarios" (
    .venv\Scripts\python tests\run_scenarios.py
    goto end
)
if "%ARG%"=="scenarios_v2" (
    .venv\Scripts\python tests\run_scenarios_v2.py
    goto end
)
if "%ARG%"=="scenarios_june" (
    .venv\Scripts\python tests\test_scenarios_29_June.py
    goto end
)
echo Usage: run_tests.bat [scenarios ^| scenarios_v2 ^| scenarios_june]

:end
endlocal
