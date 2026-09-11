@echo off
setlocal
cd /d "%~dp0"

if not exist "GRIM_Backend\run_diagnostics.py" (
    echo ERROR: GRIM_Backend\run_diagnostics.py was not found.
    echo Keep the complete combined GRIM folder structure together.
    pause
    exit /b 1
)

if exist "%~dp0.venv\Scripts\python.exe" (
    set "GRIM_DIAG_PYTHON=%~dp0.venv\Scripts\python.exe"
    goto run_python
)

if defined VIRTUAL_ENV if exist "%VIRTUAL_ENV%\Scripts\python.exe" (
    set "GRIM_DIAG_PYTHON=%VIRTUAL_ENV%\Scripts\python.exe"
    goto run_python
)

where py.exe >nul 2>&1
if not errorlevel 1 (
    py.exe -3 -c "import sys" >nul 2>&1
    if not errorlevel 1 goto run_py
)

where python.exe >nul 2>&1
if not errorlevel 1 (
    set "GRIM_DIAG_PYTHON=python.exe"
    goto run_python
)

echo ERROR: Python 3 was not found.
echo Create the shared environment at %~dp0.venv or install Python 3.10 or newer.
pause
exit /b 1

:run_python
"%GRIM_DIAG_PYTHON%" "%~dp0GRIM_Backend\run_diagnostics.py"
set "GRIM_DIAG_STATUS=%ERRORLEVEL%"
goto finished

:run_py
py.exe -3 "%~dp0GRIM_Backend\run_diagnostics.py"
set "GRIM_DIAG_STATUS=%ERRORLEVEL%"

:finished
echo.
if "%GRIM_DIAG_STATUS%"=="0" (
    echo GRIM has no required startup blockers.
) else (
    echo GRIM has one or more required startup blockers listed above.
)
pause
exit /b %GRIM_DIAG_STATUS%
