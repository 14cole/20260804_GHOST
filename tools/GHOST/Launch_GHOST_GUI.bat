@echo off
setlocal
cd /d "%~dp0"
set "GHOST_GUI=%~dp0ghost_backend\run_gui.py"

if not exist "%GHOST_GUI%" (
    echo ERROR: %GHOST_GUI% was not found.
    pause
    exit /b 1
)

for %%I in ("%~dp0..\..") do set "GRIM_REPO_ROOT=%%~fI"
set "GHOST_LAUNCH_LOG=%TEMP%\ghost-gui-launch.log"
type nul >"%GHOST_LAUNCH_LOG%"

if exist "%GRIM_REPO_ROOT%\.venv\Scripts\python.exe" (
    call :try_python "%GRIM_REPO_ROOT%\.venv\Scripts\python.exe"
    if not errorlevel 1 goto launch_python
)

if defined VIRTUAL_ENV if exist "%VIRTUAL_ENV%\Scripts\python.exe" (
    call :try_python "%VIRTUAL_ENV%\Scripts\python.exe"
    if not errorlevel 1 goto launch_python
)

where py.exe >nul 2>&1
if not errorlevel 1 (
    echo --- py.exe -3 --- >>"%GHOST_LAUNCH_LOG%"
    py.exe -3 "%GHOST_GUI%" --check >>"%GHOST_LAUNCH_LOG%" 2>&1
    if not errorlevel 1 goto launch_py
)

where python.exe >nul 2>&1
if not errorlevel 1 (
    call :try_python "python.exe"
    if not errorlevel 1 goto launch_python
)

goto missing_dependencies

:try_python
set "GHOST_PYTHON=%~1"
echo --- %GHOST_PYTHON% --- >>"%GHOST_LAUNCH_LOG%"
"%GHOST_PYTHON%" "%GHOST_GUI%" --check >>"%GHOST_LAUNCH_LOG%" 2>&1
exit /b %ERRORLEVEL%

:launch_python
for %%I in ("%GHOST_PYTHON%") do set "GHOST_PYTHONW=%%~dpIpythonw.exe"
if exist "%GHOST_PYTHONW%" (
    start "" "%GHOST_PYTHONW%" "%GHOST_GUI%"
) else (
    start "" "%GHOST_PYTHON%" "%GHOST_GUI%"
)
exit /b 0

:launch_py
where pyw.exe >nul 2>&1
if errorlevel 1 (
    start "" py.exe -3 "%GHOST_GUI%"
) else (
    start "" pyw.exe -3 "%GHOST_GUI%"
)
exit /b 0

:missing_dependencies
echo ERROR: No preferred Python interpreter could import the GHOST GUI.
echo.
if exist "%GHOST_LAUNCH_LOG%" type "%GHOST_LAUNCH_LOG%"
echo.
echo From the repository root, create the shared environment and install with:
echo     py.exe -3 -m venv .venv
echo     .venv\Scripts\python.exe -m pip install -e .
echo.
pause
exit /b 1
